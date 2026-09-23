"""Batch translation to English via the configured LLM backend.

By the time a chunk reaches this module its start_sec/end_sec are already final
(chunker.py ran first, on original-language segments). Only `text` changes here,
never a timestamp.

Translation is cached on disk by source-text hash + provider + model, so repeated local experiments do not
spend API calls again. The old daily chunk budget remains available as a configurable safety valve, but local mode
sets it to unlimited (0).

Chunks are sent in BATCHES (several per request). A batch is also capped by characters
(TRANSLATE_BATCH_MAX_CHARS) so one request stays comfortably below the provider's context/output limits. The
reply must return each chunk under its own <<<CHUNK N>>> marker. If it does not (or a chunk comes back
suspiciously short, which is what a truncated reply looks like), the batch is split once and, failing that,
translated chunk by chunk: slower, never wrong.

Batches run with up to TRANSLATE_MAX_CONCURRENCY requests in flight at once (see config.py for why this is safe
alongside "keep transcript fetching sequential", which is about a different service). Everything else about one
video stays sequential: videos in a playlist, and every step of the pipeline around this one.
"""
import concurrent.futures
import logging
import re
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from backend import config
from backend.ingest import cache
from backend.models import Chunk
from backend.util import QuotaExhausted, _chat, current_notifier, format_duration, set_notifier

log = logging.getLogger(__name__)

_MARKER = re.compile(r"<<<\s*CHUNK\s+(\d+)\s*>>>", re.IGNORECASE)
_EMPHASIS = re.compile(r"^\**|\**$")  # a model sometimes bolds the marker line: **<<<CHUNK 1>>>**

_SYSTEM = (
    "You translate spoken-video transcripts into natural, fluent English.\n"
    "Rules:\n"
    "- Keep technical terms and English loanwords exactly as spoken (for example: function, hashmap, "
    "recursion, array, API). Never translate, transliterate or alter them.\n"
    "- Translate naturally, preserving meaning, tone and idiom. Do not summarize, add or omit content.\n"
    "- Output only the translation. No notes, no commentary."
)
_BATCH_SYSTEM = (
    _SYSTEM
    + "\n- The input holds several passages, each introduced by a marker line such as <<<CHUNK 1>>>. "
    "Return every passage translated, each introduced by the identical marker line, in the same order."
)

# --- daily budget (in-memory; resets at UTC midnight or on restart) ---------------------
_budget = {"day": None, "used": 0}
_budget_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _roll_day() -> None:
    today = _utc_now().date()
    if _budget["day"] != today:
        _budget["day"], _budget["used"] = today, 0


def _seconds_until_reset() -> int:
    now = _utc_now()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((midnight - now).total_seconds())


def _check_budget(n_chunks: int) -> None:
    budget = config.TRANSLATION_DAILY_CHUNK_BUDGET
    if budget <= 0:
        return
    with _budget_lock:
        _roll_day()
        # A video is translated whole (no partial checkpoints), so on a fresh day one is let through
        # even if it alone is bigger than the budget. Otherwise it could never be indexed.
        if _budget["used"] > 0 and _budget["used"] + n_chunks > budget:
            wait = _seconds_until_reset()
            raise QuotaExhausted(
                f"Daily translation budget reached ({_budget['used']} of {budget} chunks used today). "
                f"It resets in about {format_duration(wait)}, or raise TRANSLATION_DAILY_CHUNK_BUDGET.",
                wait,
            )


def _record(n_chunks: int) -> None:
    with _budget_lock:
        _roll_day()
        _budget["used"] += n_chunks


# --- translation ---------------------------------------------------------------------------
def translate_chunks(chunks: list[Chunk], on_progress=None) -> list[Chunk]:
    """Translate chunks to English unless the transcript already was English.

    `on_progress(done, total)` is called as each batch finishes (done counts chunks translated
    so far, not necessarily in original order — see the concurrency note below). Raises
    QuotaExhausted if the daily budget or the provider's quota is used up; the first such error
    from any in-flight batch is the one that propagates, so the video is correctly NOT marked
    indexed (spec §18, §31) even though other batches may have already spent quota.
    """
    if not chunks or chunks[0].source_language.lower().replace("_", "-").split("-")[0] == "en":
        return chunks

    cache_key = cache.translation_cache_key(chunks, config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL)
    cached = cache.load_translation(cache_key, len(chunks))
    if cached is not None:
        log.info(
            "translation cache hit: %d chunks using %s/%s",
            len(chunks), config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL,
        )
        if on_progress:
            on_progress(len(chunks), len(chunks))
        return [replace(c, text=t) for c, t in zip(chunks, cached)]

    _check_budget(len(chunks))
    total = len(chunks)
    batches = _make_batches(chunks)
    workers = max(1, min(config.TRANSLATE_MAX_CONCURRENCY, len(batches)))

    results: list[list[Chunk] | None] = [None] * len(batches)
    translated = 0
    progress_lock = threading.Lock()
    # ThreadPoolExecutor does NOT copy context vars into its worker threads (that's an asyncio
    # behavior, not a general threading one), so the rate-limit notifier has to be read here, in
    # the calling thread, and re-registered inside each worker — otherwise a mid-translation 429
    # would still be handled correctly, but its "waiting Ns, then retrying..." message would
    # silently never reach the UI once this runs inside our own pool.
    notifier = current_notifier()

    def run_one(index: int, batch: list[Chunk]) -> None:
        nonlocal translated
        if notifier is not None:
            set_notifier(notifier)
        texts = _translate_batch([c.text for c in batch], batch[0].source_language)
        _record(len(batch))  # spent, whether or not another in-flight batch later fails
        results[index] = [replace(c, text=t) for c, t in zip(batch, texts)]
        if on_progress:
            with progress_lock:
                translated += len(batch)
                on_progress(translated, total)

    if workers == 1:
        for i, batch in enumerate(batches):
            run_one(i, batch)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, i, batch) for i, batch in enumerate(batches)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raises the first batch's exception, if any; others keep running

    out: list[Chunk] = []
    for batch_result in results:
        out.extend(batch_result)
    cache.save_translation(
        cache_key, config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL, [chunk.text for chunk in out]
    )
    return out


def _make_batches(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Consecutive chunks, up to TRANSLATE_BATCH_SIZE per batch and TRANSLATE_BATCH_MAX_CHARS of text (a batch
    always holds at least one chunk, however long). Whichever cap is reached first closes the batch."""
    size = max(1, config.TRANSLATE_BATCH_SIZE)
    cap = config.TRANSLATE_BATCH_MAX_CHARS
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    used = 0
    for chunk in chunks:
        if current and (len(current) >= size or (cap > 0 and used + len(chunk.text) > cap)):
            batches.append(current)
            current, used = [], 0
        current.append(chunk)
        used += len(chunk.text)
    if current:
        batches.append(current)
    return batches


def _translate_batch(texts: list[str], language: str, allow_split: bool = True) -> list[str]:
    if len(texts) == 1:
        return [_translate_one(texts[0], language)]
    parsed = _try_batch(texts, language)
    if parsed is not None:
        return parsed
    if allow_split and len(texts) >= 4:
        # One garbled batch usually means one awkward passage, not that batching is hopeless: try each half
        # as its own batch before giving up on batching (a half that fails is then done one by one).
        mid = len(texts) // 2
        log.warning("batched translation of %d chunks was unusable; retrying as two batches", len(texts))
        return _translate_batch(texts[:mid], language, False) + _translate_batch(texts[mid:], language, False)
    log.warning("batched translation couldn't be used for %d chunks; translating one by one", len(texts))
    return [_translate_one(t, language) for t in texts]


def _try_batch(texts: list[str], language: str) -> list[str] | None:
    user = f"Source language code: {language}\n\n" + "\n".join(
        f"<<<CHUNK {i}>>>\n{text}" for i, text in enumerate(texts, 1)
    )
    parsed = _split_batch(_chat(_BATCH_SYSTEM, user, role="translate"), len(texts))
    if parsed is None:
        return None
    if any(_looks_truncated(src, out) for src, out in zip(texts, parsed)):
        log.warning("a chunk came back far shorter than its source (a truncated reply?); not trusting this batch")
        return None
    return parsed


def _looks_truncated(source: str, translated: str) -> bool:
    """A translation with under 30% of the source's words is what a cut-off reply looks like. Only judged on
    passages long enough for the ratio to mean something. (Target-English has fewer/equal words than most
    source languages, and CJK sources have far fewer 'words', so this errs toward not flagging.)"""
    source_words = len(source.split())
    return source_words >= 20 and len(translated.split()) < 0.3 * source_words


def _split_batch(reply: str, expected: int) -> list[str] | None:
    """Split a delimited reply back into `expected` texts, or None if it isn't unambiguous."""
    parts = _MARKER.split(reply)  # [preamble, "1", text1, "2", text2, ...]
    numbers, bodies = parts[1::2], parts[2::2]
    if len(numbers) != expected or [int(n) for n in numbers] != list(range(1, expected + 1)):
        return None
    texts = [_EMPHASIS.sub("", b.strip()).strip() for b in bodies]
    return texts if all(texts) else None


def _translate_one(text: str, language: str) -> str:
    out = _chat(_SYSTEM, f"Source language code: {language}\n\n{text}", role="translate")
    if not out:
        raise RuntimeError("translation came back empty")
    return out
