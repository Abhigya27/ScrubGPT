"""Shared helpers for the evaluation scripts: eval-set loading, metrics, result files, small printing utilities.

Everything in this module is pure or lazily imports the backend, so the metric functions can be unit-tested
without Qdrant, an LLM or the embedding models (see tests/test_eval_metrics.py).

Eval sets use the wrapped `source_id` / `questions` schema from `evaluation/eval_set.example.json`:

    {
      "source_id": "PLAYLIST_OR_VIDEO_ID",
      "questions": [{
        "id": "e01",
        "q": "...",
        "expected": [{"video_id": "VIDEO_ID", "start": "5:42", "end": "6:30"}],
        "category": "direct_fact",
        "gold_answer_contains": "...",
        "gold_answer": "..."
      }]
    }

The loader also accepts the older flat entry form with `question`, `video_id`, and `source_id`.
Entries whose `q` or `question` is null/empty are unfilled template stubs and are ignored. Entries with
`"needs_review": true` (written by draft_eval_set.py) are LLM drafts nobody has checked yet and are ignored too, with
a notice, until you verify them and delete that flag.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WRITTEN: list[Path] = []  # every results file written by this process, in order (run_all.py reads it)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"

CATEGORIES = ("direct_fact", "paraphrase", "followup", "terminology_mismatch", "cross_video", "unanswerable")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class EvalSetError(ValueError):
    """The eval set is malformed. The message says which entry and what is wrong."""


# --- provenance -------------------------------------------------------------------------------------
def git_info() -> dict:
    """Commit hash and whether the working tree has uncommitted changes (results should say what they measured)."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_results(prefix: str, payload: dict, out_dir: Path | str | None = None) -> Path:
    """Write `<prefix>_<UTC timestamp>.json` with generated_at and the git commit added. Returns the path."""
    directory = Path(out_dir) if out_dir else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{prefix}_{stamp}.json"
    body = {"generated_at": now_iso(), "git": git_info(), **payload}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    WRITTEN.append(path)
    return path


# --- eval-set loading ---------------------------------------------------------------------------------
def _parse_seconds(value, where: str) -> int:
    from evaluation.run_eval import parse_ts

    try:
        return int(parse_ts(value))
    except ValueError as exc:
        raise EvalSetError(f"{where}: {exc}") from None


def _normalize_history(raw, where: str) -> list[dict]:
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise EvalSetError(f"{where}: 'history' must be a list of turns")
    turns = []
    for i, item in enumerate(raw, 1):
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict) or not (item.get("question") or "").strip():
            raise EvalSetError(f"{where}: history turn {i} needs a 'question'")
        turns.append({"question": item["question"].strip(), "answer": (item.get("answer") or None)})
    return turns


def load_eval_set(path: Path | str) -> tuple[list[dict], dict]:
    """Read and validate an eval set. Returns (entries, info).

    Unfilled stubs (question null/empty) are dropped and counted in info["skipped_stubs"]. Raises EvalSetError for
    anything malformed, naming the entry, so a typo does not silently shrink the question set.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalSetError(
            f"{path} not found. Generate a skeleton with `python -m evaluation.build_eval_template`, fill it in, "
            "or pass --eval-set."
        ) from None
    except ValueError as exc:
        raise EvalSetError(f"{path} is not valid JSON: {exc}") from None
    file_source_id = ""
    if isinstance(raw, dict):
        file_source_id = (raw.get("source_id") or "").strip()
        raw = raw.get("questions")
    if not isinstance(raw, list):
        raise EvalSetError(f"{path} must contain a JSON list of entries (or an object with a 'questions' list)")

    entries, seen, stubs, unreviewed = [], set(), 0, 0
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise EvalSetError(f"entry {index} is not an object")
        label = item.get("id") or f"entry {index}"
        question = item.get("question", item.get("q"))
        if not (question or "").strip():
            stubs += 1
            continue
        if item.get("needs_review"):
            unreviewed += 1
            continue
        item_source_id = (item.get("source_id") or file_source_id).strip()
        if not item_source_id:
            raise EvalSetError(f"{label}: 'source_id' is required (the playlist or video id you ingested)")
        if label in seen:
            raise EvalSetError(f"{label}: duplicate id")
        seen.add(label)

        category = item.get("category") or "direct_fact"
        if category not in CATEGORIES:
            raise EvalSetError(f"{label}: unknown category {category!r} (use one of {', '.join(CATEGORIES)})")
        expects_refusal = bool(item.get("expects_refusal", item.get("expected") == []))

        gold = item.get("gold_answer_contains")
        if gold is not None and not (isinstance(gold, str) or (isinstance(gold, list) and all(isinstance(g, str) for g in gold))):
            raise EvalSetError(f"{label}: 'gold_answer_contains' must be a string or a list of strings")
        if isinstance(gold, str) and not gold.strip():
            gold = None
        if isinstance(gold, list):
            gold = [g for g in gold if g.strip()] or None

        gold_answer = item.get("gold_answer")
        if gold_answer is not None and not isinstance(gold_answer, str):
            raise EvalSetError(f"{label}: 'gold_answer' must be a string")
        gold_answer = (gold_answer or "").strip() or None

        expected = item.get("expected")
        if expected is not None:
            if not isinstance(expected, list):
                raise EvalSetError(f"{label}: 'expected' must be a list of video/time ranges")
            for i, location in enumerate(expected, 1):
                if not isinstance(location, dict) or not (location.get("video_id") or "").strip():
                    raise EvalSetError(f"{label}: expected location {i} needs a 'video_id'")
                if "start" not in location or "end" not in location:
                    raise EvalSetError(f"{label}: expected location {i} needs 'start' and 'end'")
                start = _parse_seconds(location["start"], f"{label}: expected location {i} start")
                end = _parse_seconds(location["end"], f"{label}: expected location {i} end")
                if end <= start:
                    raise EvalSetError(f"{label}: expected location {i} end must follow start")

        ts = item.get("gold_timestamp_seconds")
        ts = None if ts in (None, "") else _parse_seconds(ts, f"{label}: gold_timestamp_seconds")

        source_id = item_source_id
        expected_videos = list(dict.fromkeys(
            (location.get("video_id") or "").strip() for location in (expected or [])
        ))
        video_id = (item.get("video_id") or "").strip()
        if not video_id and len(expected_videos) == 1:
            video_id = expected_videos[0]
        video_id = video_id or (source_id if _VIDEO_ID.match(source_id) else None)

        entries.append({
            "id": label,
            "source_id": source_id,
            "video_id": video_id,
            "question": question.strip(),
            "category": category,
            "gold_answer_contains": gold,
            "gold_answer": gold_answer,
            "gold_timestamp_seconds": ts,
            "expects_refusal": expects_refusal,
            "history": _normalize_history(item.get("history"), label),
            "expected": expected,
        })
    if unreviewed:
        print(f"note: ignoring {unreviewed} unreviewed draft question(s) (\"needs_review\": true). Check each one, then "
              "delete that flag to include it.", file=sys.stderr)
    if not entries:
        raise EvalSetError(
            f"{path} has no usable questions ({stubs} unfilled stub(s), {unreviewed} unreviewed draft(s)). "
            "Write or review the questions first; see evaluation/README.md."
        )
    info = {"path": str(path), "entries": len(entries), "skipped_stubs": stubs, "skipped_unreviewed": unreviewed}
    return entries, info


# --- relevance & metrics (pure) ------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Case-insensitive, punctuation-insensitive, whitespace-collapsed form used for gold-text matching.

    Transcripts are lightly punctuated while a human writing `gold_answer_contains` is not, so "O(1) lookups" and
    "o 1 lookups" must still match."""
    return " ".join(_PUNCT.sub(" ", (text or "").casefold()).split())


def is_relevant(text: str, gold) -> bool:
    """True when `text` contains the gold string (or any of a list of gold strings), case-insensitively."""
    if not gold:
        return False
    haystack = normalize_text(text)
    golds = [gold] if isinstance(gold, str) else list(gold)
    return any(normalize_text(g) and normalize_text(g) in haystack for g in golds)


def first_relevant_rank(texts: list[str], gold, depth: int | None = None) -> int | None:
    """1-based rank of the first text containing the gold string within the first `depth` texts, else None."""
    for rank, text in enumerate(texts[:depth] if depth else texts, 1):
        if is_relevant(text, gold):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float | None:
    """Fraction of questions whose first relevant result is within the top k. None for an empty list."""
    if not ranks:
        return None
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None], depth: int | None = None) -> float | None:
    """Mean of 1/rank; a question with no relevant result (or one beyond `depth`) contributes 0."""
    if not ranks:
        return None
    return sum(1.0 / r for r in ranks if r is not None and (depth is None or r <= depth)) / len(ranks)


def precision_at_k(flags: list[bool], k: int) -> float | None:
    """Relevant results among the top k, divided by k (so a short list is penalised, the usual definition)."""
    if k <= 0:
        return None
    return sum(1 for f in flags[:k] if f) / k


def chunk_recall_at_k(flags: list[bool], k: int, total_relevant: int | None) -> float | None:
    """The share of ALL relevant chunks in the source that appear in the top k. None when the total is unknown or 0.

    This is the information-retrieval meaning of recall. `recall_at_k` above (the spec's "Recall@5") is really a hit
    rate: did ANY relevant chunk make the top k."""
    if not total_relevant:
        return None
    return min(sum(1 for f in flags[:k] if f), total_relevant) / total_relevant


def ndcg_at_k(flags: list[bool], k: int, total_relevant: int | None = None) -> float | None:
    """Binary-relevance nDCG@k: rewards relevant chunks ranked early. 1.0 means every relevant chunk that could fit
    in the top k is at the very top. `total_relevant` is the number of relevant chunks that exist (from a corpus scan);
    when unknown, the ideal is taken as the relevant chunks that were retrieved (at least one), so a miss scores 0."""
    if total_relevant == 0:
        return None
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags[:k]) if f)
    ideal_count = min(k, total_relevant if total_relevant is not None else max(1, sum(1 for f in flags if f)))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / ideal if ideal else None


def mean(values: list[float | None]) -> float | None:
    """Mean of the values that exist (None entries are skipped). None when there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: list[float], p: float) -> float | None:
    """The p-th percentile (0-100) with linear interpolation between ranks. None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(position), math.ceil(position)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def bootstrap_ci(values: list[float | None], n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile-bootstrap confidence interval for the MEAN of continuous scores (judge metrics are not 0/1, so the
    Wilson interval does not apply). Seeded, so the same scores always give the same interval."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    rng = random.Random(seed)
    n = len(present)
    means = sorted(sum(rng.choices(present, k=n)) / n for _ in range(n_boot))
    low = means[int((1 - level) / 2 * n_boot)]
    high = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return low, high


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion. With ~40 questions the honest uncertainty is large."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


# --- printing ---------------------------------------------------------------------------------------------
def pct(x: float | None, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def format_table(headers: list[str], rows: list[list], align_right_from: int = 1) -> str:
    """A plain-text table with aligned columns (first column left-aligned, the rest right-aligned)."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]

    def line(row: list[str]) -> str:
        return "  ".join(
            c.rjust(w) if i >= align_right_from else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))
        ).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), rule, *(line(r) for r in cells)])


def quiet_backend_logging() -> None:
    """The per-search diagnostic line in qdrant_store would flood an evaluation report."""
    logging.getLogger("backend").setLevel(logging.WARNING)
    logging.getLogger("backend.index.qdrant_store").setLevel(logging.WARNING)


def fail(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


# --- the app's search pipeline, in-process (needs the backend + a live index) -------------------------------
def build_history(turns: list[dict]):
    """Eval-set history -> the ChatTurn list the app passes to enhancement and answering."""
    from backend.models import ChatTurn

    return [
        ChatTurn(question=t["question"], standalone_question=t["question"], answer=t.get("answer"), citations=[])
        for t in turns or []
    ]


def prepare_query(entry: dict, enhance: str):
    """What retrieval is given for this question: (standalone question, extra queries, history turns, note).

    enhance="none"     the question as typed (follow-ups are NOT resolved, so expect them to retrieve poorly)
    enhance="history"  resolve follow-ups from `history`, nothing else (isolates retrieval from query expansion)
    enhance="full"     exactly what the app does: rag/enhance.py (follow-up resolution plus rewrites/keywords)

    Costs one small LLM call per question for "full", and only for follow-ups for "history". Never raises for an
    LLM problem: the question is then searched as typed, like the app does, and the note says so.
    """
    from backend.util import UserFacingError

    question, prior = entry["question"], build_history(entry.get("history"))
    if enhance == "none" or (enhance == "history" and not prior):
        return question, [], prior, ""
    from backend.rag.enhance import enhance_query

    try:
        enhanced = enhance_query(prior, question)
    except UserFacingError as exc:
        return question, [], prior, f"enhancement unavailable ({exc}); searched as typed"
    expansions = enhanced.expansions if enhance == "full" else []
    return enhanced.standalone, expansions, prior, enhanced.note


def run_search(entry: dict, enhance: str = "full") -> dict:
    """The app's /search pipeline for one eval entry: enhance -> hybrid retrieve -> group into periods.

    Returns {"standalone", "expansions", "prior", "note", "chunks", "periods"}. Mirrors backend/main.py's
    _chat_events (kept in sync by hand; both call the same underlying functions)."""
    from backend.rag.retrieve import group_into_periods, retrieve_chunks

    standalone, expansions, prior, note = prepare_query(entry, enhance)
    chunks = retrieve_chunks(standalone, entry["source_id"], expansions)
    return {
        "standalone": standalone, "expansions": expansions, "prior": prior, "note": note,
        "chunks": chunks, "periods": group_into_periods(chunks),
    }
