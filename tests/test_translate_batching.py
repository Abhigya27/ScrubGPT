"""Translation batching, validation, fallback and the fallback counters, with a fake LLM (no network)."""
import pytest

from backend import config
from backend.ingest import translate
from backend.util import QuotaExhausted
from tests.helpers import echo_translator, garbling_translator, make_chunks


@pytest.fixture(autouse=True)
def translate_settings(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_PROVIDER", "fake")
    monkeypatch.setattr(config, "TRANSLATE_MODEL", "fake-model")
    monkeypatch.setattr(config, "TRANSLATE_BATCH_SIZE", 4)
    monkeypatch.setattr(config, "TRANSLATE_BATCH_MAX_CHARS", 0)
    monkeypatch.setattr(config, "TRANSLATE_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(config, "TRANSLATION_DAILY_CHUNK_BUDGET", 0)


# --- batch construction ------------------------------------------------------------------------------------
def test_batches_respect_the_chunk_count_cap():
    batches = translate._make_batches(make_chunks(10))
    assert [len(b) for b in batches] == [4, 4, 2]


def test_batches_respect_the_character_cap(monkeypatch):
    chunks = make_chunks(6)
    one = len(chunks[0].text)
    monkeypatch.setattr(config, "TRANSLATE_BATCH_SIZE", 100)
    monkeypatch.setattr(config, "TRANSLATE_BATCH_MAX_CHARS", one * 2 + 5)
    assert [len(b) for b in translate._make_batches(chunks)] == [2, 2, 2]


def test_a_batch_always_holds_at_least_one_chunk(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_BATCH_MAX_CHARS", 5)  # smaller than any chunk
    assert [len(b) for b in translate._make_batches(make_chunks(3))] == [1, 1, 1]


def test_batches_preserve_order():
    chunks = make_chunks(9)
    assert [c.chunk_id for b in translate._make_batches(chunks) for c in b] == [c.chunk_id for c in chunks]


# --- reply parsing ----------------------------------------------------------------------------------------
def test_split_batch_parses_a_well_formed_reply():
    reply = "<<<CHUNK 1>>>\nfirst\n<<<CHUNK 2>>>\nsecond"
    assert translate._split_batch(reply, 2) == ["first", "second"]


def test_split_batch_tolerates_bold_markers_and_spacing():
    reply = "**<<< CHUNK 1 >>>**\nfirst\n**<<<chunk 2>>>**\nsecond"
    assert translate._split_batch(reply, 2) == ["first", "second"]


@pytest.mark.parametrize("reply", [
    "<<<CHUNK 1>>>\nonly one",                                          # a passage missing
    "<<<CHUNK 1>>>\na\n<<<CHUNK 3>>>\nc",                               # numbering skips
    "<<<CHUNK 2>>>\nb\n<<<CHUNK 1>>>\na",                               # out of order
    "<<<CHUNK 1>>>\na\n<<<CHUNK 2>>>\n",                                # empty passage
    "A free-form summary with no markers at all.",
])
def test_split_batch_rejects_ambiguous_replies(reply):
    assert translate._split_batch(reply, 2) is None


def test_truncation_heuristic():
    long_source = " ".join(["palabra"] * 40)
    assert translate._looks_truncated(long_source, "three words only")
    assert not translate._looks_truncated(long_source, " ".join(["word"] * 30))
    assert not translate._looks_truncated("short source text", "x")  # too short to judge


# --- fallback behaviour and counters ----------------------------------------------------------------------
TEXTS = [f"texto {i}" for i in range(4)]


def test_a_good_batch_needs_no_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", echo_translator(calls))
    stats = translate.TranslationStats()
    assert translate._translate_batch(TEXTS, "es", stats=stats) == [f"EN: {t}" for t in TEXTS]
    assert len(calls) == 1
    assert (stats.batches_fell_back_to_smaller, stats.batches_fell_back_to_individual) == (0, 0)


def test_unusable_batch_is_retried_as_two_halves(monkeypatch):
    calls = []
    # Garble only the full batch of four; each half of two is fine.
    monkeypatch.setattr(translate, "_chat", garbling_translator(calls, garble_when=lambda p: len(p) == 4))
    stats = translate.TranslationStats()
    assert translate._translate_batch(TEXTS, "es", stats=stats) == [f"EN: {t}" for t in TEXTS]
    assert len(calls) == 3  # full batch, then the two halves
    assert (stats.batches_fell_back_to_smaller, stats.batches_fell_back_to_individual) == (1, 0)


def test_unusable_halves_fall_back_to_one_by_one(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", garbling_translator(calls))  # every multi-passage reply is garbage
    stats = translate.TranslationStats()
    assert translate._translate_batch(TEXTS, "es", stats=stats) == [f"EN: {t}" for t in TEXTS]
    assert len(calls) == 1 + 2 + 4  # full, two halves, four singles
    assert (stats.batches_fell_back_to_smaller, stats.batches_fell_back_to_individual) == (1, 2)


def test_a_small_unusable_batch_goes_straight_to_one_by_one(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", garbling_translator(calls))
    stats = translate.TranslationStats()
    assert translate._translate_batch(TEXTS[:3], "es", stats=stats) == [f"EN: {t}" for t in TEXTS[:3]]
    assert len(calls) == 1 + 3
    assert (stats.batches_fell_back_to_smaller, stats.batches_fell_back_to_individual) == (0, 1)


def test_a_single_chunk_is_translated_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", echo_translator(calls))
    stats = translate.TranslationStats()
    assert translate._translate_batch(["solo"], "es", stats=stats) == ["EN: solo"]
    assert len(calls) == 1 and stats.batches_fell_back_to_individual == 0


def test_a_truncated_reply_is_not_trusted(monkeypatch):
    long_texts = [" ".join(["palabra"] * 40) for _ in range(2)]
    replies = iter(["<<<CHUNK 1>>>\nshort\n<<<CHUNK 2>>>\nalso short"])

    def chat(_system, user, role="translate", **_kwargs):
        if "<<<CHUNK" in user:
            return next(replies)
        return " ".join(["word"] * 40)

    monkeypatch.setattr(translate, "_chat", chat)
    stats = translate.TranslationStats()
    out = translate._translate_batch(long_texts, "es", stats=stats)
    assert all(len(o.split()) == 40 for o in out)
    assert stats.batches_fell_back_to_individual == 1


def test_counters_are_optional(monkeypatch):
    monkeypatch.setattr(translate, "_chat", garbling_translator())
    assert translate._translate_batch(TEXTS, "es") == [f"EN: {t}" for t in TEXTS]  # no stats object: no crash


# --- translate_chunks end to end ----------------------------------------------------------------------------
def test_translation_keeps_order_and_metadata(monkeypatch):
    monkeypatch.setattr(translate, "_chat", echo_translator())
    chunks = make_chunks(10)
    out = translate.translate_chunks(chunks)
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks]
    assert all(o.text == f"EN: {c.text}" and o.start_sec == c.start_sec for o, c in zip(out, chunks))


def test_concurrent_translation_keeps_order(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_MAX_CONCURRENCY", 3)
    monkeypatch.setattr(translate, "_chat", echo_translator())
    chunks = make_chunks(24)
    out = translate.translate_chunks(chunks)
    assert [o.text for o in out] == [f"EN: {c.text}" for c in chunks]


def test_english_transcripts_are_not_translated(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", echo_translator(calls))
    chunks = make_chunks(5, language="en")
    stats = translate.TranslationStats()
    assert translate.translate_chunks(chunks, stats=stats) == chunks
    assert calls == [] and stats.batches_attempted == 0


def test_translations_are_cached_on_disk(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_chat", echo_translator(calls))
    chunks = make_chunks(8)
    first = translate.translate_chunks(chunks)
    made = len(calls)
    stats = translate.TranslationStats()
    second = translate.translate_chunks(chunks, stats=stats)
    assert [c.text for c in second] == [c.text for c in first]
    assert len(calls) == made  # the second run made no LLM calls
    assert stats.cache_hit and stats.batches_attempted == 0 and stats.chunks_from_cache == 8


def test_progress_callback_reports_every_chunk(monkeypatch):
    monkeypatch.setattr(translate, "_chat", echo_translator())
    seen = []
    translate.translate_chunks(make_chunks(10), on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (10, 10) and [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_stats_across_a_whole_video(monkeypatch):
    """8 chunks = 2 batches of 4; the second batch is garbled, so exactly one of two batches needed a fallback."""
    seen_batches = []

    def garble_second_batch(passages):
        seen_batches.append(len(passages))
        return len(seen_batches) == 2

    monkeypatch.setattr(translate, "_chat", garbling_translator(garble_when=garble_second_batch))
    stats = translate.TranslationStats()
    translate.translate_chunks(make_chunks(8), stats=stats)
    assert stats.batches_attempted == 2
    assert stats.fallback_batches == 1 and stats.fallback_rate == 0.5
    assert stats.batches_fell_back_to_smaller == 1
    assert stats.chunks_translated == 8 and stats.chunks_from_cache == 0 and not stats.cache_hit
    assert translate.totals()["batches_attempted"] == 2  # the process-wide totals saw the same run


def test_a_batch_needing_several_fallbacks_counts_once(monkeypatch):
    monkeypatch.setattr(translate, "_chat", garbling_translator())  # split, then both halves one by one
    stats = translate.TranslationStats()
    translate.translate_chunks(make_chunks(4), stats=stats)
    assert (stats.batches_attempted, stats.fallback_batches) == (1, 1)
    assert stats.batches_fell_back_to_smaller == 1 and stats.batches_fell_back_to_individual == 2
    assert stats.fallback_rate == 1.0


def test_fallback_rate_is_none_without_batches():
    assert translate.TranslationStats().fallback_rate is None


def test_stats_as_dict_is_json_friendly():
    import json

    stats = translate.TranslationStats(batches_attempted=4, fallback_batches=1, chunks_translated=16)
    data = json.loads(json.dumps(stats.as_dict()))
    assert data["fallback_rate"] == 0.25 and data["batches_attempted"] == 4


# --- the daily budget and errors ---------------------------------------------------------------------------
def test_daily_budget_stops_further_translation(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATION_DAILY_CHUNK_BUDGET", 5)
    monkeypatch.setattr(translate, "_chat", echo_translator())
    translate.translate_chunks(make_chunks(4, language="es"))  # a fresh day lets the first video through
    with pytest.raises(QuotaExhausted, match="Daily translation budget"):
        translate.translate_chunks(make_chunks(4, language="hi"))


def test_quota_errors_propagate_and_nothing_is_cached(monkeypatch):
    def broke(*_args, **_kwargs):
        raise QuotaExhausted("provider quota used up", 600)

    monkeypatch.setattr(translate, "_chat", broke)
    stats = translate.TranslationStats()
    with pytest.raises(QuotaExhausted):
        translate.translate_chunks(make_chunks(4), stats=stats)
    assert stats.batches_attempted == 1 and stats.chunks_translated == 0  # the attempt is still counted
    monkeypatch.setattr(translate, "_chat", echo_translator())
    assert translate.translate_chunks(make_chunks(4))[0].text.startswith("EN: ")  # not poisoned by the failure
