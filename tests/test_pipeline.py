"""One video through the ingest pipeline with Qdrant, YouTube and the LLM faked: chunking happens before translation,
and the translation fallback counters reach the result that jobs.py reports."""
import pytest

from backend import config
from backend.index import qdrant_store
from backend.ingest import pipeline, translate
from backend.ingest import transcript as tp
from backend.models import TranscriptResult
from tests.helpers import demo_video, garbling_translator, echo_translator, make_lines


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for name, value in dict(CHUNK_SECONDS=90, CHUNK_OVERLAP_SECONDS=20, MIN_CHUNK_WORDS=15, TRANSLATE_PROVIDER="fake",
                            TRANSLATE_MODEL="m", TRANSLATE_BATCH_SIZE=2, TRANSLATE_BATCH_MAX_CHARS=0,
                            TRANSLATE_MAX_CONCURRENCY=1, TRANSLATION_DAILY_CHUNK_BUDGET=0).items():
        monkeypatch.setattr(config, name, value)


@pytest.fixture
def fakes(monkeypatch):
    """Qdrant: nothing indexed yet, writes are recorded. YouTube: serves a 300 s transcript in the chosen language."""
    state = {"language": "hi", "upserted": []}
    monkeypatch.setattr(qdrant_store, "video_exists", lambda video_id: False)
    monkeypatch.setattr(qdrant_store, "upsert_chunks",
                        lambda chunks: state["upserted"].extend(chunks) or {"embedding_seconds": 0.5, "indexing_seconds": 0.25})

    def get_transcript(video_id, **kwargs):
        return TranscriptResult(lines=make_lines(30), language=state["language"], language_name="Test", is_generated=False,
                                provider="youtube", source_type="youtube_manual")

    monkeypatch.setattr(tp, "get_transcript", get_transcript)
    return state


def test_english_video_needs_no_translation_and_has_no_counters(fakes, monkeypatch):
    fakes["language"] = "en"
    monkeypatch.setattr(translate, "_chat", lambda *a, **k: pytest.fail("an English video must not call the LLM"))
    out = pipeline.process_video(demo_video(), "PL", 0)
    assert out.translated is False and out.translation is None and out.chunks == len(fakes["upserted"]) == 4
    assert {c.transcript_source for c in fakes["upserted"]} == {"youtube_manual"}


def test_translated_video_reports_clean_counters(fakes, monkeypatch):
    monkeypatch.setattr(translate, "_chat", echo_translator())
    out = pipeline.process_video(demo_video(), "PL", 0)
    assert out.translated and out.language == "hi"
    assert out.translation["batches_attempted"] == 2  # 4 chunks in batches of 2
    assert out.translation["fallback_batches"] == 0 and out.translation["fallback_rate"] == 0.0
    assert out.translation["chunks_translated"] == 4 and out.translation["cache_hit"] is False
    assert {c.transcript_source for c in fakes["upserted"]} == {"youtube_manual_translated"}
    assert all(c.text.startswith("EN: ") for c in fakes["upserted"])


def test_timestamps_come_from_the_original_transcript_not_the_translation(fakes, monkeypatch):
    monkeypatch.setattr(translate, "_chat", echo_translator())
    pipeline.process_video(demo_video(), "PL", 0)
    assert [(c.start_sec, c.end_sec) for c in fakes["upserted"]] == [(0, 90), (70, 160), (140, 230), (210, 300)]


def test_fallbacks_are_counted_in_the_result(fakes, monkeypatch):
    monkeypatch.setattr(translate, "_chat", garbling_translator())  # every 2-chunk batch is unusable
    out = pipeline.process_video(demo_video(), "PL", 0)
    assert out.translation["batches_attempted"] == 2 and out.translation["fallback_batches"] == 2
    assert out.translation["batches_fell_back_to_individual"] == 2 and out.translation["fallback_rate"] == 1.0
    assert all(c.text.startswith("EN: ") for c in fakes["upserted"])  # still translated, one chunk at a time


def test_a_second_run_reports_a_cache_hit_not_new_batches(fakes, monkeypatch):
    monkeypatch.setattr(translate, "_chat", echo_translator())
    pipeline.process_video(demo_video(), "PL", 0)
    again = pipeline.process_video(demo_video(), "PL", 0)
    assert again.translation["cache_hit"] is True and again.translation["batches_attempted"] == 0
    assert again.translation["chunks_from_cache"] == 4


def test_a_video_that_dies_mid_translation_is_not_written_to_the_index(fakes, monkeypatch):
    from backend.util import QuotaExhausted

    def broke(*args, **kwargs):
        raise QuotaExhausted("provider quota used up", 600)

    monkeypatch.setattr(translate, "_chat", broke)
    with pytest.raises(QuotaExhausted):
        pipeline.process_video(demo_video(), "PL", 0)
    assert fakes["upserted"] == []


def test_an_already_indexed_video_is_reused_without_a_transcript_request(fakes, monkeypatch):
    tagged = []
    monkeypatch.setattr(qdrant_store, "video_exists", lambda video_id: True)
    monkeypatch.setattr(qdrant_store, "add_playlist_tag", lambda *args: tagged.append(args))
    monkeypatch.setattr(tp, "get_transcript", lambda *a, **k: pytest.fail("a reused video must not fetch a transcript"))
    out = pipeline.process_video(demo_video(), "PL2", 3)
    assert out.reused and out.translation is None and tagged == [("vid", "PL2", 3)]


def test_a_transcript_that_is_too_short_is_skipped(fakes, monkeypatch):
    monkeypatch.setattr(tp, "get_transcript", lambda video_id, **kw: TranscriptResult(
        lines=make_lines(1, words=3), language="en", language_name="English", is_generated=False,
        provider="youtube", source_type="youtube_manual"))
    with pytest.raises(pipeline.SkipVideo):
        pipeline.process_video(demo_video(), "PL", 0)
