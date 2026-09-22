"""Dependency-light acceptance suite for the local-first YTRAG build.

Run:
    python -m tests.acceptance

The suite does not contact YouTube, Qdrant or a real LLM. It focuses on the decisions
introduced by the local-first change specification: YouTube-only transcript routing,
caching, unlimited local ingestion, rate-limit bypasses, translation batching/caching,
and basic library metadata persistence.
"""
from __future__ import annotations

import contextlib
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import types

# Keep the acceptance suite runnable in a clean checkout even when the optional
# YouTube transcript package is not installed. The real application still requires
# it; the tests only need a constructible placeholder because all network behavior
# is replaced by fakes below.
if importlib.util.find_spec("youtube_transcript_api") is None:
    yt_mod = types.ModuleType("youtube_transcript_api")

    class _DummyYouTubeTranscriptApi:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    err_mod = types.ModuleType("youtube_transcript_api._errors")
    for _name in (
        "CouldNotRetrieveTranscript",
        "IpBlocked",
        "RequestBlocked",
        "TranscriptsDisabled",
        "VideoUnavailable",
        "NoTranscriptFound",
        "InvalidVideoId",
        "AgeRestricted",
        "PoTokenRequired",
        "YouTubeTranscriptApiError",
    ):
        setattr(err_mod, _name, type(_name, (Exception,), {}))
    yt_mod.YouTubeTranscriptApi = _DummyYouTubeTranscriptApi
    yt_mod._errors = err_mod
    sys.modules["youtube_transcript_api"] = yt_mod
    sys.modules["youtube_transcript_api._errors"] = err_mod

from backend import config, guardrails, library
from backend.ingest import cache, chunker, transcript as tp, translate
from backend.models import Chunk, TranscriptLine, TranscriptResult, Video

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILED.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    else:
        PASSED.append(name)
        print(f"ok    {name}")


@contextlib.contextmanager
def patched(obj, **attrs):
    saved = {key: getattr(obj, key) for key in attrs}
    for key, value in attrs.items():
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(obj, key, value)


class FakeSnippet:
    def __init__(self, text, start, duration):
        self.text, self.start, self.duration = text, start, duration


class FakeTranscript:
    def __init__(self, lang, generated=False, count=20):
        self.language_code = lang
        self.language = {"en": "English", "hi": "Hindi"}.get(lang, lang)
        self.is_generated = generated
        self.fetch_calls = 0
        self._lines = [FakeSnippet(f"line {i} about machine learning", i * 5.0, 5.0) for i in range(count)]

    def fetch(self):
        self.fetch_calls += 1
        return self._lines


class FakeApi:
    def __init__(self, transcripts, error=None):
        self.transcripts = transcripts
        self.error = error
        self.list_calls = 0

    def list(self, _video_id):
        self.list_calls += 1
        if self.error:
            err, self.error = self.error, None
            raise err
        return list(self.transcripts)


def yt_provider(api):
    p = tp.YouTubeTranscriptProvider.__new__(tp.YouTubeTranscriptProvider)
    p._api = api
    return p


class Rate429(Exception):
    def __init__(self, retry_after=1):
        super().__init__("HTTP 429 Too Many Requests")
        self.response = SimpleNamespace(status_code=429, headers={"Retry-After": str(retry_after)})


# ----- transcript ----------------------------------------------------------------
def test_manual_selection_order():
    api = FakeApi([FakeTranscript("es"), FakeTranscript("en"), FakeTranscript("hi")])
    result = tp.TranscriptRouter([yt_provider(api)]).get_transcript("vid", original_language="hi")
    assert (result.language, result.source_type, result.provider) == ("hi", "youtube_manual", "youtube")


def test_generated_transcript_is_opt_in_by_setting_and_labeled():
    api = FakeApi([FakeTranscript("en", generated=True)])
    with patched(config, ALLOW_GENERATED_TRANSCRIPT=True):
        result = tp.TranscriptRouter([yt_provider(api)]).get_transcript("generated")
    assert result.source_type == "youtube_generated"


def test_generated_transcript_is_rejected_when_disabled():
    api = FakeApi([FakeTranscript("en", generated=True)])
    with patched(config, ALLOW_GENERATED_TRANSCRIPT=False):
        try:
            tp.TranscriptRouter([yt_provider(api)]).get_transcript("generated-off")
        except tp.TranscriptUnavailable:
            return
    raise AssertionError("generated transcript should have been unavailable")


def test_rate_limit_retries_then_succeeds():
    class RetryApi(FakeApi):
        def list(self, video_id):
            self.list_calls += 1
            if self.list_calls == 1:
                raise Rate429(1)
            return list(self.transcripts)

    sleeps = []
    api = RetryApi([FakeTranscript("en")])
    provider = yt_provider(api)
    with patched(tp, _sleep=sleeps.append), patched(config, TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS=0, TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS=0):
        result = provider.get_transcript("retry")
    assert result.provider == "youtube" and api.list_calls == 2 and sleeps


def test_router_has_only_youtube():
    assert [p.name for p in tp.build_providers()] == ["youtube"]
    assert config.effective_provider_order() == ["youtube"]


def test_transcript_disk_cache_avoids_second_youtube_call():
    with tempfile.TemporaryDirectory() as tmp:
        api = FakeApi([FakeTranscript("en")])
        provider = yt_provider(api)
        tp.set_router(tp.TranscriptRouter([provider]))
        tp.clear_cache()
        with patched(config, CACHE_DIR=tmp):
            first = tp.get_transcript("cached")
            tp.clear_cache()
            second = tp.get_transcript("cached")
        assert first.language == second.language == "en"
        assert api.list_calls == 1


# ----- local mode ----------------------------------------------------------------
def test_local_mode_disables_submission_caps():
    videos = [Video(str(i), f"Video {i}", 3600) for i in range(101)]
    with patched(config, LOCAL_MODE=True, MAX_VIDEOS_PER_JOB=0, MAX_TOTAL_HOURS=0, MAX_SINGLE_VIDEO_HOURS=0):
        assert guardrails.submission_length_error(videos) is None


def test_local_mode_bypasses_api_limiters():
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    with patched(config, LOCAL_MODE=True):
        guardrails.rate_limit_ask(request)
        guardrails.rate_limit_search(request)
        guardrails.check_ingest_rate(request)


# ----- translation --------------------------------------------------------------
def make_chunks(n=20):
    return [
        Chunk(
            chunk_id=f"v:{i}", video_id="v", video_title="Demo", playlist_ids=["p"], start_sec=i * 90,
            end_sec=(i + 1) * 90, text=f"texto del fragmento {i} sobre aprendizaje automatico", source_language="es",
        )
        for i in range(n)
    ]


def test_batches_respect_count_and_char_caps():
    chunks = make_chunks(20)
    with patched(config, TRANSLATE_BATCH_SIZE=4, TRANSLATE_BATCH_MAX_CHARS=10_000):
        batches = translate._make_batches(chunks)
    assert [len(b) for b in batches] == [4, 4, 4, 4, 4]


def test_translation_uses_disk_cache_and_keeps_order():
    calls = []

    def fake_chat(_system, user, role="translate", **_kwargs):
        calls.append((role, user))
        out = []
        for i, line in enumerate(user.split("<<<CHUNK ")[1:], 1):
            marker, text = line.split(">>>\n", 1)
            out.append(f"<<<CHUNK {marker}>>>\nTranslated {text.strip()}")
        return "\n".join(out)

    chunks = make_chunks(6)
    with tempfile.TemporaryDirectory() as tmp, patched(
        config,
        CACHE_DIR=tmp,
        TRANSLATE_PROVIDER="gemini",
        TRANSLATE_MODEL="gemini-3.5-flash-lite",
        TRANSLATE_BATCH_SIZE=3,
        TRANSLATE_BATCH_MAX_CHARS=20_000,
        TRANSLATE_MAX_CONCURRENCY=1,
        TRANSLATION_DAILY_CHUNK_BUDGET=0,
    ), patched(translate, _chat=fake_chat):
        first = translate.translate_chunks(chunks)
        second = translate.translate_chunks(chunks)
    assert len(calls) == 2  # two 3-chunk batches on the first run; second run is cached
    assert [c.text for c in first] == [c.text for c in second]
    assert first[0].text.startswith("Translated")


def test_english_translation_is_bypassed():
    chunks = [Chunk("v:0", "v", "Demo", ["p"], 0, 90, "hello", "en")]
    with patched(translate, _chat=lambda *_args, **_kw: (_ for _ in ()).throw(AssertionError("LLM called"))):
        assert translate.translate_chunks(chunks) == chunks


# ----- lightweight library ------------------------------------------------------
def test_library_metadata_persists():
    with tempfile.TemporaryDirectory() as tmp, patched(config, LIBRARY_METADATA_PATH=str(Path(tmp) / "library.json")):
        library.upsert_source("playlist-1", "playlist", [{"video_id": "v1", "title": "One", "duration": 60}])
        library.record_video("playlist-1", "v1", "One", 60, indexed=True)
        rows = library.list_sources()
    assert rows[0]["source_id"] == "playlist-1" and rows[0]["indexed_count"] == 1


# ----- chunking/provenance ------------------------------------------------------
def test_chunking_preserves_stable_timestamp_identity():
    lines = [TranscriptLine(i * 10, i * 10 + 10, f"sentence {i} with enough words to make a useful chunk") for i in range(12)]
    chunks = chunker.chunk_transcript(lines, Video("vid", "Demo", 180), "playlist", "hi", "youtube_manual", 0)
    assert chunks and chunks[0].chunk_id.startswith("vid:")
    assert chunks[0].transcript_source == "youtube_manual"


# ----- project configuration ----------------------------------------------------
def test_no_external_transcript_symbols_remain_in_config():
    assert not hasattr(config, "TRANSCRIPT_EXTERNAL_ENABLED")
    assert not hasattr(config, "TRANSCRIPT_EXTERNAL_API_KEY")


def test_local_defaults_match_project_target():
    assert config.LOCAL_MODE is True
    assert config.TRANSLATE_MODEL == "gemini-3.5-flash-lite"
    assert config.TRANSLATE_BATCH_SIZE == 16
    assert config.TRANSLATE_MAX_CONCURRENCY == 4
    assert config.TRANSLATION_DAILY_CHUNK_BUDGET == 0
    assert config.MAX_VIDEOS_PER_JOB == 0
    assert config.MAX_TOTAL_HOURS == 0
    assert config.MAX_SINGLE_VIDEO_HOURS == 0


# ----- run ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_manual_selection_order,
        test_generated_transcript_is_opt_in_by_setting_and_labeled,
        test_generated_transcript_is_rejected_when_disabled,
        test_rate_limit_retries_then_succeeds,
        test_router_has_only_youtube,
        test_transcript_disk_cache_avoids_second_youtube_call,
        test_local_mode_disables_submission_caps,
        test_local_mode_bypasses_api_limiters,
        test_batches_respect_count_and_char_caps,
        test_translation_uses_disk_cache_and_keeps_order,
        test_english_translation_is_bypassed,
        test_library_metadata_persists,
        test_chunking_preserves_stable_timestamp_identity,
        test_no_external_transcript_symbols_remain_in_config,
        test_local_defaults_match_project_target,
    ]
    for fn in tests:
        check(fn.__name__, fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    raise SystemExit(1 if FAILED else 0)
