"""Transcript selection, retry behaviour, error classification, router precedence and the disk cache. All fakes."""
import pytest

from backend import config
from backend.ingest import cache
from backend.ingest import transcript as tp
from tests.helpers import FakeApi, FakeTranscript, Rate429, yt_provider


def route(api):
    return tp.TranscriptRouter([yt_provider(api)])


# --- selection order -----------------------------------------------------------------------------------------
def test_manual_selection_prefers_original_language():
    api = FakeApi([FakeTranscript("es"), FakeTranscript("en"), FakeTranscript("hi")])
    result = route(api).get_transcript("vid", original_language="hi")
    assert (result.language, result.source_type, result.provider) == ("hi", "youtube_manual", "youtube")


def test_manual_selection_falls_back_to_english_then_anything():
    api = FakeApi([FakeTranscript("es"), FakeTranscript("en")])
    assert route(api).get_transcript("vid", original_language="hi").language == "en"
    api = FakeApi([FakeTranscript("es"), FakeTranscript("fr")])
    assert route(api).get_transcript("vid", original_language="hi").language == "es"


def test_language_codes_match_on_primary_subtag():
    api = FakeApi([FakeTranscript("en-US"), FakeTranscript("hi-IN")])
    assert route(api).get_transcript("vid", original_language="hi").language == "hi-IN"


def test_manual_transcript_beats_generated_in_the_original_language(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_GENERATED_TRANSCRIPT", True)
    api = FakeApi([FakeTranscript("hi", generated=True), FakeTranscript("en")])
    result = route(api).get_transcript("vid", original_language="hi")
    assert (result.language, result.source_type) == ("en", "youtube_manual")


def test_generated_transcript_is_used_and_labelled_when_allowed(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_GENERATED_TRANSCRIPT", True)
    result = route(FakeApi([FakeTranscript("en", generated=True)])).get_transcript("generated")
    assert result.source_type == "youtube_generated" and result.is_generated


def test_generated_transcript_is_rejected_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_GENERATED_TRANSCRIPT", False)
    with pytest.raises(tp.TranscriptUnavailable):
        route(FakeApi([FakeTranscript("en", generated=True)])).get_transcript("generated-off")


def test_no_transcripts_at_all_is_unavailable():
    with pytest.raises(tp.TranscriptUnavailable):
        route(FakeApi([])).get_transcript("none")


def test_empty_transcript_is_unavailable():
    with pytest.raises(tp.TranscriptUnavailable, match="empty"):
        route(FakeApi([FakeTranscript("en", count=0)])).get_transcript("empty")


def test_transcript_lines_carry_start_and_end_times():
    result = route(FakeApi([FakeTranscript("en", count=3)])).get_transcript("vid")
    assert [(ln.start, ln.end) for ln in result.lines] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def test_router_has_only_youtube():
    assert [p.name for p in tp.build_providers()] == ["youtube"]
    assert config.effective_provider_order() == ["youtube"]


# --- retries ------------------------------------------------------------------------------------------------
def test_rate_limit_retries_then_succeeds(no_sleep):
    api = FakeApi([FakeTranscript("en")], error=Rate429(1))
    result = yt_provider(api).get_transcript("retry")
    assert result.provider == "youtube" and api.list_calls == 2
    assert len(no_sleep) == 1 and no_sleep[0] >= 1  # honoured Retry-After


def test_rate_limit_gives_up_after_the_retry_budget(no_sleep, monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_MAX_RETRIES", 1)

    class AlwaysLimited(FakeApi):
        def list(self, _video_id):
            self.list_calls += 1
            raise Rate429(1)

    api = AlwaysLimited([])
    with pytest.raises(tp.YouTubeTemporaryRateLimit):
        yt_provider(api).get_transcript("limited")
    assert api.list_calls == 2  # the first try plus one retry


def test_retry_after_longer_than_the_cap_is_not_retried(no_sleep, monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_RETRY_MAX_SECONDS", 30)
    api = FakeApi([FakeTranscript("en")], error=Rate429(600))
    with pytest.raises(tp.YouTubeTemporaryRateLimit):
        yt_provider(api).get_transcript("long-wait")
    assert api.list_calls == 1 and not no_sleep  # retrying early would be retrying too soon


# --- error classification ----------------------------------------------------------------------------------
def test_classify_rate_limit_keeps_retry_after():
    err = tp._classify(Rate429(7), "v")
    assert isinstance(err, tp.YouTubeTemporaryRateLimit) and err.retry_after == 7


def test_classify_unknown_error_is_temporary_not_a_skip():
    err = tp._classify(RuntimeError("socket exploded"), "v")
    assert isinstance(err, tp.TemporaryTranscriptFailure)
    assert not isinstance(err, tp.TranscriptUnavailable)


# --- router precedence --------------------------------------------------------------------------------------
class _Provider(tp.TranscriptProvider):
    def __init__(self, name, exc):
        self.name, self._exc = name, exc

    def get_transcript(self, *args, **kwargs):
        raise self._exc


def test_temporary_failure_outranks_unavailable():
    """If any provider failed temporarily we cannot know the video has no transcript, so it must not be a skip."""
    router = tp.TranscriptRouter([
        _Provider("a", tp.TranscriptUnavailable()),
        _Provider("b", tp.TemporaryTranscriptFailure("try later", 60, "b")),
    ])
    with pytest.raises(tp.TemporaryTranscriptFailure):
        router.get_transcript("v")


def test_parse_errors_outrank_temporary_failures():
    router = tp.TranscriptRouter([
        _Provider("a", tp.TemporaryTranscriptFailure("try later", 60, "a")),
        _Provider("b", tp.TranscriptParseError("unreadable")),
    ])
    with pytest.raises(tp.TranscriptParseError):
        router.get_transcript("v")


def test_access_denied_stops_immediately():
    router = tp.TranscriptRouter([_Provider("a", tp.YouTubeAccessDenied("private")), _Provider("b", tp.TranscriptUnavailable())])
    with pytest.raises(tp.YouTubeAccessDenied):
        router.get_transcript("v")


# --- caching ------------------------------------------------------------------------------------------------
def test_transcript_disk_cache_avoids_a_second_youtube_call():
    api = FakeApi([FakeTranscript("en")])
    tp.set_router(route(api))
    tp.clear_cache()
    first = tp.get_transcript("cached")
    tp.clear_cache()  # drop the in-process copy so the second call must come from disk
    second = tp.get_transcript("cached")
    assert first.language == second.language == "en"
    assert len(second.lines) == len(first.lines)
    assert api.list_calls == 1


def test_temporary_failures_are_never_cached():
    tp.set_router(tp.TranscriptRouter([_Provider("fake", tp.YouTubeTemporaryRateLimit("slow down", 5))]))
    with pytest.raises(tp.TemporaryTranscriptFailure):
        tp.get_transcript("not-cached")
    assert not cache.transcript_path("not-cached").exists()


def test_unavailable_transcripts_are_never_cached():
    tp.set_router(tp.TranscriptRouter([_Provider("fake", tp.TranscriptUnavailable())]))
    with pytest.raises(tp.TranscriptUnavailable):
        tp.get_transcript("none-here")
    assert not cache.transcript_path("none-here").exists()


def test_corrupt_cache_file_is_a_miss_not_a_crash():
    path = cache.transcript_path("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")
    api = FakeApi([FakeTranscript("en")])
    tp.set_router(route(api))
    assert tp.get_transcript("broken").language == "en"
    assert api.list_calls == 1
