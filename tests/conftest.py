"""Shared pytest setup. Keeps the offline suite hermetic: no .env, no network, no real sleeps, no shared state.

Anything that needs live YouTube, Qdrant or an LLM is marked @pytest.mark.integration and left out of CI
(`pytest -m "not integration"`).
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest

# 1) Never read a developer's .env. It holds real API keys and overrides the very defaults some tests assert on.
#    backend/config.py does `from dotenv import load_dotenv; load_dotenv()` at import, so neutralise it first.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

# 2) Keep the suite runnable in a clean checkout even when the YouTube transcript package is not installed. The real
#    app requires it; the tests only need a constructible placeholder, because all network behaviour is faked.
if importlib.util.find_spec("youtube_transcript_api") is None:
    _yt = types.ModuleType("youtube_transcript_api")

    class _DummyYouTubeTranscriptApi:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _errors = types.ModuleType("youtube_transcript_api._errors")
    for _name in (
        "CouldNotRetrieveTranscript", "IpBlocked", "RequestBlocked", "TranscriptsDisabled", "VideoUnavailable",
        "NoTranscriptFound", "InvalidVideoId", "AgeRestricted", "PoTokenRequired", "YouTubeTranscriptApiError",
    ):
        setattr(_errors, _name, type(_name, (Exception,), {}))
    _yt.YouTubeTranscriptApi = _DummyYouTubeTranscriptApi
    _yt._errors = _errors
    sys.modules["youtube_transcript_api"] = _yt
    sys.modules["youtube_transcript_api._errors"] = _errors


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs live YouTube, Qdrant or an LLM provider")


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Every test gets its own cache directory and library file, and leaves no module-level state behind."""
    from backend import config

    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(config, "LIBRARY_METADATA_PATH", str(tmp_path / "library.json"))
    yield
    # Reset process-wide state, but only in modules this test actually imported (keeps light tests light).
    transcript = sys.modules.get("backend.ingest.transcript")
    if transcript is not None:
        transcript.set_router(None)
        transcript.clear_cache()
        transcript.reset_provider_health()
    translate = sys.modules.get("backend.ingest.translate")
    if translate is not None:
        translate.reset_totals()
        translate._budget.update(day=None, used=0)
    jobs = sys.modules.get("backend.jobs")
    if jobs is not None:
        jobs._JOBS.clear()
    enhance = sys.modules.get("backend.rag.enhance")
    if enhance is not None:
        enhance.clear_cache()


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace the transcript layer's sleep with a recorder, and switch request pacing off. Returns the recorded sleeps."""
    from backend import config
    from backend.ingest import transcript as tp

    sleeps: list[float] = []
    monkeypatch.setattr(tp, "_sleep", sleeps.append)
    monkeypatch.setattr(config, "TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS", 0)
    return sleeps
