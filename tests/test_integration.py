"""Live checks against real services. Excluded from CI (`pytest -m "not integration"`); run locally with
`pytest -m integration` once .env holds your keys. Each test skips itself when its prerequisites are missing."""
import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def real_credentials(monkeypatch):
    """conftest stops backend/config.py reading .env (so unit tests are hermetic). Live tests want the real keys, so
    copy them onto the config module for the duration of one test; monkeypatch restores everything afterwards."""
    from dotenv import dotenv_values

    from backend import config

    env = {**dotenv_values(), **{k: v for k, v in os.environ.items() if v}}
    for name in ("QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        if env.get(name):
            monkeypatch.setattr(config, name, env[name])
    if config.GROQ_API_KEY:
        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        monkeypatch.setattr(config, "LLM_MODEL", config.GROQ_MODEL)
    elif config.GEMINI_API_KEY:
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "LLM_MODEL", config.GEMINI_MODEL)


def test_qdrant_is_reachable_and_the_collection_is_usable():
    from backend import config

    if not (config.QDRANT_URL and config.QDRANT_API_KEY):
        pytest.skip("needs QDRANT_URL and QDRANT_API_KEY")
    from backend.index import qdrant_store

    qdrant_store.ensure_ready()
    assert qdrant_store.count_points() >= 0


def test_youtube_transcript_fetch_for_a_captioned_video():
    from backend.ingest import transcript as tp

    try:
        result = tp.get_transcript("dQw4w9WgXcQ")
    except tp.TemporaryTranscriptFailure as exc:
        pytest.skip(f"YouTube is blocking or rate limiting this machine: {exc.reason}")
    assert result.lines and result.lines[0].start >= 0


def test_llm_translation_round_trip():
    from backend import config

    if not config.GEMINI_API_KEY:
        pytest.skip("needs GEMINI_API_KEY (the default translation provider)")
    from backend.ingest import translate
    from tests.helpers import make_chunks

    source = make_chunks(2)
    out = translate.translate_chunks(source)
    assert len(out) == 2 and out[0].text != source[0].text
