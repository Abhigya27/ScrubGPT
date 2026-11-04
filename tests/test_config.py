"""Configuration defaults. conftest.py stops backend/config.py reading a local .env, so these are the code's own defaults."""
from backend import config


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
    assert config.effective_provider_order() == ["youtube"]


def test_retrieval_defaults_weight_sparse_above_dense():
    """The documented design (and the baseline the ablation evaluates): keyword matching leads, embeddings support."""
    assert (config.SPARSE_WEIGHT, config.DENSE_WEIGHT) == (0.65, 0.35)
    assert config.SPARSE_WEIGHT > config.DENSE_WEIGHT
    assert 0 < config.RETRIEVAL_THRESHOLD < 1
