"""Environment-driven configuration for the local-first YTRAG application."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- local-first mode -------------------------------------------------------------------------
LOCAL_MODE = _bool("LOCAL_MODE", True)

# --- transcript acquisition -------------------------------------------------------------------
ALLOW_GENERATED_TRANSCRIPT = _bool("ALLOW_GENERATED_TRANSCRIPT", True)
TRANSCRIPT_PROVIDER_ORDER = ["youtube"]
TRANSCRIPT_PROVIDER_COOLDOWN_SECONDS = _int("TRANSCRIPT_PROVIDER_COOLDOWN_SECONDS", 900)
TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS = _float("TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS", 1.0)
TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS = _float("TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS", 2.5)
TRANSCRIPT_BETWEEN_VIDEOS_SECONDS = _float("TRANSCRIPT_BETWEEN_VIDEOS_SECONDS", 1.5)
TRANSCRIPT_MAX_RETRIES = _int("TRANSCRIPT_MAX_RETRIES", 2)
TRANSCRIPT_RETRY_BASE_SECONDS = _float("TRANSCRIPT_RETRY_BASE_SECONDS", 3.0)
TRANSCRIPT_RETRY_MAX_SECONDS = _float("TRANSCRIPT_RETRY_MAX_SECONDS", 30.0)
TRANSCRIPT_PAUSE_SECONDS = _int("TRANSCRIPT_PAUSE_SECONDS", 900)

# Optional direct HTTP/HTTPS proxy. Leave empty by default. A rotating residential proxy can be
# supplied here without changing the transcript provider itself.
YOUTUBE_HTTP_PROXY = _str("YOUTUBE_HTTP_PROXY")
YOUTUBE_HTTPS_PROXY = _str("YOUTUBE_HTTPS_PROXY")

# --- local caches / lightweight library metadata ----------------------------------------------
CACHE_DIR = _str("CACHE_DIR", "data/cache")
LIBRARY_METADATA_PATH = _str("LIBRARY_METADATA_PATH", "data/library.json")

# --- retrieval/chunking ------------------------------------------------------------------------
CHUNK_SECONDS = _int("CHUNK_SECONDS", 90)
CHUNK_OVERLAP_SECONDS = _int("CHUNK_OVERLAP_SECONDS", 20)
MIN_CHUNK_WORDS = _int("MIN_CHUNK_WORDS", 15)
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 500)
RETRIEVAL_THRESHOLD = _float("RETRIEVAL_THRESHOLD", 0.55)
MAX_RESULTS = _int("MAX_RESULTS", 12)
SPARSE_WEIGHT = _float("SPARSE_WEIGHT", 0.65)
DENSE_WEIGHT = _float("DENSE_WEIGHT", 0.35)
RELATIVE_SCORE_FLOOR = _float("RELATIVE_SCORE_FLOOR", 0.8)
MIN_DENSE_SIMILARITY = _float("MIN_DENSE_SIMILARITY", 0.2)
STRIP_NAVIGATION_WORDS = _bool("STRIP_NAVIGATION_WORDS", True)
SPARSE_HALF_SATURATION_TERMS = _float("SPARSE_HALF_SATURATION_TERMS", 0.6)
PERIOD_GAP_SECONDS = _int("PERIOD_GAP_SECONDS", 180)

# --- public API limiters -----------------------------------------------------------------------
# Still configurable for a future demo mode. LOCAL_MODE bypasses them explicitly in guardrails.py.
RATE_LIMIT_REQUESTS = _int("RATE_LIMIT_REQUESTS", 2)
RATE_LIMIT_GLOBAL_REQUESTS = _int("RATE_LIMIT_GLOBAL_REQUESTS", 3)
RATE_LIMIT_WINDOW_SECONDS = _int("RATE_LIMIT_WINDOW_SECONDS", 60)
SEARCH_RATE_LIMIT_REQUESTS = _int("SEARCH_RATE_LIMIT_REQUESTS", 6)
SEARCH_RATE_LIMIT_GLOBAL_REQUESTS = _int("SEARCH_RATE_LIMIT_GLOBAL_REQUESTS", 12)
INGEST_RATE_LIMIT_REQUESTS = _int("INGEST_RATE_LIMIT_REQUESTS", 2)
INGEST_RATE_LIMIT_WINDOW_SECONDS = _int("INGEST_RATE_LIMIT_WINDOW_SECONDS", 600)

# --- no artificial local playlist limits ------------------------------------------------------
MAX_VIDEOS_PER_JOB = _int("MAX_VIDEOS_PER_JOB", 0)
MAX_TOTAL_HOURS = _float("MAX_TOTAL_HOURS", 0)
MAX_TOTAL_SECONDS = int(MAX_TOTAL_HOURS * 3600) if MAX_TOTAL_HOURS > 0 else 0
MAX_SINGLE_VIDEO_HOURS = _float("MAX_SINGLE_VIDEO_HOURS", 0)
MAX_SINGLE_VIDEO_SECONDS = int(MAX_SINGLE_VIDEO_HOURS * 3600) if MAX_SINGLE_VIDEO_HOURS > 0 else 0
MIN_TOTAL_SECONDS = CHUNK_SECONDS

# --- Qdrant: deliberately kept as the persistent vector store -------------------------------
QDRANT_URL = _str("QDRANT_URL")
QDRANT_API_KEY = _str("QDRANT_API_KEY")

# --- LLMs -------------------------------------------------------------------------------------
GROQ_API_KEY = _str("GROQ_API_KEY")
GEMINI_API_KEY = _str("GEMINI_API_KEY")
GROQ_MODEL = _str("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = _str("GEMINI_MODEL", "gemini-3.5-flash")

if GROQ_API_KEY:
    LLM_PROVIDER, LLM_MODEL = "groq", GROQ_MODEL
elif GEMINI_API_KEY:
    LLM_PROVIDER, LLM_MODEL = "gemini", GEMINI_MODEL
else:
    LLM_PROVIDER, LLM_MODEL = None, "none (set GEMINI_API_KEY or GROQ_API_KEY)"

# Translation is intentionally separate from answer generation. Gemini 3.5 Flash-Lite is the
# local-first default because it is Google's high-throughput Flash-Lite model with a large context.
TRANSLATE_PROVIDER = _str("TRANSLATE_PROVIDER", "gemini").lower()
TRANSLATE_MODEL = _str("TRANSLATE_MODEL", "gemini-3.5-flash-lite")

# Query enhancement is small and latency-sensitive; keep it on the same high-throughput model by default.
ENHANCE_PROVIDER = _str("ENHANCE_PROVIDER", "gemini").lower()
ENHANCE_MODEL = _str("ENHANCE_MODEL", "gemini-3.5-flash-lite")

QUOTA_MAX_WAIT_SECONDS = _int("QUOTA_MAX_WAIT_SECONDS", 90)
TRANSLATION_DAILY_CHUNK_BUDGET = _int("TRANSLATION_DAILY_CHUNK_BUDGET", 0)

# --- embeddings ----------------------------------------------------------------------------
EMBED_MODEL = _str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL = _str("SPARSE_MODEL", "Qdrant/bm25")
FASTEMBED_CACHE_DIR = _str("FASTEMBED_CACHE_DIR") or None

# --- translation batching ---------------------------------------------------------------------
TRANSLATE_BATCH_SIZE = _int("TRANSLATE_BATCH_SIZE", 16)
TRANSLATE_BATCH_MAX_CHARS = _int("TRANSLATE_BATCH_MAX_CHARS", 24000)
TRANSLATE_MAX_CONCURRENCY = _int("TRANSLATE_MAX_CONCURRENCY", 4)

# --- other small constants ---------------------------------------------------------------------
HISTORY_WINDOW_TURNS = _int("HISTORY_WINDOW_TURNS", 3)
DOMINANT_SENTENCE_RATIO = _float("DOMINANT_SENTENCE_RATIO", 0.5)
QUERY_ENHANCE_ENABLED = _bool("QUERY_ENHANCE_ENABLED", True)
QUERY_ENHANCE_MAX_REWRITES = _int("QUERY_ENHANCE_MAX_REWRITES", 2)
QUERY_ENHANCE_MAX_KEYWORDS = _int("QUERY_ENHANCE_MAX_KEYWORDS", 6)
QUERY_ENHANCE_MAX_WAIT_SECONDS = _int("QUERY_ENHANCE_MAX_WAIT_SECONDS", 8)
EXPANDED_QUERY_DISCOUNT = _float("EXPANDED_QUERY_DISCOUNT", 0.9)

if SPARSE_WEIGHT <= DENSE_WEIGHT:
    log.warning(
        "SPARSE_WEIGHT (%s) should exceed DENSE_WEIGHT (%s): sparse-weighted retrieval is intentional.",
        SPARSE_WEIGHT,
        DENSE_WEIGHT,
    )


def effective_provider_order() -> list[str]:
    """The local build has one transcript provider by design: direct YouTube."""
    return ["youtube"]


def config_warnings() -> list[str]:
    """Human-readable startup notes. Missing keys are warnings, not boot failures."""
    notes: list[str] = []
    if not (QDRANT_URL and QDRANT_API_KEY):
        notes.append("QDRANT_URL / QDRANT_API_KEY are not set: indexing and retrieval need the persistent Qdrant store.")
    if LLM_PROVIDER is None:
        notes.append("No answer-model key is set: translation, /ask and query enhancement will be unavailable.")
    if TRANSLATE_PROVIDER == "gemini" and not GEMINI_API_KEY:
        notes.append("TRANSLATE_PROVIDER=gemini but GEMINI_API_KEY is empty: non-English translation will be unavailable.")
    if ENHANCE_PROVIDER == "gemini" and not GEMINI_API_KEY:
        notes.append("ENHANCE_PROVIDER=gemini but GEMINI_API_KEY is empty: query enhancement will fall back to the typed question.")
    return notes
