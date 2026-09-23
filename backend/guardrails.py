"""Rate limiters and input validation, retained for an optional future public/demo mode."""
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from backend import config
from backend.models import Video
from backend.util import format_duration


class SlidingWindowRateLimiter:
    """In-memory per-key sliding window (single process, per spec §1)."""

    def __init__(self, max_requests: int, window_seconds: int):
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record a hit and return None, or return the seconds to wait if over the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return max(1, int(self._window - (now - hits[0])) + 1)
            hits.append(now)
            return None


_ask_limiter = SlidingWindowRateLimiter(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
# One shared bucket for every visitor: the LLM provider's quota is per account, not per IP.
_global_limiter = SlidingWindowRateLimiter(config.RATE_LIMIT_GLOBAL_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
# /search now spends one small LLM call (query enhancement), so it is rationed too, but more loosely than /ask.
_search_limiter = SlidingWindowRateLimiter(config.SEARCH_RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
_search_global_limiter = SlidingWindowRateLimiter(config.SEARCH_RATE_LIMIT_GLOBAL_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
_ingest_limiter = SlidingWindowRateLimiter(config.INGEST_RATE_LIMIT_REQUESTS, config.INGEST_RATE_LIMIT_WINDOW_SECONDS)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _too_many(detail: str, wait: int) -> HTTPException:
    return HTTPException(status_code=429, detail=f"{detail} Try again in {wait} seconds.", headers={"Retry-After": str(wait)})


def rate_limit_ask(request: Request) -> None:
    """FastAPI dependency for /ask. Local mode bypasses public-demo limits explicitly."""
    if config.LOCAL_MODE:
        return
    wait = _ask_limiter.check(_client_key(request))
    scope = "You're asking too fast"
    if wait is None:
        wait = _global_limiter.check("all")
        scope = "The demo is busy right now"
    if wait is not None:
        raise _too_many(f"{scope}: answers are limited to protect the shared language-model quota.", wait)


def rate_limit_search(request: Request) -> None:
    """FastAPI dependency for /search. Local mode bypasses public-demo limits explicitly."""
    if config.LOCAL_MODE:
        return
    wait = _search_limiter.check(_client_key(request))
    scope = "You're searching too fast"
    if wait is None:
        wait = _search_global_limiter.check("all")
        scope = "The demo is busy right now"
    if wait is not None:
        raise _too_many(f"{scope}: searches are limited to protect the shared language-model quota.", wait)


def check_ingest_rate(request: Request) -> None:
    """Called by /ingest only when it is about to START a job."""
    if config.LOCAL_MODE:
        return
    wait = _ingest_limiter.check(_client_key(request))
    if wait is not None:
        raise _too_many("Too many indexing requests: indexing uses shared YouTube and language-model quota.", wait)


def check_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if len(q) > config.MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Question is too long ({len(q)} characters). The limit is {config.MAX_QUESTION_CHARS}.",
        )
    return q


def submission_length_error(videos: list[Video]) -> str | None:
    """Judge a submission against the free-demo limits: video count, any single video, and total duration.

    Returns a user-facing reason to decline it, or None if it's fine. Each limit is configurable
    (MAX_VIDEOS_PER_JOB, MAX_SINGLE_VIDEO_HOURS, MAX_TOTAL_HOURS) and 0 switches it off.
    """
    if not videos:
        return "That playlist has no videos we can access."
    limit = config.MAX_VIDEOS_PER_JOB
    if limit > 0 and len(videos) > limit:
        # The playlist listing is capped at limit + 1 entries, so this may be a lower bound, not the true size.
        return (
            f"That playlist has more than {limit} videos, and this demo indexes at most {limit} per submission. "
            f"Please submit a shorter playlist or a single video."
        )
    # LOCAL_MODE deliberately does not enforce arbitrary playlist/video-duration caps.
    if config.LOCAL_MODE:
        total = sum(v.duration for v in videos)
        if total <= 0:
            return "Couldn't determine how long that is (live streams and premieres aren't supported)."
        if total < config.MIN_TOTAL_SECONDS:
            return (
                f"That's only {format_duration(total)} long. Videos or playlists shorter than "
                f"{config.MIN_TOTAL_SECONDS} seconds (one chunk) can't be indexed."
            )
        return None

    longest = max(videos, key=lambda v: v.duration)
    if config.MAX_SINGLE_VIDEO_SECONDS > 0 and longest.duration > config.MAX_SINGLE_VIDEO_SECONDS:
        return (
            f"\"{longest.title}\" is {format_duration(longest.duration)} long, but this demo indexes videos of at most "
            f"{config.MAX_SINGLE_VIDEO_HOURS:g} hours each. Please submit something shorter."
        )
    total = sum(v.duration for v in videos)
    if total <= 0:
        return "Couldn't determine how long that is (live streams and premieres aren't supported)."
    if total < config.MIN_TOTAL_SECONDS:
        return (
            f"That's only {format_duration(total)} long. Videos or playlists shorter than "
            f"{config.MIN_TOTAL_SECONDS} seconds (one chunk) can't be indexed."
        )
    if config.MAX_TOTAL_SECONDS > 0 and total > config.MAX_TOTAL_SECONDS:
        return (
            f"That's {format_duration(total)} in total, but the limit is {config.MAX_TOTAL_HOURS:g} hours. "
            f"Please submit something shorter."
        )
    return None
