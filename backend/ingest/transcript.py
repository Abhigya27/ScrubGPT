"""YouTube transcript acquisition for local-first YTRAG.

YouTube metadata is discovered by yt-dlp elsewhere. This module only talks to
YouTube's transcript endpoint through youtube-transcript-api, with optional
HTTP/HTTPS proxy configuration, pacing, retry/backoff, circuit-breaker behavior,
and a persistent local transcript cache.

Selection order:
    1. manual transcript in the video's original language
    2. manual English transcript
    3. another manual transcript
    4. YouTube auto-generated transcript when ALLOW_GENERATED_TRANSCRIPT=true
    5. no usable transcript -> skip the video

Temporary rate limits / IP blocks are never reinterpreted as "no transcript".
They pause the ingestion job and preserve all completed work.
"""
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import requests

from backend import config
from backend.models import TranscriptLine, TranscriptResult

log = logging.getLogger(__name__)

# Imported at module level so a missing dependency fails loudly at startup rather than in the middle of
# somebody's first indexing job.
from youtube_transcript_api import YouTubeTranscriptApi  # noqa: E402
from youtube_transcript_api import _errors as yt_errors  # noqa: E402

# Injectable so tests neither sleep nor wait for real cooldowns.
_sleep = time.sleep
_now = time.monotonic


# --- exception hierarchy -----------------------------------------------------------------------------
class TranscriptError(Exception):
    """Base class. `reason` is safe to show a user as-is; the traceback stays in the server log."""

    outcome = "error"  # short label used in the structured log line

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _no_transcript_reason() -> str:
    if config.ALLOW_GENERATED_TRANSCRIPT:
        return "YouTube has no usable transcript (manual or auto-generated) for this video."
    return "No YouTube-provided manual transcript was available for this video."


class TranscriptUnavailable(TranscriptError):
    """No transcript we are willing to use. An EXPECTED outcome: the video is skipped."""

    outcome = "unavailable"

    def __init__(self, reason: str | None = None):
        super().__init__(reason or _no_transcript_reason())


class YouTubeAccessDenied(TranscriptError):
    """Private, age-restricted, region-blocked or otherwise unplayable. Expected: the video is skipped."""

    outcome = "access_denied"


class TemporaryTranscriptFailure(TranscriptError):
    """Something outside this video is wrong right now. NEVER a skip: the job pauses and can be resumed."""

    outcome = "temporary_failure"
    trips_breaker = False  # True: the provider's route is unhealthy, so cool it down

    def __init__(self, reason: str, retry_after: int | None = None, provider: str = "youtube"):
        super().__init__(reason)
        self.retry_after = retry_after  # seconds until trying again is sensible, if known
        self.provider = provider

    @property
    def pause_reason(self) -> str:
        return "youtube" if self.provider == "youtube" else "external"


class YouTubeTemporaryRateLimit(TemporaryTranscriptFailure):
    """HTTP 429 from YouTube. The route may work again after waiting."""

    outcome = "rate_limited"


class YouTubeRouteBlocked(TemporaryTranscriptFailure):
    """IpBlocked / RequestBlocked: YouTube is rejecting this server's outbound route. Do not retry it."""

    outcome = "ip_blocked"
    trips_breaker = True


class TranscriptProviderUnavailable(TemporaryTranscriptFailure):
    """A provider is down, or unusable from here (5xx, a required PO token, an unfinished async job)."""

    outcome = "provider_unavailable"


class TranscriptFetchError(TemporaryTranscriptFailure):
    """Transport-level trouble. Recoverable in principle: bounded retry, then the job pauses."""

    outcome = "fetch_error"


class TranscriptParseError(TranscriptError):
    """A provider returned something we could not read. Treated as a bug, not a skip: the job errors."""

    outcome = "parse_error"


class TranscriptConfigError(TranscriptError):
    """A provider rejected our credentials or plan. Waiting cannot fix it: the job errors."""

    outcome = "config_error"


# --- structured logging ------------------------------------------------------------------------------
def _log_attempt(provider: str, video_id: str, attempt: int, outcome: str, retry_after=None, source_type=None) -> None:
    """One line per attempt. Never includes keys, proxy credentials, headers or cookies."""
    level = logging.INFO if outcome in {"success", "unavailable", "access_denied", "cooldown_skip"} else logging.WARNING
    log.log(
        level,
        "transcript provider=%s video=%s attempt=%d outcome=%s retry_after=%s source_type=%s",
        provider, video_id, attempt, outcome, retry_after if retry_after is not None else "-", source_type or "-",
    )


# --- provider health: a process-local circuit breaker --------------------------------------------------
# Not persisted: on a free host a restart simply probes the route again.
_provider_blocked_until: dict[str, float] = {}
_health_lock = threading.Lock()


def _block_provider(name: str, seconds: float) -> None:
    with _health_lock:
        _provider_blocked_until[name] = _now() + seconds
    log.warning("transcript provider=%s marked unhealthy for %.0f seconds", name, seconds)


def provider_cooldown_remaining(name: str) -> int:
    """Whole seconds left in the provider's cooldown, 0 when it is healthy (a probe is then allowed)."""
    with _health_lock:
        remaining = _provider_blocked_until.get(name, 0.0) - _now()
    return int(remaining) + 1 if remaining > 0 else 0


def reset_provider_health() -> None:
    with _health_lock:
        _provider_blocked_until.clear()


# --- classifying whatever the YouTube library (or requests) throws ------------------------------------
def _yt(*names: str) -> tuple[type, ...]:
    """Exception classes that exist in the installed library version (an empty tuple matches nothing)."""
    return tuple(c for c in (getattr(yt_errors, n, None) for n in names) if isinstance(c, type))


_IP_BLOCKED = _yt("IpBlocked")
_REQUEST_BLOCKED = _yt("RequestBlocked")
_PO_TOKEN = _yt("PoTokenRequired")
_DISABLED = _yt("TranscriptsDisabled")
_NOT_FOUND = _yt("NoTranscriptFound")
_DENIED = _yt("AgeRestricted", "VideoUnplayable")
_GONE = _yt("VideoUnavailable", "InvalidVideoId")
_UNPARSABLE = _yt("YouTubeDataUnparsable")
_REQUEST_FAILED = _yt("YouTubeRequestFailed")

_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "rate-limit")


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None), getattr(exc, "__cause__", None)):
        code = getattr(obj, "status_code", None)
        if code:
            return int(code)
        response = getattr(obj, "response", None)
        if response is not None and getattr(response, "status_code", None):
            return int(response.status_code)
    return None


def _retry_after_header(source) -> int | None:
    """Retry-After (seconds) from an exception, its cause, or a response object."""
    for obj in (source, getattr(source, "__cause__", None)):
        response = getattr(obj, "response", obj)
        headers = getattr(response, "headers", None)
        if not headers:
            continue
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
            if value is not None:
                return max(1, int(float(value)))
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def _looks_rate_limited(exc: BaseException) -> bool:
    if _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _classify(exc: BaseException, video_id: str) -> TranscriptError:
    """Turn any exception from the YouTube library into one of ours."""
    if isinstance(exc, TranscriptError):
        return exc
    # The route itself is rejected (very common on cloud hosts): never retried against the same IP.
    if isinstance(exc, _IP_BLOCKED) or isinstance(exc, _REQUEST_BLOCKED):
        err = YouTubeRouteBlocked("YouTube is blocking transcript requests from this server.")
        err.outcome = "ip_blocked" if isinstance(exc, _IP_BLOCKED) else "request_blocked"
        return err
    if isinstance(exc, _PO_TOKEN):
        err = TranscriptProviderUnavailable("YouTube now requires a token that this server cannot provide for transcript requests.")
        err.outcome, err.trips_breaker = "po_token_required", True
        return err
    if isinstance(exc, _DISABLED):
        return TranscriptUnavailable("The uploader turned transcripts off for this video.")
    if isinstance(exc, _NOT_FOUND):
        return TranscriptUnavailable()
    if isinstance(exc, _DENIED):
        return YouTubeAccessDenied("YouTube would not play this video for an automated request (age-restricted, private or region-locked).")
    if isinstance(exc, _GONE):
        return YouTubeAccessDenied("This video is unavailable.")
    if isinstance(exc, _UNPARSABLE):
        return TranscriptParseError("YouTube returned transcript data in a format this app could not read.")
    if _looks_rate_limited(exc):
        return YouTubeTemporaryRateLimit("YouTube rate-limited the transcript request (HTTP 429).", _retry_after_header(exc))
    if isinstance(exc, _REQUEST_FAILED) or isinstance(exc, requests.RequestException):
        return TranscriptFetchError("The request to YouTube failed. This is usually temporary.")
    log.warning("unclassified transcript error for %s: %s: %s", video_id, type(exc).__name__, exc)
    return TranscriptFetchError("Could not reach YouTube to read the transcript. This is usually temporary.")


# --- pacing and retry timing ----------------------------------------------------------------------------
_pace_lock = threading.Lock()
_last_request_at = {"t": 0.0}


def _pace() -> None:
    """Randomised gap between YouTube requests so a playlist never bursts at YouTube."""
    low = max(0.0, config.TRANSCRIPT_REQUEST_MIN_DELAY_SECONDS)
    high = max(low, config.TRANSCRIPT_REQUEST_MAX_DELAY_SECONDS)
    if high <= 0:
        return
    with _pace_lock:
        wait = random.uniform(low, high) - (time.monotonic() - _last_request_at["t"])
        if wait > 0:
            _sleep(wait)
        _last_request_at["t"] = time.monotonic()


def _retry_delay(attempt: int, retry_after: int | None) -> float | None:
    """Seconds to wait after failed attempt number `attempt` (1-based), or None when waiting is pointless.

    Retry-After is honoured. If it asks for longer than TRANSCRIPT_RETRY_MAX_SECONDS we do not sleep a shorter
    time and retry anyway (that would be retrying too early): the caller gives up on this provider instead.
    """
    cap = config.TRANSCRIPT_RETRY_MAX_SECONDS
    if retry_after is not None:
        if retry_after > cap:
            return None
        base = float(retry_after)
    else:
        base = config.TRANSCRIPT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
    base = min(base, cap)
    return base + random.uniform(0, min(1.5, base * 0.25))


# --- the provider interface ---------------------------------------------------------------------------------
OnRetry = Callable[[str, float], None]


class TranscriptProvider(ABC):
    name: str

    @abstractmethod
    def get_transcript(
        self,
        video_id: str,
        original_language: str | None = None,
        video_url: str | None = None,
        on_retry: OnRetry | None = None,
    ) -> TranscriptResult:
        """A TranscriptResult, or one of the TranscriptError subclasses above. Never a raw library error."""


def _primary(code: str) -> str:
    """'en-US' -> 'en', 'hi-IN' -> 'hi'."""
    return (code or "").lower().replace("_", "-").split("-")[0]


@dataclass
class _Candidate:
    transcript: object  # youtube_transcript_api.Transcript
    language_code: str
    is_generated: bool


class YouTubeTranscriptProvider(TranscriptProvider):
    """youtube-transcript-api, directly (optionally through a proxy)."""

    name = "youtube"

    def __init__(self):
        self._api = YouTubeTranscriptApi(proxy_config=_proxy_config())

    # -- selection -------------------------------------------------------------------------------
    @staticmethod
    def _split(transcript_list) -> tuple[list[_Candidate], list[_Candidate]]:
        manual, generated = [], []
        for t in transcript_list:
            entry = _Candidate(t, getattr(t, "language_code", "") or "", bool(getattr(t, "is_generated", False)))
            (generated if entry.is_generated else manual).append(entry)
        return manual, generated

    @staticmethod
    def _pick(pool: list[_Candidate], original_language: str | None) -> _Candidate | None:
        """Original language, then English, then anything else."""
        if not pool:
            return None
        if original_language:
            hit = next((c for c in pool if _primary(c.language_code) == _primary(original_language)), None)
            if hit:
                return hit
        english = next((c for c in pool if _primary(c.language_code) == "en"), None)
        if english:
            return english
        return pool[0]

    # -- one network operation, retried on its own -------------------------------------------------
    def _call(self, fn: Callable, video_id: str, on_retry: OnRetry | None, attempts: list[int]):
        """Run ONE operation (list, or fetch). A temporary failure retries only this operation, so a rate-limited
        fetch never repeats the listing that already succeeded."""
        retries = max(0, config.TRANSCRIPT_MAX_RETRIES)
        failures = 0
        while True:
            attempts[0] += 1
            _pace()
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - every failure is classified below
                err = _classify(exc, video_id)
                retry_after = getattr(err, "retry_after", None)
                _log_attempt(self.name, video_id, attempts[0], err.outcome, retry_after)
                failures += 1
                retryable = isinstance(err, (YouTubeTemporaryRateLimit, TranscriptFetchError))
                wait = _retry_delay(failures, retry_after) if retryable and failures <= retries else None
                if wait is None:
                    if err is exc:
                        raise
                    raise err from exc
                if on_retry:
                    what = "rate-limiting transcript requests" if isinstance(err, YouTubeTemporaryRateLimit) else "not responding"
                    on_retry(f"YouTube is {what}. Waiting {wait:.0f} seconds, then retrying...", wait)
                _sleep(wait)

    # -- retrieval -----------------------------------------------------------------------------------
    def get_transcript(self, video_id, original_language=None, video_url=None, on_retry=None) -> TranscriptResult:
        attempts = [0]
        transcript_list = self._call(lambda: self._api.list(video_id), video_id, on_retry, attempts)

        manual, generated = self._split(transcript_list)
        chosen = self._pick(manual, original_language)
        source_type = "youtube_manual"

        if chosen is None:
            if not (config.ALLOW_GENERATED_TRANSCRIPT and generated):
                available = ", ".join(sorted({c.language_code for c in generated})) or "none"
                log.info("video %s: no acceptable transcript (generated tracks: %s; generated allowed: %s)",
                         video_id, available, config.ALLOW_GENERATED_TRANSCRIPT)
                _log_attempt(self.name, video_id, attempts[0], "unavailable")
                raise TranscriptUnavailable()
            chosen = self._pick(generated, original_language)
            source_type = "youtube_generated"
            log.info("video %s: using a GENERATED transcript (%s)", video_id, chosen.language_code)

        fetched = self._call(chosen.transcript.fetch, video_id, on_retry, attempts)
        lines = _to_lines(fetched)
        if not lines:
            _log_attempt(self.name, video_id, attempts[0], "unavailable")
            raise TranscriptUnavailable("The transcript YouTube returned for this video was empty.")

        _log_attempt(self.name, video_id, attempts[0], "success", source_type=source_type)
        return TranscriptResult(
            lines=lines,
            language=chosen.language_code or "en",
            language_name=getattr(chosen.transcript, "language", chosen.language_code) or chosen.language_code,
            is_generated=chosen.is_generated,
            provider=self.name,
            source_type=source_type,
        )


def _to_lines(fetched) -> list[TranscriptLine]:
    """FetchedTranscript (or a plain list of dicts on older library versions) -> TranscriptLine."""
    lines: list[TranscriptLine] = []
    try:
        for snippet in fetched:
            if isinstance(snippet, dict):
                text, start, duration = snippet.get("text", ""), snippet.get("start", 0.0), snippet.get("duration", 0.0)
            else:
                text, start, duration = snippet.text, snippet.start, snippet.duration
            text = " ".join(str(text).split())
            if not text:
                continue
            start = float(start or 0.0)
            end = start + max(0.0, float(duration or 0.0))
            lines.append(TranscriptLine(start=start, end=max(end, start), text=text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TranscriptParseError("The transcript data from YouTube could not be read.") from exc
    lines.sort(key=lambda ln: ln.start)
    return lines


def _proxy_config():
    """Optional direct HTTP/HTTPS proxy. Empty by default."""
    try:
        if config.YOUTUBE_HTTP_PROXY or config.YOUTUBE_HTTPS_PROXY:
            from youtube_transcript_api.proxies import GenericProxyConfig

            return GenericProxyConfig(
                http_url=config.YOUTUBE_HTTP_PROXY or None,
                https_url=config.YOUTUBE_HTTPS_PROXY or config.YOUTUBE_HTTP_PROXY or None,
            )
    except Exception:  # noqa: BLE001
        log.exception("proxy configuration is invalid; continuing without a proxy")
    return None


# --- the router ---------------------------------------------------------------------------------------------------
_PROVIDER_LABEL = {"youtube": "YouTube"}


def _label(name: str) -> str:
    return _PROVIDER_LABEL.get(name, name)


def _sentence(text: str) -> str:
    """Upper-case only the first letter ('YouTube' must not become 'Youtube', which str.capitalize would do)."""
    return text[:1].upper() + text[1:]


def _noop(*args, **kwargs) -> None:
    pass


class TranscriptRouter:
    """Tries each provider in order and decides what a set of failures means.

    Precedence when nothing succeeded: a parse error or a config error (ours to fix) beats a temporary
    failure (pause), which beats "no transcript" (skip). A temporary failure anywhere means we cannot know
    the video is transcript-less, so it is never reported as a skip.
    """

    def __init__(self, providers: list[TranscriptProvider]):
        self.providers = list(providers)

    def get_transcript(
        self,
        video_id: str,
        original_language: str | None = None,
        video_url: str | None = None,
        on_retry: OnRetry | None = None,
        on_fallback: Callable[[str], None] | None = None,
    ) -> TranscriptResult:
        notify = on_fallback or _noop
        video_url = video_url or f"https://www.youtube.com/watch?v={video_id}"
        temporary: list[TemporaryTranscriptFailure] = []
        unavailable: list[TranscriptUnavailable] = []
        parse_error: TranscriptParseError | None = None
        config_error: TranscriptConfigError | None = None

        for index, provider in enumerate(self.providers):
            following = self.providers[index + 1] if index + 1 < len(self.providers) else None

            cooldown = provider_cooldown_remaining(provider.name)
            if cooldown > 0:  # known-bad route: do not touch it, go straight to the next provider
                _log_attempt(provider.name, video_id, 0, "cooldown_skip", retry_after=cooldown)
                temporary.append(TemporaryTranscriptFailure(
                    f"{_sentence(_label(provider.name))} was blocking this server earlier and is being given a rest "
                    f"(about {max(1, cooldown // 60)} min).", cooldown, provider.name))
                if following:
                    notify(f"{_sentence(_label(provider.name))} is cooling down after being blocked, so going straight to {_label(following.name)}...")
                continue

            try:
                return provider.get_transcript(video_id, original_language, video_url, on_retry)
            except YouTubeAccessDenied:
                raise  # private / restricted: no other provider will do better
            except TranscriptUnavailable as exc:
                unavailable.append(exc)
                if following:
                    notify(f"{_sentence(_label(provider.name))} has no usable transcript for this video. Asking {_label(following.name)}...")
            except TemporaryTranscriptFailure as exc:
                temporary.append(exc)
                self._cool_down(provider, exc)
                if following:
                    notify(f"{exc.reason} Trying {_label(following.name)} instead...")
            except TranscriptConfigError as exc:
                config_error = config_error or exc
            except TranscriptParseError as exc:
                parse_error = parse_error or exc

        if parse_error:
            raise parse_error
        if config_error:
            raise config_error
        if temporary:
            raise self._combine(temporary)
        if unavailable:
            raise unavailable[0]
        raise TranscriptUnavailable()

    @staticmethod
    def _cool_down(provider: TranscriptProvider, exc: TemporaryTranscriptFailure) -> None:
        """Stop hammering a route we now know is bad (direct YouTube only)."""
        if provider.name != "youtube":
            return
        if exc.trips_breaker:
            seconds = config.TRANSCRIPT_PROVIDER_COOLDOWN_SECONDS
        elif isinstance(exc, YouTubeTemporaryRateLimit):
            seconds = int(exc.retry_after or config.TRANSCRIPT_PAUSE_SECONDS)  # retries are used up
        else:
            return
        if seconds > 0:
            _block_provider(provider.name, seconds)
            exc.retry_after = exc.retry_after or seconds

    def _combine(self, failures: list[TemporaryTranscriptFailure]) -> TemporaryTranscriptFailure:
        if len(failures) == 1:
            only = failures[0]
            if len(self.providers) == 1 and only.provider == "youtube":
                only.reason += " No backup transcript provider is configured."
            if only.retry_after is None:
                only.retry_after = config.TRANSCRIPT_PAUSE_SECONDS
            return only
        reasons: list[str] = []
        for failure in failures:
            if failure.reason not in reasons:
                reasons.append(failure.reason)
        waits = [f.retry_after for f in failures if f.retry_after]
        combined = TemporaryTranscriptFailure(
            " ".join(reasons), min(waits) if waits else config.TRANSCRIPT_PAUSE_SECONDS, failures[-1].provider
        )
        combined.outcome = failures[-1].outcome
        return combined


def build_providers() -> list[TranscriptProvider]:
    """Build the single local transcript provider: direct YouTube."""
    return [YouTubeTranscriptProvider()]


# --- module-level entry point: router + small in-process cache -------------------------------
_router: TranscriptRouter | None = None
_router_lock = threading.Lock()

# Small process-local cache for rapid retries/resume inside one Python process. The disk cache below
# is the durable local cache that survives restarts.
_cache: dict[str, TranscriptResult] = {}
_CACHE_MAX = 8
_cache_lock = threading.Lock()


def get_router() -> TranscriptRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = TranscriptRouter(build_providers())
        return _router


def set_router(router: TranscriptRouter | None) -> None:
    """Replace the router (tests). None rebuilds it from the local configuration."""
    global _router
    with _router_lock:
        _router = router


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def youtube_route_status() -> dict:
    """For /health. Reads in-process state only; never makes a YouTube request."""
    remaining = provider_cooldown_remaining("youtube")
    return {"youtube_route": "cooling_down" if remaining else "ok", "youtube_cooldown_seconds": remaining}


def get_transcript(
    video_id: str,
    original_language: str | None = None,
    on_retry: OnRetry | None = None,
    on_fallback: Callable[[str], None] | None = None,
    video_url: str | None = None,
) -> TranscriptResult:
    """Get one transcript, preferring the persistent disk cache before contacting YouTube."""
    # Persistent cache first: repeated local development runs should not spend another YouTube request.
    from backend.ingest import cache

    with _cache_lock:
        cached = _cache.get(video_id)
    if cached is not None:
        log.info("video %s: using transcript from the in-process cache", video_id)
        return cached

    disk_cached = cache.load_transcript(video_id)
    if disk_cached is not None:
        with _cache_lock:
            _cache[video_id] = disk_cached
            while len(_cache) > _CACHE_MAX:
                oldest = next(iter(_cache))
                _cache.pop(oldest, None)
        log.info("video %s: using transcript from the local disk cache", video_id)
        return disk_cached

    result = get_router().get_transcript(video_id, original_language, video_url, on_retry, on_fallback)

    # Cache only successful results. Temporary failures and missing transcripts must be retried/re-evaluated.
    cache.save_transcript(video_id, result)
    with _cache_lock:
        _cache[video_id] = result
        while len(_cache) > _CACHE_MAX:
            oldest = next(iter(_cache))
            _cache.pop(oldest, None)
    return result
