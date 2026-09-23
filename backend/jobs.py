"""In-memory job-status store + a ONE-worker ingestion queue.

No Redis, no Celery: a dict, and a `ThreadPoolExecutor(max_workers=1)`. Only one ingestion job can be doing
transcript work at a time on a backend instance, so two visitors (or one double-click) can never hammer YouTube
or YouTube transcript requests in parallel. Jobs submitted while another runs wait their turn in the executor,
and the API additionally refuses a *different* source while one is active (see `submit_ingest`).

Job status is allowed to vanish on restart, because the Qdrant index does not: re-submitting the same link
picks up where it left off, since finished videos are recognised and skipped without any transcript request.

The three ways a video can fail to be indexed are genuinely different things and must not be flattened into one:

    expected      no usable transcript, private, unusable     -> skipped[], carry on
    temporary     provider blocked / rate-limited / quota      -> PAUSE the job, skip nothing
    a bug         parse error, bad API key, Qdrant error, ...  -> the JOB errors, and says so

Run the backend as ONE process and ONE instance (see README): this state is per process.
"""
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from backend import config, guardrails
from backend.ingest import metadata, pipeline
from backend.ingest import transcript as transcript_provider
from backend import library
from backend.models import JobStatus, Video
from backend.util import QuotaExhausted, UserFacingError, format_duration, set_notifier

log = logging.getLogger(__name__)

_JOBS: dict[str, JobStatus] = {}
_JOBS_LOCK = threading.Lock()
_MAX_EVENTS = 80
_MAX_JOBS = 50  # finished jobs are forgotten beyond this, so a long-lived demo does not grow without bound

BUSY_MESSAGE = "An indexing job is already running. Please wait for it to finish or use the same source to resume."

# Where each pipeline stage sits inside one video's 0..1 progress.
_STAGE_RANGE = {
    "checking": (0.0, 0.05),
    "transcript": (0.05, 0.20),
    "chunking": (0.20, 0.30),
    "translating": (0.30, 0.85),
    "saving": (0.85, 1.0),
}


class IngestBusy(UserFacingError):
    """Another source is being indexed. One ingestion job at a time per backend instance."""


def create_job(source_id: str) -> JobStatus:
    job = JobStatus(
        job_id=uuid.uuid4().hex,
        source_id=source_id,
        status="queued",
        total_videos=0,  # unknown until the playlist has been analysed
        processed_videos=0,
        skipped=[],
        error=None,
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        for old_id in [j.job_id for j in _JOBS.values() if j.status not in ("queued", "processing")][: max(0, len(_JOBS) - _MAX_JOBS)]:
            del _JOBS[old_id]  # dicts keep insertion order, so these are the oldest finished jobs
    return job


def get_job(job_id: str) -> JobStatus | None:
    return _JOBS.get(job_id)


def active_job() -> JobStatus | None:
    """The job that is queued or running, if any."""
    with _JOBS_LOCK:
        return next((j for j in _JOBS.values() if j.status in ("queued", "processing")), None)


# --- the one-worker queue --------------------------------------------------------------------------------
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_submit_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        return _executor


def enqueue(job_id: str, kind: str, ident: str):
    """Hand a job to the single worker. Jobs run strictly one after another."""
    future = _get_executor().submit(run_job, job_id, kind, ident)
    future.add_done_callback(lambda f: f.exception() and log.error("ingest worker crashed: %r", f.exception()))
    return future


def submit_ingest(kind: str, ident: str) -> tuple[JobStatus, bool]:
    """Create and queue a job, or explain why not. Returns (job, attached).

    - nothing active               -> a new job is queued            (attached = False)
    - the SAME source is active    -> that job is returned instead   (attached = True): a double click, a Streamlit
                                      rerun or a second tab watches the running job rather than starting another
    - a DIFFERENT source is active -> IngestBusy (the API answers 409): the single local ingestion worker avoids concurrent YouTube transcript traffic

    Check-and-create is atomic, so two simultaneous requests cannot both start a job.
    """
    with _submit_lock:
        running = active_job()
        if running is not None:
            if running.source_id == ident:
                return running, True
            raise IngestBusy(BUSY_MESSAGE)
        job = create_job(ident)
        enqueue(job.job_id, kind, ident)
        return job, False


def _log(job: JobStatus, msg: str) -> None:
    job.events.append({"t": time.time(), "msg": msg})
    del job.events[:-_MAX_EVENTS]


def _set(job: JobStatus, stage: str, detail: str, fraction: float | None = None, log_it: bool = True, prefix: str = "") -> None:
    job.stage, job.detail = stage, detail
    if fraction is not None:
        low, high = _STAGE_RANGE.get(stage, (0.0, 1.0))
        job.stage_progress = low + (high - low) * max(0.0, min(1.0, fraction))
    if log_it:
        _log(job, f"{prefix}{detail}")


def _reject(job: JobStatus, message: str) -> None:
    job.status, job.stage, job.error, job.detail = "error", "rejected", message, message
    _log(job, f"Can't index this: {message}")


def _pause(job: JobStatus, message: str, retry_after: int | None, reason: str, prefix: str = "") -> None:
    """Stop cleanly and keep everything already done (spec §12).

    Nothing goes into skipped[]: the videos not reached are not unindexable, they
    just have not been tried yet.
    """
    job.status, job.stage = "paused", "paused"
    job.error, job.detail = message, message
    job.resume_after_seconds, job.pause_reason = retry_after, reason
    _log(job, f"{prefix}Paused: {message}")


def _analyze(job: JobStatus, kind: str, ident: str) -> list[Video] | None:
    """Look up the video(s); local mode removes artificial playlist-size caps."""
    started = time.perf_counter()
    try:
        if kind == "playlist":
            _set(job, "analyzing", "Looking up the playlist on YouTube...")
            videos = metadata.list_playlist(ident)
            missing = sum(1 for v in videos if v.duration <= 0)
            _set(job, "analyzing", f"Found {len(videos)} videos in the playlist.")
            if missing:
                _set(job, "measuring", f"Looking up the length of {missing} videos YouTube didn't report...")
                videos = metadata.fill_missing_durations(videos)
        else:
            _set(job, "analyzing", "Looking up the video on YouTube...")
            videos = [metadata.resolve_video(ident)]
    except UserFacingError as exc:
        _reject(job, str(exc))
        return None

    job.timings["metadata_seconds"] += time.perf_counter() - started

    problem = guardrails.submission_length_error(videos)
    if problem:
        _reject(job, problem)
        return None

    job.total_videos = len(videos)
    job.total_seconds = sum(v.duration for v in videos)
    _set(
        job, "measuring",
        f"{len(videos)} video{'s' if len(videos) != 1 else ''}, {format_duration(job.total_seconds)} in total. Ready for local indexing.",
    )
    library.upsert_source(
        ident, kind, [
            {"video_id": v.video_id, "title": v.title, "duration": v.duration}
            for v in videos
        ],
    )
    return videos


def _process_all(job: JobStatus, videos: list[Video]) -> None:
    total = len(videos)
    for position, video in enumerate(videos):  # sequential, in playlist order
        job.current_video, job.current_title, job.stage_progress = position + 1, video.title, 0.0
        prefix = f"[{position + 1}/{total}] " if total > 1 else ""
        _log(job, f"{prefix}Starting: {video.title}")
        stage_started = {"name": None, "at": None}

        def finalize_stage() -> None:
            name = stage_started["name"]
            at = stage_started["at"]
            if name in {"transcript", "translating"} and at is not None:
                key = "transcript_seconds" if name == "transcript" else "translation_seconds"
                job.timings[key] += max(0.0, time.perf_counter() - at)
            stage_started["name"] = stage_started["at"] = None

        def progress(stage, detail, fraction=None, log=True, _prefix=prefix):
            if stage != stage_started["name"]:
                finalize_stage()
                stage_started["name"] = stage
                stage_started["at"] = time.perf_counter()
            _set(job, stage, detail, fraction, log_it=log, prefix=_prefix)

        def timing(name: str, seconds: float) -> None:
            job.timings[name] = job.timings.get(name, 0.0) + max(0.0, seconds)

        try:
            outcome = pipeline.process_video(video, job.source_id, position, progress, timing)

        except transcript_provider.TemporaryTranscriptFailure as exc:
            finalize_stage()
            log.warning("pausing at video %s: %s (provider=%s)", video.video_id, exc.reason, exc.provider)
            _pause(job, exc.reason, exc.retry_after or config.TRANSCRIPT_PAUSE_SECONDS, exc.pause_reason, prefix)
            return

        except (transcript_provider.TranscriptConfigError, transcript_provider.TranscriptParseError) as exc:
            finalize_stage()
            log.exception("transcript retrieval failed for video %s", video.video_id)
            job.status, job.stage = "error", "failed"
            job.error = job.detail = exc.reason
            _log(job, f"{prefix}Failed: {exc.reason}")
            return

        except QuotaExhausted as exc:
            finalize_stage()
            _pause(job, str(exc), exc.retry_after, "llm", prefix)
            return

        except pipeline.SkipVideo as skip:
            finalize_stage()
            job.skipped.append({"video_id": video.video_id, "title": video.title, "reason": skip.reason})
            job.processed_videos += 1
            library.record_video(job.source_id, video.video_id, video.title, video.duration, indexed=False)
            job.stage_progress = 0.0
            _pace_between_videos()
            continue

        except Exception as exc:  # noqa: BLE001
            finalize_stage()
            log.exception("unexpected failure on video %s", video.video_id)
            job.status, job.stage = "error", "failed"
            job.error = "The indexing service encountered an unexpected error."
            job.detail = job.error
            _log(job, f"{prefix}Failed: {job.error} ({type(exc).__name__})")
            return

        finalize_stage()
        job.indexed.append(asdict(outcome))
        job.processed_videos += 1
        library.record_video(
            job.source_id, video.video_id, video.title, video.duration, indexed=True, reused=outcome.reused
        )
        job.stage_progress = 0.0
        _pace_between_videos()

    indexed = total - len(job.skipped)
    job.status, job.stage, job.current_video, job.current_title = "done", "done", 0, ""
    job.detail = f"Indexed/reused {indexed} of {total} video{'s' if total != 1 else ''}."
    _log(job, job.detail)


def _pace_between_videos() -> None:
    """A breather between videos of a playlist, on top of per-request pacing (spec §10)."""
    gap = config.TRANSCRIPT_BETWEEN_VIDEOS_SECONDS
    if gap > 0:
        time.sleep(gap + random.uniform(0, gap * 0.5))


def run_job(job_id: str, kind: str, ident: str) -> None:
    """Runs on the one ingestion worker thread. Never raises."""
    job = _JOBS[job_id]
    job.status = "processing"
    started = time.perf_counter()
    # If the LLM rate limits us, say so in the activity log instead of appearing to hang.
    set_notifier(lambda message: _set(job, job.stage, message))
    try:
        videos = _analyze(job, kind, ident)
        if videos is not None:
            _process_all(job, videos)
    except Exception:  # noqa: BLE001 — the runner itself must never take the server down
        log.exception("job %s failed", job_id)
        job.status, job.stage = "error", "failed"
        job.error = "The indexing service encountered an unexpected error."
        job.detail = job.error
        _log(job, f"Failed: {job.error}")
    finally:
        job.timings["total_seconds"] = time.perf_counter() - started
        # The worker thread is reused for the next job, so don't leave this job's notifier behind. The small
        # transcript LRU is deliberately NOT cleared: if this job paused on an LLM quota, resuming can reuse the
        # transcript it already fetched instead of spending another provider request on it.
        set_notifier(None)
