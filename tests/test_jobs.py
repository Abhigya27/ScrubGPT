"""Ingestion job state machine: transitions, pause semantics, skips, resumption, queueing. Pipeline and YouTube are faked."""
import threading

import pytest

from backend import config, jobs, library
from backend.ingest import transcript as tp
from backend.ingest.pipeline import SkipVideo
from backend.models import IndexedVideo, Video
from backend.util import QuotaExhausted, UserFacingError


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(jobs, "_pace_between_videos", lambda: None)


def videos(n=3, duration=600):
    return [Video(f"vid{i}", f"Title {i}", duration) for i in range(n)]


def outcome(video, **kwargs):
    fields = dict(video_id=video.video_id, title=video.title, chunks=7, language="en", language_name="English",
                  transcript_source="youtube_manual", translated=False, provider="youtube")
    fields.update(kwargs)
    return IndexedVideo(**fields)


def run(monkeypatch, vids, process, kind="playlist", ident="PLsource"):
    """Run a job synchronously with `process` standing in for pipeline.process_video."""
    monkeypatch.setattr(jobs.metadata, "list_playlist", lambda _ident: vids)
    monkeypatch.setattr(jobs.metadata, "resolve_video", lambda _ident: vids[0])
    monkeypatch.setattr(jobs.pipeline, "process_video", process)
    job = jobs.create_job(ident)
    jobs.run_job(job.job_id, kind, ident)
    return jobs.get_job(job.job_id)


def ok(video, source_id, position, progress, timing):
    return outcome(video)


# --- the happy path ---------------------------------------------------------------------------------------
def test_job_starts_queued():
    job = jobs.create_job("PLsource")
    assert (job.status, job.stage, job.total_videos, job.processed_videos) == ("queued", "queued", 0, 0)


def test_single_video_runs_to_done(monkeypatch):
    job = run(monkeypatch, videos(1), ok, kind="video", ident="vid0")
    assert (job.status, job.stage) == ("done", "done")
    assert (job.total_videos, job.processed_videos) == (1, 1)
    assert job.detail == "Indexed/reused 1 of 1 video."
    assert [v["video_id"] for v in job.indexed] == ["vid0"]


def test_playlist_is_processed_in_order_with_positions(monkeypatch):
    seen = []

    def process(video, source_id, position, progress, timing):
        seen.append((video.video_id, source_id, position))
        return outcome(video)

    job = run(monkeypatch, videos(4), process)
    assert seen == [("vid0", "PLsource", 0), ("vid1", "PLsource", 1), ("vid2", "PLsource", 2), ("vid3", "PLsource", 3)]
    assert job.status == "done" and job.processed_videos == 4


def test_videos_are_never_processed_concurrently(monkeypatch):
    active, peak = [0], [0]

    def process(video, source_id, position, progress, timing):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        active[0] -= 1
        return outcome(video)

    run(monkeypatch, videos(5), process)
    assert peak[0] == 1


def test_progress_and_timings_are_recorded(monkeypatch):
    def process(video, source_id, position, progress, timing):
        progress("transcript", "Fetching", 0.5)
        progress("translating", "Translating", 0.5)
        timing("embedding_seconds", 1.5)
        return outcome(video)

    job = run(monkeypatch, videos(1), process, kind="video", ident="vid0")
    assert job.timings["embedding_seconds"] == 1.5
    assert job.timings["total_seconds"] > 0 and job.timings["transcript_seconds"] >= 0
    assert any("Fetching" in e["msg"] for e in job.events)


def test_library_records_language_chunks_and_translation(monkeypatch):
    stats = {"batches_attempted": 3, "fallback_batches": 1, "cache_hit": False}

    def process(video, source_id, position, progress, timing):
        return outcome(video, language="hi", translated=True, translation=stats)

    run(monkeypatch, videos(1), process, kind="video", ident="vid0")
    entry = library._load()["sources"]["vid0"]["indexed_videos"]["vid0"]
    assert entry["indexed"] is True and entry["language"] == "hi" and entry["chunks"] == 7
    assert entry["translation"] == stats


# --- skips ------------------------------------------------------------------------------------------------
def test_a_skipped_video_is_reported_and_the_job_continues(monkeypatch):
    def process(video, source_id, position, progress, timing):
        if video.video_id == "vid1":
            raise SkipVideo("no manual transcript")
        return outcome(video)

    job = run(monkeypatch, videos(3), process)
    assert job.status == "done"
    assert job.skipped == [{"video_id": "vid1", "title": "Title 1", "reason": "no manual transcript"}]
    assert [v["video_id"] for v in job.indexed] == ["vid0", "vid2"]
    assert job.processed_videos == 3
    assert job.detail == "Indexed/reused 2 of 3 videos."
    assert library._load()["sources"]["PLsource"]["indexed_videos"]["vid1"]["indexed"] is False


# --- pausing ----------------------------------------------------------------------------------------------
def test_a_temporary_youtube_failure_pauses_and_skips_nothing(monkeypatch):
    def process(video, source_id, position, progress, timing):
        if video.video_id == "vid2":
            raise tp.TemporaryTranscriptFailure("YouTube is rate limiting us.", 120)
        return outcome(video)

    job = run(monkeypatch, videos(4), process)
    assert (job.status, job.stage, job.pause_reason) == ("paused", "paused", "youtube")
    assert job.resume_after_seconds == 120 and "rate limiting" in job.error
    assert job.skipped == []  # the failing video and the ones after it are untried, not skipped
    assert job.processed_videos == 2 and [v["video_id"] for v in job.indexed] == ["vid0", "vid1"]


def test_a_pause_without_a_hint_uses_the_default_wait(monkeypatch):
    def process(video, source_id, position, progress, timing):
        raise tp.TemporaryTranscriptFailure("try later")

    job = run(monkeypatch, videos(1), process, kind="video", ident="vid0")
    assert job.status == "paused" and job.resume_after_seconds == config.TRANSCRIPT_PAUSE_SECONDS


def test_an_llm_quota_error_pauses_with_the_llm_reason(monkeypatch):
    def process(video, source_id, position, progress, timing):
        if video.video_id == "vid1":
            raise QuotaExhausted("Daily translation budget reached.", 3600)
        return outcome(video)

    job = run(monkeypatch, videos(3), process)
    assert (job.status, job.pause_reason, job.resume_after_seconds) == ("paused", "llm", 3600)
    assert job.skipped == [] and job.processed_videos == 1


def test_resubmitting_a_paused_source_resumes_where_it_stopped(monkeypatch):
    done = set()
    fail_on = {"vid2"}

    def process(video, source_id, position, progress, timing):
        if video.video_id in done:
            return outcome(video, reused=True)  # already in the index: no transcript request
        if video.video_id in fail_on:
            raise tp.TemporaryTranscriptFailure("rate limited", 60)
        done.add(video.video_id)
        return outcome(video)

    first = run(monkeypatch, videos(4), process)
    assert first.status == "paused" and first.processed_videos == 2
    fail_on.clear()
    second = run(monkeypatch, videos(4), process)
    assert second.status == "done" and second.processed_videos == 4
    assert [v["reused"] for v in second.indexed] == [True, True, False, False]


# --- failures ---------------------------------------------------------------------------------------------
def test_an_unexpected_error_fails_the_job_without_leaking_details(monkeypatch):
    def process(video, source_id, position, progress, timing):
        raise RuntimeError("secret internal detail")

    job = run(monkeypatch, videos(2), process)
    assert (job.status, job.stage) == ("error", "failed")
    assert job.error == "The indexing service encountered an unexpected error."
    assert "secret" not in job.error and job.skipped == []


@pytest.mark.parametrize("exc_type", [tp.TranscriptConfigError, tp.TranscriptParseError])
def test_our_own_transcript_errors_fail_the_job_with_their_reason(monkeypatch, exc_type):
    def process(video, source_id, position, progress, timing):
        raise exc_type("fix the configuration")

    job = run(monkeypatch, videos(1), process, kind="video", ident="vid0")
    assert (job.status, job.error) == ("error", "fix the configuration") and job.skipped == []


def test_a_lookup_failure_rejects_the_submission(monkeypatch):
    def broken(_ident):
        raise UserFacingError("That playlist is private.")

    monkeypatch.setattr(jobs.metadata, "list_playlist", broken)
    job = jobs.create_job("PLsource")
    jobs.run_job(job.job_id, "playlist", "PLsource")
    assert (job.status, job.stage, job.error) == ("error", "rejected", "That playlist is private.")


def test_a_submission_with_no_duration_is_rejected(monkeypatch):
    job = run(monkeypatch, videos(2, duration=0), ok)
    assert job.status == "error" and job.stage == "rejected" and "how long" in job.error


def test_an_exception_in_the_runner_itself_is_contained(monkeypatch):
    monkeypatch.setattr(jobs.metadata, "list_playlist", lambda _ident: 1 / 0)
    job = jobs.create_job("PLsource")
    jobs.run_job(job.job_id, "playlist", "PLsource")  # must not raise
    assert job.status == "error" and job.timings["total_seconds"] >= 0


# --- queueing ---------------------------------------------------------------------------------------------
@pytest.fixture
def no_worker(monkeypatch):
    queued = []
    monkeypatch.setattr(jobs, "enqueue", lambda job_id, kind, ident: queued.append((job_id, kind, ident)))
    return queued


def test_submitting_creates_and_queues_a_job(no_worker):
    job, attached = jobs.submit_ingest("playlist", "PLone")
    assert attached is False and job.status == "queued"
    assert no_worker == [(job.job_id, "playlist", "PLone")]


def test_submitting_the_same_source_again_attaches_to_the_running_job(no_worker):
    first, _ = jobs.submit_ingest("playlist", "PLone")
    second, attached = jobs.submit_ingest("playlist", "PLone")
    assert attached is True and second.job_id == first.job_id and len(no_worker) == 1


def test_a_different_source_is_refused_while_one_is_active(no_worker):
    jobs.submit_ingest("playlist", "PLone")
    with pytest.raises(jobs.IngestBusy):
        jobs.submit_ingest("playlist", "PLtwo")


def test_a_new_source_is_accepted_once_the_active_job_finishes(no_worker):
    first, _ = jobs.submit_ingest("playlist", "PLone")
    first.status = "done"
    second, attached = jobs.submit_ingest("playlist", "PLtwo")
    assert attached is False and second.job_id != first.job_id


def test_a_paused_job_does_not_block_other_sources(no_worker):
    first, _ = jobs.submit_ingest("playlist", "PLone")
    first.status = "paused"
    assert jobs.submit_ingest("playlist", "PLtwo")[1] is False


def test_concurrent_submissions_create_exactly_one_job(no_worker):
    results, barrier = [], threading.Barrier(8)

    def submit():
        barrier.wait()
        try:
            results.append(jobs.submit_ingest("playlist", "PLone"))
        except jobs.IngestBusy:
            results.append(None)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(no_worker) == 1 and len({r[0].job_id for r in results if r}) == 1


def test_finished_jobs_are_forgotten_beyond_the_cap():
    for i in range(jobs._MAX_JOBS + 10):
        jobs.create_job(f"PL{i}").status = "done"
    assert len(jobs._JOBS) <= jobs._MAX_JOBS + 1
