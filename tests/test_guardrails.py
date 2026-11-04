"""Abuse limits: question checks, the sliding-window rate limiter, per-endpoint limits and submission size caps."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend import config, guardrails
from backend.guardrails import SlidingWindowRateLimiter
from backend.models import Video


def request(host="1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=host), headers={})


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(guardrails, "time", fake)
    return fake


# --- local mode ---------------------------------------------------------------------------------------------
def test_local_mode_disables_submission_caps(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MODE", True)
    monkeypatch.setattr(config, "MAX_VIDEOS_PER_JOB", 0)
    monkeypatch.setattr(config, "MAX_TOTAL_HOURS", 0)
    monkeypatch.setattr(config, "MAX_SINGLE_VIDEO_HOURS", 0)
    videos = [Video(str(i), f"Video {i}", 3600) for i in range(101)]
    assert guardrails.submission_length_error(videos) is None
    assert guardrails.submission_length_error([Video("v", "t", 3600 * 20)]) is None


def test_local_mode_still_rejects_what_cannot_be_indexed(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MODE", True)
    assert guardrails.submission_length_error([])  # nothing accessible
    assert "how long" in guardrails.submission_length_error([Video("live", "A live stream", 0)])
    assert "only" in guardrails.submission_length_error([Video("tiny", "Clip", 10)])


def test_local_mode_does_not_rate_limit(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MODE", True)
    for _ in range(500):
        guardrails.rate_limit_ask(request())
        guardrails.rate_limit_search(request())
        guardrails.check_ingest_rate(request())


# --- question checks ----------------------------------------------------------------------------------------
def test_check_question_strips_whitespace():
    assert guardrails.check_question("  what is a hashmap?  ") == "what is a hashmap?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_check_question_rejects_empty(question):
    with pytest.raises(HTTPException) as err:
        guardrails.check_question(question)
    assert err.value.status_code == 422


def test_check_question_rejects_overlong(monkeypatch):
    monkeypatch.setattr(config, "MAX_QUESTION_CHARS", 50)
    with pytest.raises(HTTPException) as err:
        guardrails.check_question("x" * 51)
    assert err.value.status_code == 422
    assert guardrails.check_question("x" * 50) == "x" * 50


# --- sliding window -----------------------------------------------------------------------------------------
def test_sliding_window_allows_up_to_the_limit_then_reports_a_wait(clock):
    limiter = SlidingWindowRateLimiter(2, 10)
    assert limiter.check("a") is None and limiter.check("a") is None
    wait = limiter.check("a")
    assert isinstance(wait, int) and 1 <= wait <= 11  # rounded up: at most window + 1


def test_sliding_window_frees_up_as_time_passes(clock):
    limiter = SlidingWindowRateLimiter(2, 10)
    limiter.check("a"), limiter.check("a")
    assert limiter.check("a") is not None
    clock.now += 11
    assert limiter.check("a") is None


def test_sliding_window_keys_are_independent(clock):
    limiter = SlidingWindowRateLimiter(1, 10)
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None


def test_a_rejected_request_does_not_extend_the_wait(clock):
    limiter = SlidingWindowRateLimiter(1, 10)
    limiter.check("a")
    first_wait = limiter.check("a")
    clock.now += 5
    second_wait = limiter.check("a")
    assert second_wait < first_wait


# --- public mode --------------------------------------------------------------------------------------------
def test_public_mode_returns_429_with_retry_after(monkeypatch, clock):
    monkeypatch.setattr(config, "LOCAL_MODE", False)
    monkeypatch.setattr(guardrails, "_ask_limiter", SlidingWindowRateLimiter(2, 60))
    monkeypatch.setattr(guardrails, "_global_limiter", SlidingWindowRateLimiter(100, 60))
    guardrails.rate_limit_ask(request())
    guardrails.rate_limit_ask(request())
    with pytest.raises(HTTPException) as err:
        guardrails.rate_limit_ask(request())
    assert err.value.status_code == 429
    assert int(err.value.headers["Retry-After"]) >= 1


def test_public_mode_limits_each_client_separately(monkeypatch, clock):
    monkeypatch.setattr(config, "LOCAL_MODE", False)
    monkeypatch.setattr(guardrails, "_ask_limiter", SlidingWindowRateLimiter(1, 60))
    monkeypatch.setattr(guardrails, "_global_limiter", SlidingWindowRateLimiter(100, 60))
    guardrails.rate_limit_ask(request("1.1.1.1"))
    guardrails.rate_limit_ask(request("2.2.2.2"))  # a different client is unaffected
    with pytest.raises(HTTPException):
        guardrails.rate_limit_ask(request("1.1.1.1"))


def test_public_mode_global_limit_applies_to_everyone(monkeypatch, clock):
    monkeypatch.setattr(config, "LOCAL_MODE", False)
    monkeypatch.setattr(guardrails, "_ask_limiter", SlidingWindowRateLimiter(100, 60))
    monkeypatch.setattr(guardrails, "_global_limiter", SlidingWindowRateLimiter(2, 60))
    guardrails.rate_limit_ask(request("1.1.1.1"))
    guardrails.rate_limit_ask(request("2.2.2.2"))
    with pytest.raises(HTTPException) as err:
        guardrails.rate_limit_ask(request("3.3.3.3"))
    assert err.value.status_code == 429


@pytest.fixture
def public_caps(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MODE", False)
    monkeypatch.setattr(config, "MAX_VIDEOS_PER_JOB", 3)
    monkeypatch.setattr(config, "MAX_SINGLE_VIDEO_SECONDS", 3600)
    monkeypatch.setattr(config, "MAX_TOTAL_SECONDS", 7200)
    monkeypatch.setattr(config, "MIN_TOTAL_SECONDS", 90)


def test_public_mode_accepts_a_reasonable_submission(public_caps):
    assert guardrails.submission_length_error([Video("a", "t", 600), Video("b", "t", 900)]) is None


def test_public_mode_rejects_too_many_videos(public_caps):
    message = guardrails.submission_length_error([Video(f"v{i}", "t", 300) for i in range(4)])
    assert message and "3" in message


def test_public_mode_rejects_one_overlong_video(public_caps):
    assert guardrails.submission_length_error([Video("a", "t", 3601)])


def test_public_mode_rejects_too_much_total_audio(public_caps):
    assert guardrails.submission_length_error([Video("a", "t", 3600), Video("b", "t", 3600), Video("c", "t", 600)])


def test_public_mode_rejects_a_submission_that_is_too_short(public_caps):
    assert guardrails.submission_length_error([Video("a", "t", 30)])


def test_public_mode_rejects_an_empty_list(public_caps):
    assert guardrails.submission_length_error([])
