"""performance_eval: stage timing, percentile summaries, LLM usage per question, and the command."""
import json

import pytest

from evaluation import common, performance_eval as P
from evaluation.metering import Meter


class Clock:
    """Returns 0, 1, 3, 6, ... : each call advances by one more second, so stage durations are 1, 2, 3."""

    def __init__(self):
        self.t, self.step = 0.0, 0

    def __call__(self):
        value = self.t
        self.step += 1
        self.t += self.step
        return value


def deps(periods=("p",), grounded=True, meter=None):
    def generate(standalone, found, prior):
        if meter and found:
            meter.record("default", 0.5, {"input_tokens": 100, "output_tokens": 20})
        return {"answer": "x", "grounded": grounded and bool(found)}

    def prepare(entry, mode):
        if meter:
            meter.record("enhance", 0.1, {"input_tokens": 30, "output_tokens": 5})
        return entry["question"], [], [], ""

    return {"prepare_query": prepare, "retrieve_chunks": lambda q, s, x: list(periods),
            "group_into_periods": lambda chunks: list(chunks), "generate_answer": generate}


ENTRY = {"id": "e1", "source_id": "PL", "question": "q?"}


def test_stage_durations_come_from_the_clock():
    row = P.run_timed(ENTRY, "full", deps(), clock=Clock())
    assert row["seconds"] == {"enhance": 1.0, "retrieve": 2.0, "answer": 3.0, "total": 6.0}
    assert row["llm_answer_stage_ran"] is True and row["n_periods"] == 1


def test_llm_usage_is_attributed_to_the_question_that_made_it():
    meter = Meter()
    meter.record("default", 1.0, {"input_tokens": 999, "output_tokens": 999})  # an earlier question's call
    row = P.run_timed(ENTRY, "full", deps(meter=meter), meter=meter, clock=Clock())
    assert row["llm_usage"]["default"]["calls"] == 1 and row["llm_usage"]["default"]["input_tokens"] == 100
    assert row["llm_usage"]["enhance"]["calls"] == 1


def test_a_question_with_no_periods_makes_no_answer_call():
    meter = Meter()
    row = P.run_timed(ENTRY, "full", deps(periods=(), meter=meter), meter=meter, clock=Clock())
    assert row["llm_answer_stage_ran"] is False and "default" not in row["llm_usage"] and "enhance" in row["llm_usage"]


def make_row(total, answered=True, usage=None):
    return {"id": "x", "seconds": {"enhance": total / 4, "retrieve": total / 4, "answer": total / 2, "total": total},
            "llm_answer_stage_ran": answered, "n_periods": 1, "grounded": True, "llm_usage": usage or {}}


def usage(calls=1, i=100, o=20, token_calls=None):
    return {"calls": calls, "input_tokens": i, "output_tokens": o, "token_calls": calls if token_calls is None else token_calls}


def test_latency_percentiles_expose_the_slow_tail():
    rows = [make_row(t) for t in (1, 1, 1, 1, 1, 1, 1, 1, 1, 10)]
    s = P.summarize(rows)["latency_seconds"]["total"]
    assert s["p50"] == 1 and s["max"] == 10 and s["p95"] > 5 and s["mean"] == pytest.approx(1.9)


def test_llm_calls_and_tokens_per_question_by_role():
    rows = [make_row(2, usage={"default": usage(), "enhance": usage(i=30, o=5)}), make_row(2, answered=False, usage={"enhance": usage(i=30, o=5)})]
    s = P.summarize(rows)
    assert s["llm_by_role"]["default"]["calls_per_question"] == 0.5
    assert s["llm_by_role"]["enhance"]["calls_per_question"] == 1.0 and s["llm_by_role"]["enhance"]["input_tokens_per_question"] == 30
    assert s["llm_calls_per_question"] == 1.5 and s["share_without_answer_call"] == 0.5 and s["tokens_complete"] is True


def test_cost_needs_both_prices():
    rows = [make_row(2, usage={"default": usage(i=1000, o=200)})]
    assert P.summarize(rows)["cost_per_100_questions_usd"] is None
    assert P.summarize(rows, price_in=1.0)["cost_per_100_questions_usd"] is None
    # (1000 * 1.0 + 200 * 2.0) per million tokens = $0.0014 per question = $0.14 per 100
    assert P.summarize(rows, 1.0, 2.0)["cost_per_100_questions_usd"] == pytest.approx(0.14)


def test_missing_token_reports_are_flagged_rather_than_counted_as_zero():
    rows = [make_row(2, usage={"default": usage(token_calls=0, i=0, o=0)})]
    s = P.summarize(rows)
    assert s["tokens_complete"] is False and s["llm_by_role"]["default"]["input_tokens_per_question"] is None


def test_summary_of_nothing_does_not_crash():
    s = P.summarize([])
    assert s["n_questions"] == 0 and s["llm_calls_per_question"] is None


def test_command_end_to_end(tmp_path, monkeypatch):
    from backend.rag import answer, retrieve

    monkeypatch.setattr(common, "prepare_query", lambda e, mode: (e["question"], [], [], ""))
    monkeypatch.setattr(retrieve, "retrieve_chunks", lambda q, s, x=(), **kw: [{"video_id": "v"}])
    monkeypatch.setattr(retrieve, "group_into_periods", lambda chunks: chunks)
    monkeypatch.setattr(answer, "generate_answer", lambda q, periods, prior: {"answer": "x", "grounded": True})
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([{"id": "a", "source_id": "PL", "question": "q1?"}, {"id": "b", "source_id": "PL", "question": "q2?"}]), encoding="utf-8")
    out = tmp_path / "r"
    assert P.main(["--eval-set", str(path), "--out-dir", str(out), "--warmup", "1", "--repeat", "2"]) == 0
    data = json.loads(next(out.glob("performance_*.json")).read_text(encoding="utf-8"))
    assert data["summary"]["n_questions"] == 4          # 2 questions x 2 repeats; the warm-up run is not counted
    assert data["summary"]["latency_seconds"]["total"]["p50"] >= 0
