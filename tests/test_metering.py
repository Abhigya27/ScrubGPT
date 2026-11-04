"""LLM call/token/latency metering without touching the backend's call sites."""
from types import SimpleNamespace

import pytest

from backend import util
from evaluation.metering import Meter, MeteredLLM, Usage, metered, sum_usage


class FakeLLM:
    def __init__(self, usage=None, error=None):
        self.usage, self.error, self.seen = usage, error, []
        self.model_name = "fake"

    def invoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content="the reply", usage_metadata=self.usage)


def test_a_call_is_counted_with_its_tokens_and_time():
    meter = Meter()
    out = MeteredLLM(FakeLLM({"input_tokens": 10, "output_tokens": 3}), "default", meter).invoke([("human", "hi")])
    assert out.content == "the reply"
    usage = meter.by_role()["default"]
    assert (usage["calls"], usage["input_tokens"], usage["output_tokens"], usage["token_calls"]) == (1, 10, 3, 1)
    assert usage["seconds"] >= 0 and usage["errors"] == 0


def test_a_call_without_usage_is_counted_but_not_as_a_token_call():
    meter = Meter()
    MeteredLLM(FakeLLM(None), "judge", meter).invoke([])
    usage = meter.by_role()["judge"]
    assert usage["calls"] == 1 and usage["token_calls"] == 0 and usage["input_tokens"] == 0


def test_a_failing_call_is_recorded_and_still_raises():
    meter = Meter()
    with pytest.raises(RuntimeError):
        MeteredLLM(FakeLLM(error=RuntimeError("boom")), "default", meter).invoke([])
    assert meter.by_role()["default"]["errors"] == 1 and meter.totals()["calls"] == 1


def test_other_attributes_pass_through():
    assert MeteredLLM(FakeLLM(), "default", Meter()).model_name == "fake"


def test_since_reports_only_new_calls_per_role():
    meter = Meter()
    meter.record("default", 1.0, {"input_tokens": 5, "output_tokens": 1})
    mark = meter.snapshot()
    meter.record("judge", 0.5, None)
    meter.record("judge", 0.5, None)
    meter.record("default", 2.0, {"input_tokens": 7, "output_tokens": 2})
    delta = meter.since(mark)
    assert delta["judge"].calls == 2 and delta["default"].calls == 1 and delta["default"].input_tokens == 7
    assert meter.since(meter.snapshot()) == {}


def test_totals_add_up_across_roles():
    meter = Meter()
    meter.record("default", 1.0, {"input_tokens": 5, "output_tokens": 1})
    meter.record("enhance", 1.0, {"input_tokens": 3, "output_tokens": 1})
    assert meter.totals()["calls"] == 2 and meter.totals()["input_tokens"] == 8
    assert sum_usage(meter.snapshot()).calls == 2


def test_metered_wraps_get_llm_for_the_duration_of_the_block(monkeypatch):
    fake = FakeLLM({"input_tokens": 4, "output_tokens": 2})
    monkeypatch.setattr(util, "get_llm", lambda role="default": fake)
    patched_before = util.get_llm
    meter = Meter()
    with metered(meter):
        assert util.get_llm is not patched_before
        assert util._chat("system prompt", "user prompt", role="judge") == "the reply"  # the app's real entry point
    assert util.get_llm is patched_before
    assert meter.by_role()["judge"]["calls"] == 1 and meter.by_role()["judge"]["input_tokens"] == 4
    assert fake.seen == [[("system", "system prompt"), ("human", "user prompt")]]


def test_metered_restores_get_llm_even_if_the_block_raises(monkeypatch):
    monkeypatch.setattr(util, "get_llm", lambda role="default": FakeLLM())
    before = util.get_llm
    with pytest.raises(ValueError):
        with metered(Meter()):
            raise ValueError("inside")
    assert util.get_llm is before


def test_usage_arithmetic():
    a, b = Usage(calls=3, seconds=2.0), Usage(calls=1, seconds=0.5)
    assert a.minus(b).calls == 2
    a.add(b)
    assert a.calls == 4 and a.as_dict()["seconds"] == 2.5
