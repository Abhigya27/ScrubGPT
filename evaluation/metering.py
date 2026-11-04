"""Count LLM calls, tokens and latency without touching the backend's call sites.

Every LLM use in the app (answering, query enhancement, translation) goes through `backend.util._chat`, which asks
`backend.util.get_llm(role)` for a model and calls `.invoke(messages)`. `metered()` swaps `get_llm` for a version that
returns a thin proxy recording each invoke, then restores it. Nothing else changes: same model, same retries.

    meter = Meter()
    with metered(meter):
        ...run the pipeline...
    meter.totals()       # {"calls": 3, "input_tokens": 1500, ...}
    meter.by_role()      # {"default": {...}, "enhance": {...}, "judge": {...}}

Token counts come from the provider's `usage_metadata` when it reports one. Calls without it are counted in `calls`
but not in `token_calls`, so a total that silently covers only some calls can be told apart from a complete one.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass

_FIELDS = ("calls", "errors", "seconds", "input_tokens", "output_tokens", "token_calls")


@dataclass
class Usage:
    calls: int = 0
    errors: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    token_calls: int = 0  # calls that reported token usage

    def add(self, other: "Usage") -> None:
        for name in _FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def minus(self, other: "Usage") -> "Usage":
        return Usage(**{name: getattr(self, name) - getattr(other, name) for name in _FIELDS})

    def as_dict(self) -> dict:
        return asdict(self)


class Meter:
    """Thread-safe accumulator, keyed by LLM role ('default' = answers, 'enhance', 'translate', 'judge')."""

    def __init__(self):
        self._roles: dict[str, Usage] = defaultdict(Usage)
        self._lock = threading.Lock()

    def record(self, role: str, seconds: float, usage: dict | None, error: bool = False) -> None:
        with self._lock:
            u = self._roles[role]
            u.calls += 1
            u.seconds += seconds
            u.errors += 1 if error else 0
            if usage:
                u.input_tokens += int(usage.get("input_tokens") or 0)
                u.output_tokens += int(usage.get("output_tokens") or 0)
                u.token_calls += 1

    def snapshot(self) -> dict[str, Usage]:
        """A copy of the current per-role totals, to diff against later (per-question accounting)."""
        with self._lock:
            return {role: Usage(**asdict(u)) for role, u in self._roles.items()}

    def since(self, earlier: dict[str, Usage]) -> dict[str, Usage]:
        """What was recorded after `earlier` (a snapshot()), per role, omitting roles with no new calls."""
        now = self.snapshot()
        out = {}
        for role, usage in now.items():
            delta = usage.minus(earlier.get(role, Usage()))
            if delta.calls:
                out[role] = delta
        return out

    def by_role(self) -> dict[str, dict]:
        return {role: u.as_dict() for role, u in self.snapshot().items()}

    def totals(self) -> dict:
        total = Usage()
        for usage in self.snapshot().values():
            total.add(usage)
        return total.as_dict()


class MeteredLLM:
    """Forwards to the real chat model and records each invoke. Anything else is passed straight through."""

    def __init__(self, llm, role: str, meter: Meter):
        self._llm, self._role, self._meter = llm, role, meter

    def invoke(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            reply = self._llm.invoke(*args, **kwargs)
        except Exception:
            self._meter.record(self._role, time.perf_counter() - started, None, error=True)
            raise
        usage = getattr(reply, "usage_metadata", None)
        self._meter.record(self._role, time.perf_counter() - started, usage if isinstance(usage, dict) else None)
        return reply

    def __getattr__(self, name):
        return getattr(self._llm, name)


@contextmanager
def metered(meter: Meter):
    """Record every LLM call made inside the block (including by other threads)."""
    from backend import util

    original = util.get_llm
    util.get_llm = lambda role="default": MeteredLLM(original(role), role, meter)
    try:
        yield meter
    finally:
        util.get_llm = original


def sum_usage(usages: dict[str, Usage]) -> Usage:
    total = Usage()
    for usage in usages.values():
        total.add(usage)
    return total
