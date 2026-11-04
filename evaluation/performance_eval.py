"""Latency and LLM cost per question.

    python -m evaluation.performance_eval --eval-set evaluation/eval_set.json
    python -m evaluation.performance_eval --repeat 3 --price-in 0.30 --price-out 2.50        # USD per 1M tokens

Runs every question through the app's /ask pipeline in-process and times each stage separately:

  enhance    follow-up resolution and query expansion (1 small LLM call)
  retrieve   embedding the query, hybrid search in Qdrant, fusion
  answer     the answer model (skipped when nothing cleared the threshold: guard 1 never calls the LLM)
  total      the sum; this is what a user waits for before the answer starts to appear

and counts the LLM calls and tokens each question used, per role. Reports p50 / p95 / max (a mean hides the slow
tail) and the share of questions that cost no answer call at all.

NOTES  The first question pays for loading the embedding models, so --warmup (default 2) questions run first and are
not counted; keep it, or the p95 will measure model loading. These are YOUR machine and network: a different laptop
or region will give different numbers. Token counts exist only when the provider reports usage; the output says how
many calls did. Pass --price-in / --price-out (USD per million tokens, from your provider's price list) to get a cost
per 100 questions; no price is built in, because prices change.
"""
from __future__ import annotations

import argparse
import sys
import time

from evaluation.common import DEFAULT_EVAL_SET, EvalSetError, fail, format_table, load_eval_set, mean, percentile, quiet_backend_logging, write_results

STAGES = ("enhance", "retrieve", "answer", "total")


def run_timed(entry: dict, enhance: str, deps: dict, meter=None, clock=time.perf_counter) -> dict:
    """One question through the pipeline, timing each stage.

    `deps` holds the pipeline functions (injected so this can be tested): prepare_query, retrieve_chunks,
    group_into_periods, generate_answer. Returns stage seconds, whether the answer stage ran, and LLM usage by role."""
    before = meter.snapshot() if meter is not None else {}
    t0 = clock()
    standalone, expansions, prior, _note = deps["prepare_query"](entry, enhance)
    t1 = clock()
    chunks = deps["retrieve_chunks"](standalone, entry["source_id"], expansions)
    periods = deps["group_into_periods"](chunks)
    t2 = clock()
    answered = bool(periods)
    output = deps["generate_answer"](standalone, periods, prior)
    t3 = clock()
    usage = {role: u.as_dict() for role, u in meter.since(before).items()} if meter is not None else {}
    return {
        "id": entry["id"], "seconds": {"enhance": t1 - t0, "retrieve": t2 - t1, "answer": t3 - t2, "total": t3 - t0},
        "llm_answer_stage_ran": answered, "n_periods": len(periods), "grounded": bool(output.get("grounded")), "llm_usage": usage,
    }


def summarize(rows: list[dict], price_in: float | None = None, price_out: float | None = None) -> dict:
    """Latency percentiles per stage, LLM calls/tokens per question per role, and an optional cost estimate."""
    latency = {}
    for stage in STAGES:
        values = [r["seconds"][stage] for r in rows]
        latency[stage] = {"p50": percentile(values, 50), "p95": percentile(values, 95), "max": max(values) if values else None,
                          "mean": mean(values)}
    roles = sorted({role for r in rows for role in r["llm_usage"]})
    per_role = {}
    for role in roles:
        usages = [r["llm_usage"].get(role, {}) for r in rows]
        token_calls = sum(u.get("token_calls", 0) for u in usages)
        per_role[role] = {
            "calls_per_question": mean([float(u.get("calls", 0)) for u in usages]),
            "input_tokens_per_question": mean([float(u.get("input_tokens", 0)) for u in usages]) if token_calls else None,
            "output_tokens_per_question": mean([float(u.get("output_tokens", 0)) for u in usages]) if token_calls else None,
            "calls": sum(u.get("calls", 0) for u in usages), "calls_reporting_tokens": token_calls,
        }
    total_calls = sum(v["calls"] for v in per_role.values())
    out = {
        "n_questions": len(rows), "latency_seconds": latency, "llm_by_role": per_role,
        "llm_calls_per_question": total_calls / len(rows) if rows else None,
        "share_without_answer_call": mean([0.0 if r["llm_answer_stage_ran"] else 1.0 for r in rows]),
        "tokens_complete": all(v["calls_reporting_tokens"] == v["calls"] for v in per_role.values()) if per_role else None,
        "cost_per_100_questions_usd": None,
    }
    if price_in is not None and price_out is not None and rows:
        tokens_in = sum(u.get("input_tokens", 0) for r in rows for u in r["llm_usage"].values())
        tokens_out = sum(u.get("output_tokens", 0) for r in rows for u in r["llm_usage"].values())
        out["cost_per_100_questions_usd"] = (tokens_in * price_in + tokens_out * price_out) / 1_000_000 / len(rows) * 100
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.performance_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="full")
    parser.add_argument("--repeat", type=int, default=1, help="run the whole set this many times (steadier percentiles)")
    parser.add_argument("--warmup", type=int, default=2, help="questions to run first and discard (loads the models)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions (rate limits); not counted in the timings")
    parser.add_argument("--price-in", type=float, help="USD per 1M input tokens, to estimate cost")
    parser.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    try:
        entries, info = load_eval_set(args.eval_set)
    except EvalSetError as exc:
        return fail(str(exc))
    if args.limit:
        entries = entries[: args.limit]

    quiet_backend_logging()
    from backend.rag.answer import generate_answer
    from backend.rag.retrieve import group_into_periods, retrieve_chunks
    from backend.util import QuotaExhausted
    from evaluation.common import prepare_query
    from evaluation.metering import Meter, metered

    deps = {"prepare_query": prepare_query, "retrieve_chunks": retrieve_chunks,
            "group_into_periods": group_into_periods, "generate_answer": generate_answer}
    meter, rows, errors, aborted = Meter(), [], [], None
    workload = [e for _ in range(max(1, args.repeat)) for e in entries]
    print(f"Timing {len(workload)} question run(s) after {min(args.warmup, len(entries))} warm-up question(s)...")
    with metered(meter):
        try:
            for entry in entries[: args.warmup]:
                run_timed(entry, args.enhance, deps)  # not recorded: pays for model loading
            for number, entry in enumerate(workload, 1):
                try:
                    rows.append(run_timed(entry, args.enhance, deps, meter))
                except QuotaExhausted:
                    raise
                except Exception as exc:  # noqa: BLE001
                    errors.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
                if args.delay and number < len(workload):
                    time.sleep(args.delay)
        except QuotaExhausted as exc:
            aborted = str(exc)
            print(f"\nQuota exhausted: {exc}\nStopping; partial results are saved.")

    if not rows:
        return fail("no question completed" + (f": {aborted}" if aborted else "; first error: " + (errors[0]["error"] if errors else "?")))
    summary = summarize(rows, args.price_in, args.price_out)
    print()
    print(format_table(["stage", "p50", "p95", "max", "mean"],
                       [[s, *(f"{summary['latency_seconds'][s][k]:.2f}s" for k in ("p50", "p95", "max", "mean"))] for s in STAGES]))
    print()
    print(format_table(["LLM role", "calls/question", "in tokens/q", "out tokens/q"],
                       [[role, f"{v['calls_per_question']:.2f}",
                         "n/a" if v["input_tokens_per_question"] is None else f"{v['input_tokens_per_question']:.0f}",
                         "n/a" if v["output_tokens_per_question"] is None else f"{v['output_tokens_per_question']:.0f}"]
                        for role, v in summary["llm_by_role"].items()]))
    print(f"\n{summary['llm_calls_per_question']:.2f} LLM calls per question; "
          f"{summary['share_without_answer_call'] * 100:.0f}% of questions needed no answer call (nothing retrieved).")
    if summary["tokens_complete"] is False:
        print("note: some calls did not report token usage, so token totals (and any cost) are under-counted.")
    if summary["cost_per_100_questions_usd"] is not None:
        print(f"Estimated cost: ${summary['cost_per_100_questions_usd']:.4f} per 100 questions at your prices.")
    for err in errors:
        print(f"  failed: {err['id']}: {err['error']}")

    path = write_results("performance", {
        "eval_set": info, "params": {"enhance": args.enhance, "repeat": args.repeat, "warmup": args.warmup,
                                      "price_in": args.price_in, "price_out": args.price_out},
        "summary": summary, "errors": errors, "aborted": aborted, "questions": rows,
    }, args.out_dir)
    print(f"\nWrote {path}")
    return 3 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
