"""Run the whole evaluation suite with one command.

    python -m evaluation.run_all                              # everything below, on evaluation/eval_set.json
    python -m evaluation.run_all --fast                       # skip the two LLM-heavy steps (answer_quality, performance)
    python -m evaluation.run_all --only ablation,timestamp    # just these
    python -m evaluation.run_all --skip robustness --write-readme --count-tests

Steps, in order (what each measures is in evaluation/README.md):
  ablation        retrieval: Recall@k (hit rate), MRR, Precision@k, nDCG@k, ChunkRecall@k, by category and language
  robustness      retrieval under typos, shouting, filler, truncation          (no LLM)
  timestamp       citation timestamps within +-15s / +-30s of the true moment
  grounding       refusal accuracy and false-refusal rate
  answer_quality  faithfulness, citation support, relevancy, correctness, context precision/recall   (LLM judge, slow)
  performance     latency percentiles, LLM calls and tokens per question                              (LLM)
  translation     fallback rate of batched translation, from data/library.json
  corpus          videos, hours, chunks and languages indexed                                         (read-only)

NOT included, on purpose: the cold-vs-re-ingest benchmark, because it deletes data. Run it yourself:
`python -m evaluation.ingestion_stats --playlist-url URL` (see evaluation/README.md).

Every step runs even if an earlier one fails (a missing gold_timestamp_seconds should not stop the grounding eval). The
exit code is 0 when all steps succeeded, 3 when one ran out of LLM quota, and 1 for any other failure. Run locally, against
your real Qdrant collection.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from evaluation import common
from evaluation.common import DEFAULT_EVAL_SET, EvalSetError, fail, format_table, load_eval_set

# name -> (module, needs the eval set, accepts --delay, LLM heavy)
STEPS = {
    "ablation": ("ablation_eval", True, False, False),
    "robustness": ("robustness_eval", True, False, False),
    "timestamp": ("timestamp_accuracy_eval", True, False, False),
    "grounding": ("grounding_eval", True, True, True),
    "answer_quality": ("answer_quality_eval", True, True, True),
    "performance": ("performance_eval", True, True, True),
    "translation": ("translation_stats", False, False, False),
    "corpus": ("ingestion_stats", False, False, False),
}
FAST_SKIPS = ("answer_quality", "performance")


def select_steps(only: str | None, skip: str | None, fast: bool) -> list[str]:
    """The step names to run, in canonical order. Raises ValueError for an unknown name."""
    names = list(STEPS)
    for label, spec in (("--only", only), ("--skip", skip)):
        unknown = [n for n in (spec or "").replace(" ", "").split(",") if n and n not in STEPS]
        if unknown:
            raise ValueError(f"{label}: unknown step(s) {', '.join(unknown)}. Choose from {', '.join(STEPS)}")
    if only:
        wanted = {n.strip() for n in only.split(",") if n.strip()}
        names = [n for n in names if n in wanted]
    if skip:
        dropped = {n.strip() for n in skip.split(",") if n.strip()}
        names = [n for n in names if n not in dropped]
    if fast:
        names = [n for n in names if n not in FAST_SKIPS]
    return names


def step_args(name: str, eval_set: str, out_dir: str | None, delay: float) -> list[str]:
    _module, needs_eval, takes_delay, _heavy = STEPS[name]
    args = ["--eval-set", eval_set] if needs_eval else []
    if takes_delay and delay:
        args += ["--delay", str(delay)]
    if out_dir:
        args += ["--out-dir", out_dir]
    return args


def run_steps(names: list[str], eval_set: str, out_dir: str | None, delay: float = 0.0, runner=None) -> list[dict]:
    """Run each step, never stopping early. `runner(name, args) -> exit code` is injectable for tests."""

    def real_runner(name: str, args: list[str]) -> int:
        module = importlib.import_module(f"evaluation.{STEPS[name][0]}")
        return module.main(args)

    run = runner or real_runner
    results = []
    for name in names:
        print(f"\n{'=' * 78}\n== {name}\n{'=' * 78}", flush=True)
        before, started = len(common.WRITTEN), time.perf_counter()
        try:
            code = run(name, step_args(name, eval_set, out_dir, delay))
            error = None
        except SystemExit as exc:  # argparse exits instead of returning
            code, error = (exc.code if isinstance(exc.code, int) else 1), "exited"
        except Exception as exc:  # noqa: BLE001 - a crashing step must not hide the others
            code, error = 1, f"{type(exc).__name__}: {exc}"
            print(f"error: {error}", file=sys.stderr)
        results.append({
            "step": name, "code": code or 0, "seconds": round(time.perf_counter() - started, 1), "error": error,
            "files": [p.name for p in common.WRITTEN[before:]],
        })
    return results


def overall_code(results: list[dict]) -> int:
    codes = {r["code"] for r in results}
    return 0 if codes <= {0} else 3 if 3 in codes and codes <= {0, 3} else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_all", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out-dir", help="where the result files go (default evaluation/results)")
    parser.add_argument("--only", help="comma separated step names")
    parser.add_argument("--skip", help="comma separated step names to leave out")
    parser.add_argument("--fast", action="store_true", help=f"skip the LLM-heavy steps ({', '.join(FAST_SKIPS)})")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions in the LLM steps (rate limits)")
    parser.add_argument("--write-readme", action="store_true", help="afterwards, refresh the Results section of README.md")
    parser.add_argument("--count-tests", action="store_true", help="include the pytest test count in that section")
    args = parser.parse_args(argv)

    try:
        names = select_steps(args.only, args.skip, args.fast)
    except ValueError as exc:
        return fail(str(exc))
    if not names:
        return fail("no steps selected.")
    if any(STEPS[n][1] for n in names):
        try:
            entries, info = load_eval_set(args.eval_set)  # fail now, not after the first step has spent LLM quota
        except EvalSetError as exc:
            return fail(str(exc))
        refusals = sum(e["expects_refusal"] for e in entries)
        print(f"Eval set: {len(entries)} questions ({refusals} unanswerable, "
              f"{sum(1 for e in entries if e['gold_answer'])} with a reference answer, "
              f"{sum(1 for e in entries if e['gold_timestamp_seconds'] is not None)} with a gold timestamp).")
        if not any(e["gold_answer"] for e in entries) and "answer_quality" in names:
            print("note: no `gold_answer` anywhere, so answer_quality will skip correctness and context_recall.")
    print("Steps: " + ", ".join(names))

    results = run_steps(names, args.eval_set, args.out_dir, args.delay)
    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(format_table(["step", "result", "seconds", "wrote"], [
        [r["step"], "ok" if r["code"] == 0 else ("quota exhausted" if r["code"] == 3 else f"FAILED ({r['code']})"),
         r["seconds"], ", ".join(r["files"]) or "-"] for r in results], align_right_from=2))

    if args.write_readme or args.count_tests:
        from evaluation import render_results

        extra = ["--results-dir", args.out_dir] if args.out_dir else []
        render_results.main([*extra, *(["--write-readme"] if args.write_readme else []), *(["--count-tests"] if args.count_tests else [])])
    code = overall_code(results)
    if code:
        print("\nSome steps did not complete; see their messages above. The others' results are saved.")
    return code


if __name__ == "__main__":
    sys.exit(main())
