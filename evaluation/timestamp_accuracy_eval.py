"""Timestamp accuracy (spec 1.3): do the returned citations land where the answer really is?

    python -m evaluation.timestamp_accuracy_eval --eval-set evaluation/eval_set.json

Uses the eval-set questions that have `gold_timestamp_seconds` (the moment you checked by hand in the video) and a
known video (`video_id`, or a single-video `source_id`). Each question goes through the app's /search pipeline
in-process (enhance -> hybrid retrieval -> merge into periods). Only periods of the CORRECT video are considered.

  within ±T   the gold timestamp lies inside [start - T, end + T] of at least one period. This is the metric defined
              in the spec; reported for T = 15 s and 30 s.
  start ±T    some period's START is within T seconds of the gold timestamp. Stricter: the start is the timestamp the
              user clicks, and a long merged period can contain the gold moment while starting minutes before it.

A question that retrieves nothing, or nothing from the right video, counts as a miss at every tolerance.
"""
from __future__ import annotations

import argparse
import sys

from evaluation.common import (
    DEFAULT_EVAL_SET, EvalSetError, fail, format_table, load_eval_set, median, pct, quiet_backend_logging,
    run_search, write_results,
)

DEFAULT_TOLERANCES = (15, 30)


def within_window(gold: float, start: float, end: float, tol: float) -> bool:
    """The gold timestamp lies inside [start - tol, end + tol]."""
    return start - tol <= gold <= end + tol


def distance_to_window(gold: float, start: float, end: float) -> float:
    """0 when the gold timestamp is inside [start, end], else the gap to the nearer edge (seconds)."""
    if start <= gold <= end:
        return 0.0
    return float(min(abs(gold - start), abs(gold - end)))


def score_question(gold: int, periods: list[dict], video_id: str, tolerances=DEFAULT_TOLERANCES) -> dict:
    """Judge one question's periods against its gold timestamp."""
    own = [p for p in periods if p.get("video_id") == video_id]
    result = {
        "returned_periods": len(periods), "periods_in_video": len(own),
        "within": {t: any(within_window(gold, p["start_sec"], p["end_sec"], t) for p in own) for t in tolerances},
        "start_within": {t: any(abs(p["start_sec"] - gold) <= t for p in own) for t in tolerances},
        "best_distance_s": min((distance_to_window(gold, p["start_sec"], p["end_sec"]) for p in own), default=None),
        "best_start_offset_s": min((abs(p["start_sec"] - gold) for p in own), default=None),
        "nearest": None,
    }
    if own:
        nearest = min(own, key=lambda p: distance_to_window(gold, p["start_sec"], p["end_sec"]))
        result["nearest"] = {"label": nearest["label"], "start_sec": nearest["start_sec"], "end_sec": nearest["end_sec"]}
    return result


def summarize(rows: list[dict], tolerances=DEFAULT_TOLERANCES) -> dict:
    n = len(rows)

    def frac(kind: str, tol: int):
        return sum(1 for r in rows if r["score"][kind][tol]) / n if n else None

    with_results = [r for r in rows if r["score"]["periods_in_video"]]
    return {
        "n": n,
        "no_result": sum(1 for r in rows if not r["score"]["periods_in_video"]),
        "within": {str(t): frac("within", t) for t in tolerances},
        "start_within": {str(t): frac("start_within", t) for t in tolerances},
        "median_distance_to_window_s": median([r["score"]["best_distance_s"] for r in with_results]),
        "median_start_offset_s": median([r["score"]["best_start_offset_s"] for r in with_results]),
        "median_period_width_s": median(
            [p["end_sec"] - p["start_sec"] for r in rows for p in r.get("periods", [])]
        ),
    }


def select_questions(entries: list[dict]) -> tuple[list[dict], dict]:
    """Questions with a gold timestamp and a known video. Returns (selected, left-out counts)."""
    selected, no_video = [], 0
    for entry in entries:
        if entry["expects_refusal"] or entry["gold_timestamp_seconds"] is None:
            continue
        if not entry["video_id"]:
            no_video += 1
            continue
        selected.append(entry)
    return selected, {"no_video_id": no_video}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.timestamp_accuracy_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--tolerances", default=",".join(map(str, DEFAULT_TOLERANCES)), help="seconds, comma separated (default 15,30)")
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="full",
                        help="query handling; 'full' (default) is what the app does (one LLM call per question)")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    try:
        tolerances = tuple(int(x) for x in args.tolerances.split(","))
        entries, info = load_eval_set(args.eval_set)
    except (ValueError, EvalSetError) as exc:
        return fail(str(exc))
    questions, left_out = select_questions(entries)
    if left_out["no_video_id"]:
        print(f"warning: {left_out['no_video_id']} question(s) have a gold timestamp but no video_id "
              "(and a playlist source_id), so they were skipped. Add `video_id` to them.")
    if not questions:
        return fail("no questions with gold_timestamp_seconds in the eval set (expected ~15-20; see evaluation/README.md).")

    quiet_backend_logging()
    print(f"Running {len(questions)} questions through search (query handling: {args.enhance})...")
    rows, errors = [], []
    for entry in questions:
        try:
            found = run_search(entry, args.enhance)
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        periods = [{k: p[k] for k in ("video_id", "start_sec", "end_sec", "label", "score")} for p in found["periods"]]
        rows.append({
            "id": entry["id"], "question": entry["question"], "category": entry["category"],
            "video_id": entry["video_id"], "gold_timestamp_seconds": entry["gold_timestamp_seconds"],
            "standalone": found["standalone"], "periods": periods,
            "score": score_question(entry["gold_timestamp_seconds"], periods, entry["video_id"], tolerances),
        })
    if not rows:
        return fail("every question failed; first error: " + (errors[0]["error"] if errors else "?"))

    summary = summarize(rows, tolerances)
    print(f"\n{summary['n']} questions ({len(errors)} failed, {summary['no_result']} returned nothing from the right video)\n")
    table = [
        [f"±{t}s", pct(summary["within"][str(t)], 1), pct(summary["start_within"][str(t)], 1)] for t in tolerances
    ]
    print(format_table(["tolerance", "gold inside [start-T, end+T]", "period start within T"], table))
    if summary["median_period_width_s"] is not None:
        print(f"\nMedian returned period width: {summary['median_period_width_s']:.0f}s. "
              "Wide periods make the first column generous; the second shows where the citation actually points.")
    misses = [r for r in rows if not r["score"]["within"][tolerances[-1]]]
    if misses:
        print(f"\nMisses at ±{tolerances[-1]}s:")
        for r in misses:
            nearest = r["score"]["nearest"]
            where = f"nearest period {nearest['label']}" if nearest else "no period from the right video"
            print(f"  {r['id']}: gold {r['gold_timestamp_seconds']}s; {where}\n      {r['question']}")
    for err in errors:
        print(f"  failed: {err['id']}: {err['error']}")

    path = write_results("timestamp", {
        "eval_set": info, "params": {"tolerances": list(tolerances), "enhance": args.enhance},
        "summary": summary, "left_out": left_out, "errors": errors, "questions": rows,
    }, args.out_dir)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
