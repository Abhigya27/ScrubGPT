"""Retrieval evaluation: measure how good the returned sections are, and find better settings.

You label a set of questions (where the answer really is), then:

    python -m evaluation.run_eval suggest --source SRC --questions questions.txt   # helps you label
    python -m evaluation.run_eval run     --set evaluation/eval_set.json           # score the current settings
    python -m evaluation.run_eval sweep   --set evaluation/eval_set.json           # try many settings, rank them

Add --enhance to `run` or `sweep` to search with LLM query enhancement (rag/enhance.py) as the app does, so you can
compare the two scores on YOUR videos. It makes one small LLM call per question (uses your provider quota).

Run from the repo root with your .env in place (it queries the same Qdrant index the app uses).
Questions are searched as written, without conversation history (navigational words are still
stripped, exactly as the app does it). MIN_DENSE_SIMILARITY is held at its configured value
throughout a sweep; only threshold, floor and terms are varied.

Scoring, per question, on the merged SECTIONS the user would actually see:
  positive question (expected ranges given)
      precision = returned sections overlapping an expected range / returned sections
      recall    = expected ranges overlapped by at least one returned section / expected ranges
  negative question (expected = [], the videos can't answer it)
      correct only if NOTHING is returned (this is what lets the app refuse instead of guessing)
"""
import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

GRID = {
    "threshold": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "floor": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
    "terms": [0.3, 0.6, 1.0, 1.5],
}


# --- dataset ---------------------------------------------------------------------------
def parse_ts(value) -> int:
    """41, "41" (seconds), "41:02" (mm:ss) or "1:02:30" (h:mm:ss) -> seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    try:
        nums = [int(p) for p in str(value).strip().split(":")]
    except ValueError:
        raise ValueError(f"bad timestamp {value!r} (use mm:ss, h:mm:ss or seconds)") from None
    if not 1 <= len(nums) <= 3:
        raise ValueError(f"bad timestamp {value!r} (use mm:ss, h:mm:ss or seconds)")
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def load_set(path) -> dict:
    """Read and validate an eval set. Timestamps become seconds."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "REPLACE" in json.dumps(raw):
        raise ValueError(f"{path} still contains REPLACE placeholders. Copy the example and fill in your own labels.")
    source_id = raw.get("source_id")
    if not source_id:
        raise ValueError("the eval set needs a source_id (the playlist id or video id you indexed)")
    questions = []
    for i, item in enumerate(raw.get("questions", []), 1):
        if not item.get("q"):
            raise ValueError(f"question {i} has no 'q'")
        expected = []
        for exp in item.get("expected", []):
            try:
                start, end = parse_ts(exp["start"]), parse_ts(exp["end"])
                video_id = exp["video_id"]
            except KeyError as missing:
                raise ValueError(f"question {i}: an expected range is missing {missing}") from None
            if end <= start:
                raise ValueError(f"question {i}: end must be after start ({exp['start']} .. {exp['end']})")
            expected.append({"video_id": video_id, "start": start, "end": end})
        questions.append({"q": item["q"], "expected": expected})
    if not questions:
        raise ValueError("the eval set has no questions")
    return {"source_id": source_id, "questions": questions}


# --- metrics (pure) --------------------------------------------------------------------------
def overlaps(period: dict, expected: dict) -> bool:
    return period["video_id"] == expected["video_id"] and max(period["start_sec"], expected["start"]) < min(period["end_sec"], expected["end"])


def score_question(periods: list[dict], expected: list[dict]) -> dict:
    if expected:
        hits = sum(1 for p in periods if any(overlaps(p, e) for e in expected))
        covered = sum(1 for e in expected if any(overlaps(p, e) for p in periods))
        return {
            "kind": "positive", "returned": len(periods),
            "precision": hits / len(periods) if periods else 0.0,  # returning nothing for a real question is a miss
            "recall": covered / len(expected),
        }
    return {"kind": "negative", "returned": len(periods), "correct": not periods}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["kind"] == "positive"]
    neg = [r for r in rows if r["kind"] == "negative"]
    precision, recall = _mean(r["precision"] for r in pos), _mean(r["recall"] for r in pos)
    f1 = None
    if pos:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    neg_acc = _mean(1.0 if r["correct"] else 0.0 for r in neg)
    parts = [x for x in (f1, neg_acc) if x is not None]
    return {
        "precision": precision, "recall": recall, "f1": f1, "negative_accuracy": neg_acc,
        "objective": sum(parts) / len(parts) if parts else 0.0,  # equal weight: finding answers, and refusing when there are none
        "avg_sections": _mean(r["returned"] for r in rows) or 0.0,
        "n_positive": len(pos), "n_negative": len(neg),
    }


def evaluate(questions: list[dict], retrieve_fn) -> list[dict]:
    rows = []
    for q in questions:
        periods = retrieve_fn(q["q"])
        row = score_question(periods, q["expected"])
        row["q"], row["expected"], row["got"] = q["q"], q["expected"], [(p["video_id"], p["label"]) for p in periods]
        rows.append(row)
    return rows


def sweep(candidates, questions, fuse, group, config, source_id, max_results, grid=None) -> list[dict]:
    """Re-score every question under every settings combination WITHOUT re-querying Qdrant.

    `candidates[i]` is (dense_hits, sparse_hits, n_points) for questions[i]. Restores `config` afterwards.
    Returns settings + metrics, best first (ties: fewer sections, then closest to the current settings).
    """
    grid = grid or GRID
    saved = (config.RELATIVE_SCORE_FLOOR, config.SPARSE_HALF_SATURATION_TERMS)
    results = []
    try:
        for threshold, floor, terms in itertools.product(grid["threshold"], grid["floor"], grid["terms"]):
            config.RELATIVE_SCORE_FLOOR, config.SPARSE_HALF_SATURATION_TERMS = floor, terms
            rows = []
            for q, (dense, sparse, n_points) in zip(questions, candidates):
                periods = group(fuse(dense, sparse, threshold, max_results, source_id, n_points))
                rows.append(score_question(periods, q["expected"]))
            results.append({"threshold": threshold, "floor": floor, "terms": terms, **aggregate(rows)})
    finally:
        config.RELATIVE_SCORE_FLOOR, config.SPARSE_HALF_SATURATION_TERMS = saved
    # Best score first. Ties are common with a small eval set, so break them toward the settings closest to the
    # current ones (in units of each knob's grid step): only recommend a change the data actually supports.
    now = (config.RETRIEVAL_THRESHOLD, saved[0], saved[1])
    scale = (0.25, 0.5, 1.0)

    def distance(r):
        return sum(abs(r[k] - c) / sc for k, c, sc in zip(("threshold", "floor", "terms"), now, scale))

    results.sort(key=lambda r: (-r["objective"], r["avg_sections"], distance(r)))
    return results


# --- commands (need the backend and a live index) ---------------------------------------------
def _backend():
    logging.getLogger("backend").setLevel(logging.WARNING)  # the per-search diagnostic line would flood the report
    from backend import config
    from backend.index import qdrant_store
    from backend.rag.retrieve import group_into_periods

    logging.getLogger("backend.index.qdrant_store").setLevel(logging.WARNING)
    return config, qdrant_store, group_into_periods


def _pct(x):
    return "  n/a" if x is None else f"{x * 100:5.0f}%"


def _print_summary(agg: dict) -> None:
    print(f"  precision {_pct(agg['precision'])}   recall {_pct(agg['recall'])}   F1 {_pct(agg['f1'])}   "
          f"correct refusals {_pct(agg['negative_accuracy'])}   avg sections/question {agg['avg_sections']:.1f}")
    print(f"  overall score {_pct(agg['objective'])}  ({agg['n_positive']} answerable, {agg['n_negative']} unanswerable questions)")


def _expansions(question: str, enabled: bool) -> list[str]:
    """The extra queries the app would add for this question (none unless --enhance). No chat history: each eval
    question is asked cold, exactly as the rest of this tool treats it."""
    if not enabled:
        return []
    from backend.rag.enhance import enhance_query

    return enhance_query([], question).expansions


def cmd_run(args) -> int:
    config, qdrant_store, group = _backend()
    data = load_set(args.set)
    enhance = getattr(args, "enhance", False)
    rows = evaluate(
        data["questions"],
        lambda q: group(qdrant_store.hybrid_search(
            q, data["source_id"], config.RETRIEVAL_THRESHOLD, config.MAX_RESULTS, _expansions(q, enhance)
        )),
    )
    print(f"\nSettings: RETRIEVAL_THRESHOLD={config.RETRIEVAL_THRESHOLD}  RELATIVE_SCORE_FLOOR={config.RELATIVE_SCORE_FLOOR}  "
          f"SPARSE_HALF_SATURATION_TERMS={config.SPARSE_HALF_SATURATION_TERMS}  query enhancement: {'ON' if enhance else 'off'}\n")
    for r in rows:
        if r["kind"] == "positive":
            mark = "ok " if r["precision"] == 1.0 and r["recall"] == 1.0 else "-- "
            detail = f"precision {r['precision']:.2f} recall {r['recall']:.2f}"
        else:
            mark, detail = ("ok " if r["correct"] else "-- "), ("correctly returned nothing" if r["correct"] else "should have returned nothing")
        print(f"{mark}{r['q']}\n     {detail}\n     got: {r['got'] or 'nothing'}")
    print("\nSummary")
    _print_summary(aggregate(rows))
    return 0


def cmd_sweep(args) -> int:
    config, qdrant_store, group = _backend()
    data = load_set(args.set)
    enhance = getattr(args, "enhance", False)
    print(f"Fetching candidates for {len(data['questions'])} questions (query enhancement {'ON' if enhance else 'off'})...")
    candidates = [
        qdrant_store.fetch_candidates(q["q"], data["source_id"], config.MAX_RESULTS, _expansions(q["q"], enhance))
        for q in data["questions"]
    ]
    grid = {k: sorted(set(v) | {cur}) for (k, v), cur in zip(GRID.items(), (config.RETRIEVAL_THRESHOLD, config.RELATIVE_SCORE_FLOOR, config.SPARSE_HALF_SATURATION_TERMS))}
    results = sweep(candidates, data["questions"], qdrant_store._fuse_and_threshold, group, config, data["source_id"], config.MAX_RESULTS, grid)

    print(f"\nTried {len(results)} settings combinations. Best first:\n")
    print("  threshold  floor  terms |  score  precision  recall  refusals  sections")
    for r in results[: args.top]:
        print(f"     {r['threshold']:.2f}    {r['floor']:.2f}   {r['terms']:.1f}  | {_pct(r['objective'])}   {_pct(r['precision'])}   {_pct(r['recall'])}  {_pct(r['negative_accuracy'])}    {r['avg_sections']:.1f}")
    current = next(r for r in results if (r["threshold"], r["floor"], r["terms"]) == (config.RETRIEVAL_THRESHOLD, config.RELATIVE_SCORE_FLOOR, config.SPARSE_HALF_SATURATION_TERMS))
    print(f"\nYour current settings ({current['threshold']:.2f} / {current['floor']:.2f} / {current['terms']:.1f}) score {_pct(current['objective'])}.")
    best = results[0]
    near = sum(1 for r in results if r["objective"] >= best["objective"] - 0.02)
    print(f"{near} of {len(results)} combinations are within 2 points of the best.")
    print("\nTo try the best one, put this in .env and restart the backend:")
    print(f"  RETRIEVAL_THRESHOLD={best['threshold']}\n  RELATIVE_SCORE_FLOOR={best['floor']}\n  SPARSE_HALF_SATURATION_TERMS={best['terms']}")
    if len(data["questions"]) < 20:
        print(f"\nNote: only {len(data['questions'])} questions. With so few, the 'best' settings mostly memorise them. Aim for 20-30, "
              "and prefer a setting in the middle of a plateau of good scores over a lone peak.")
    return 0


def cmd_suggest(args) -> int:
    """Show candidate sections per question (deliberately permissive) so you can pick the correct ones."""
    config, qdrant_store, group = _backend()
    saved = config.RELATIVE_SCORE_FLOOR
    config.RELATIVE_SCORE_FLOOR = 0.0
    questions = [line.strip() for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    skeleton = {"source_id": args.source, "questions": []}
    try:
        for q in questions:
            periods = group(qdrant_store.hybrid_search(q, args.source, 0.80, config.MAX_RESULTS))
            print(f"\n{q}")
            for p in periods:
                print(f"   {p['video_id']}  {p['label']:>16}  (distance {p['distance']:.2f})  {p['snippet'][:90]}")
            if not periods:
                print("   (nothing)")
            skeleton["questions"].append({"q": q, "expected": []})
    finally:
        config.RELATIVE_SCORE_FLOOR = saved
    print("\nCopy this into evaluation/eval_set.json, then fill in each `expected` with the ranges that are truly right")
    print("(leave it [] for questions the videos do not answer):\n")
    print(json.dumps(skeleton, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="score the current settings")
    run_p.add_argument("--set", required=True)
    run_p.add_argument("--enhance", action="store_true", help="search with LLM query enhancement (one LLM call per question)")
    sweep_p = sub.add_parser("sweep", help="try many settings and rank them")
    sweep_p.add_argument("--set", required=True)
    sweep_p.add_argument("--top", type=int, default=10)
    sweep_p.add_argument("--enhance", action="store_true", help="search with LLM query enhancement (one LLM call per question)")
    sug_p = sub.add_parser("suggest", help="print candidate sections for a list of questions, to help you label")
    sug_p.add_argument("--source", required=True)
    sug_p.add_argument("--questions", required=True)
    args = parser.parse_args(argv)
    try:
        return {"run": cmd_run, "sweep": cmd_sweep, "suggest": cmd_suggest}[args.cmd](args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
