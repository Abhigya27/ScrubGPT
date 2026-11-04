"""Robustness: does retrieval survive messy questions?

    python -m evaluation.robustness_eval --eval-set evaluation/eval_set.json

Real users mistype, shout, ramble and ask half a question. For every answerable question this script also asks versions
altered in ways a person might, and compares retrieval against the clean question, for dense-only, sparse-only and the
default hybrid weights. The question is: how much does each lose, and does hybrid hold up better than either leg?

  typos       about one word in five has two neighbouring letters swapped (words of 4+ letters only)
  uppercase   THE WHOLE QUESTION IN CAPITALS
  filler      "hey so um, quick question: ... thanks!!"
  truncated   only the first 60% of the words (an unfinished question)
  distractor  an unrelated clause tacked on ("by the way, I like football and bananas")

The perturbations are seeded, so the same eval set always gives the same altered questions. No LLM is used by default
(--enhance none), so this is cheap. A drop of a few points on ~40 questions is within noise; look at the confidence
intervals and at which perturbation hurts which configuration most.

Metrics are the ablation's: Recall@k (hit rate) and MRR on the ranking, no threshold.
"""
from __future__ import annotations

import argparse
import random
import re
import sys

from evaluation import ablation_eval
from evaluation.common import DEFAULT_EVAL_SET, EvalSetError, fail, format_table, load_eval_set, pct, quiet_backend_logging, write_results

KINDS = ("typos", "uppercase", "filler", "truncated", "distractor")
_DISTRACTOR = "by the way, I also like football and bananas"
_WORD = re.compile(r"([A-Za-z]{4,})([^A-Za-z0-9]*)")  # a plain word of 4+ letters plus any trailing punctuation


def perturb(question: str, kind: str, seed: int = 0) -> str:
    """A deterministically altered copy of `question`. `kind` is one of KINDS (or "clean" to return it unchanged)."""
    if kind == "clean":
        return question
    if kind == "uppercase":
        return question.upper()
    if kind == "filler":
        return f"hey so um, quick question: {question.rstrip('?. ')} thanks!!"
    if kind == "distractor":
        return f"{question.rstrip()} {_DISTRACTOR}"
    words = question.split()
    if kind == "truncated":
        keep = max(2, round(len(words) * 0.6))
        return " ".join(words[:keep]) if len(words) > 2 else question
    if kind == "typos":
        rng = random.Random(f"{seed}:{question}")
        out = []
        for word in words:
            match = _WORD.fullmatch(word)  # leave numbers, mixed tokens and short words alone
            if match and rng.random() < 0.2:
                letters, tail = match.groups()
                i = rng.randrange(len(letters) - 1)
                word = letters[:i] + letters[i + 1] + letters[i] + letters[i + 2:] + tail
            out.append(word)
        return " ".join(out)
    raise ValueError(f"unknown perturbation {kind!r}; choose from {', '.join(KINDS)}")


def run_robustness(entries, kinds, fetch, rank_fn, configs, k: int = 5, depth: int = 10, seed: int = 0) -> dict:
    """Score the clean questions and each perturbed version. Returns {"clean": {...}, kind: {...}} where each value is
    {"summary": ablation summary per config, "n": int, "errors": [...], "examples": [(original, perturbed), ...]}."""
    results = {}
    for kind in ("clean", *kinds):
        altered = [{**e, "question": perturb(e["question"], kind, seed)} for e in entries]
        rows, errors = ablation_eval.run_ablation(altered, fetch, rank_fn, lambda *a: [], configs, depth, k)
        results[kind] = {
            "n": len(rows), "errors": errors, "summary": ablation_eval.summarize(rows, configs, "ranking_rank", k, depth),
            "examples": [(e["question"], a["question"]) for e, a in zip(entries[:3], altered[:3])], "rows": rows,
        }
    return results


def degradation(results: dict, configs) -> dict:
    """Recall@k and MRR change (perturbed minus clean) per perturbation per config; negative means it got worse."""
    clean = results["clean"]["summary"]
    out = {}
    for kind, data in results.items():
        if kind == "clean":
            continue
        out[kind] = {}
        for cfg in configs:
            c, p = clean[cfg.name], data["summary"][cfg.name]
            out[kind][cfg.name] = {
                "recall_change": None if c["recall_at_k"] is None or p["recall_at_k"] is None else p["recall_at_k"] - c["recall_at_k"],
                "mrr_change": None if c["mrr"] is None or p["mrr"] is None else p["mrr"] - c["mrr"],
            }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.robustness_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--perturbations", default=",".join(KINDS), help=f"comma separated; default all of: {', '.join(KINDS)}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="none",
                        help="query handling; 'none' (default) needs no LLM, so the perturbation is what retrieval sees")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    kinds = tuple(x.strip() for x in args.perturbations.split(",") if x.strip())
    bad = [x for x in kinds if x not in KINDS]
    if bad:
        return fail(f"unknown perturbation(s): {', '.join(bad)}. Choose from {', '.join(KINDS)}")
    try:
        entries, info = load_eval_set(args.eval_set)
    except EvalSetError as exc:
        return fail(str(exc))
    questions, left_out = ablation_eval.select_questions(entries)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        return fail("no answerable questions with gold_answer_contains in the eval set.")

    quiet_backend_logging()
    from backend import config

    configs = [c for c in ablation_eval.parse_configs("0:1,1:0", config.SPARSE_WEIGHT, config.DENSE_WEIGHT)]
    fetch, rank_fn, _ = ablation_eval.backend_io(args.enhance, no_production=True)
    print(f"Running {len(questions)} questions x {1 + len(kinds)} variants x {len(configs)} configurations...")
    results = run_robustness(questions, kinds, fetch, rank_fn, configs, args.k, args.depth, args.seed)
    changes = degradation(results, configs)

    clean = results["clean"]["summary"]
    print(f"\nClean questions ({results['clean']['n']} scored)")
    print(format_table(["config", f"Recall@{args.k}", "MRR"], [[c.name, pct(clean[c.name]["recall_at_k"], 1), f"{clean[c.name]['mrr']:.3f}"] for c in configs]))
    rows = []
    for kind in kinds:
        for cfg in configs:
            s, d = results[kind]["summary"][cfg.name], changes[kind][cfg.name]
            rows.append([kind, cfg.name, pct(s["recall_at_k"], 1), f"{d['recall_change'] * 100:+.1f} pts", f"{s['mrr']:.3f}", f"{d['mrr_change']:+.3f}"])
    print(f"\nWith each perturbation (change is perturbed minus clean)")
    print(format_table(["perturbation", "config", f"Recall@{args.k}", "change", "MRR", "change"], rows, align_right_from=2))
    worst = min(((d["recall_change"], kind, name) for kind, per in changes.items() for name, d in per.items() if d["recall_change"] is not None), default=None)
    if worst:
        print(f"\nBiggest drop: {worst[2]} under '{worst[1]}' ({worst[0] * 100:+.1f} points).")
    for kind in kinds[:1]:
        for original, altered in results[kind]["examples"]:
            print(f"  e.g. {kind}: {original!r} -> {altered!r}")
    if len(questions) < 30:
        print(f"Note: only {len(questions)} questions; changes of a few points are noise.")

    path = write_results("robustness", {
        "eval_set": info, "params": {"k": args.k, "depth": args.depth, "seed": args.seed, "enhance": args.enhance, "perturbations": list(kinds)},
        "left_out": left_out, "configs": [c.name for c in configs],
        "results": {kind: {key: v for key, v in data.items() if key != "rows"} for kind, data in results.items()}, "changes": changes,
    }, args.out_dir)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
