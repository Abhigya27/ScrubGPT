"""Answer quality: faithfulness, citation support, relevancy, correctness and context quality (LLM-as-judge).

    python -m evaluation.answer_quality_eval --eval-set evaluation/eval_set.json
    python -m evaluation.answer_quality_eval --metrics faithfulness,answer_relevancy --limit 10     # cheaper trial
    python -m evaluation.answer_quality_eval --audit 8                                              # + a sample to check by hand

Each answerable question runs through the app's /ask pipeline in-process. A separate judge model (role "judge", set with
EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL) then scores the result. The metrics are defined in evaluation/judge.py:

  faithfulness       are the answer's claims supported by the retrieved excerpts?        (hallucination check)
  citation_support   does each cited sentence really follow from the excerpts it cites?  (wrong-citation check)
  answer_relevancy   does the answer address the question?
  correctness        does it match your `gold_answer`?                                  (needs gold_answer)
  context_precision  how much of the retrieved context is useful for the question?
  context_recall     does the retrieved context contain what `gold_answer` says?        (needs gold_answer)

Questions the app refused are not scored on the answer metrics (they are the grounding eval's business) but are counted
in `answered_rate`; their retrieved context is still scored. Questions with `expects_refusal` are skipped.

COST: about one judge call per metric per question (up to 6), on top of the app's own 1-2. Use --metrics / --limit to
trial it. If the provider's quota runs out the run stops, saves what it has and exits with code 3.

Read the scores with care: they come from a model. Use a judge that differs from the answer model, look at the
confidence intervals, and check a sample by hand with --audit.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from evaluation.common import (
    DEFAULT_EVAL_SET, EvalSetError, bootstrap_ci, fail, format_table, load_eval_set, mean, pct,
    quiet_backend_logging, run_search, write_results,
)
from evaluation.judge import Judge, JudgeError, cited_sentences

QUOTA_EXIT_CODE = 3
ALL_METRICS = ("faithfulness", "citation_support", "answer_relevancy", "correctness", "context_precision", "context_recall")
NEEDS_REFERENCE = {"correctness", "context_recall"}
ANSWER_METRICS = {"faithfulness", "citation_support", "answer_relevancy", "correctness"}  # undefined for a refusal
SUBSTANTIVE_WORDS = 4  # a sentence shorter than this ("Sure.") is not expected to carry a citation


def parse_metrics(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return ALL_METRICS
    chosen = tuple(m.strip() for m in spec.split(",") if m.strip())
    unknown = [m for m in chosen if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}. Choose from {', '.join(ALL_METRICS)}")
    return chosen


def build_context(periods: list[dict]) -> str:
    """The excerpts as the answer model saw them (numbered, with title and time range)."""
    return "\n\n".join(f"[{n}] \"{p.get('video_title', '')}\" ({p.get('label', '')})\n{p['text']}" for n, p in enumerate(periods, 1))


def citation_coverage(answer: str) -> float | None:
    """Share of substantive sentences that carry at least one citation. A cheap, judge-free check on the citing habit."""
    sentences = [(s, n) for s, n in cited_sentences(answer) if len(s.split()) >= SUBSTANTIVE_WORDS]
    return sum(1 for _, n in sentences if n) / len(sentences) if sentences else None


def score_question(judge: Judge, entry: dict, output: dict, periods: list[dict], metrics, refused: bool) -> dict:
    """Run the requested judge metrics for one question. Returns {"scores", "details", "judge_errors"}.

    A metric that cannot apply (no reference answer, a refusal, no context) is simply absent from `scores`; one the
    judge failed on is None and listed in `judge_errors`. Nothing is guessed."""
    scores, details, errors = {}, {}, []
    answer, question, reference = output.get("answer", ""), entry["question"], entry.get("gold_answer")
    context = build_context(periods)
    excerpts = {c["n"]: c["text"] for c in output.get("citations") or [] if "n" in c}

    def run(name: str, call):
        try:
            result = call()
        except JudgeError as exc:
            scores[name], details[name] = None, {"error": str(exc)}
            errors.append(name)
            return
        scores[name] = result.get("score")
        details[name] = {k: v for k, v in result.items() if k != "score"}

    for metric in metrics:
        if metric in NEEDS_REFERENCE and not reference:
            continue
        if metric in ANSWER_METRICS and refused:
            continue
        if metric == "faithfulness":
            run(metric, lambda: judge.faithfulness(answer, context))
        elif metric == "citation_support":
            items = [(s, n) for s, n in cited_sentences(answer) if n]
            if items:
                run(metric, lambda: judge.citation_support(items, excerpts))
        elif metric == "answer_relevancy":
            run(metric, lambda: judge.answer_relevancy(question, answer))
        elif metric == "correctness":
            run(metric, lambda: judge.correctness(question, answer, reference))
        elif metric == "context_precision" and periods:
            run(metric, lambda: judge.context_precision(question, [p["text"] for p in periods]))
        elif metric == "context_recall":
            run(metric, lambda: judge.context_recall(reference, context))
    return {"scores": scores, "details": details, "judge_errors": errors}


def summarize(rows: list[dict], metrics) -> dict:
    """Mean score (with a bootstrap 95% CI) per metric, overall and by category, plus answer/refusal rates."""
    out = {"n_questions": len(rows), "answered_rate": mean([0.0 if r["refused"] else 1.0 for r in rows]),
           "refused_answerable": sum(1 for r in rows if r["refused"]),
           "judge_errors": sum(len(r["judge_errors"]) for r in rows),
           "citation_coverage": mean([r.get("citation_coverage") for r in rows]), "metrics": {}}
    for metric in metrics:
        values = [r["scores"].get(metric) for r in rows if metric in r["scores"]]
        scored = [v for v in values if v is not None]
        ci = bootstrap_ci(scored)
        by_category = {}
        for category in sorted({r["category"] for r in rows}):
            cat = [r["scores"].get(metric) for r in rows if r["category"] == category and r["scores"].get(metric) is not None]
            if cat:
                by_category[category] = {"n": len(cat), "mean": mean(cat)}
        out["metrics"][metric] = {
            "n": len(scored), "mean": mean(scored), "ci95": list(ci) if ci else None,
            "fraction_perfect": (sum(1 for v in scored if v >= 0.999) / len(scored)) if scored else None,
            "n_not_applicable": len(rows) - len(values), "n_judge_failed": len(values) - len(scored),
            "by_category": by_category,
        }
    return out


def write_audit(rows: list[dict], n: int, seed: int, path: Path) -> Path:
    """A Markdown sample of judged answers with the judge's reasons, so a person can check whether they agree."""
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    lines = ["# Judge audit sample", "",
             "For each item, read the answer and the judge's reasons and note whether you agree. If you often disagree, "
             "the judge (or its prompt) is not trustworthy enough to quote its scores.", ""]
    for row in sample:
        lines += [f"## {row['id']}  ({row['category']})", f"**Question:** {row['question']}", "",
                  f"**Answer:** {row['answer'] or '(refused)'}", "", "| metric | score | judge's detail |", "|---|---|---|"]
        for name, score in row["scores"].items():
            detail = {k: v for k, v in row["details"].get(name, {}).items() if k not in ("claims",)}
            lines.append(f"| {name} | {'n/a' if score is None else f'{score:.2f}'} | {str(detail)[:300]} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_summary(summary: dict, metrics) -> None:
    rows = []
    for name in metrics:
        m = summary["metrics"][name]
        ci = m["ci95"]
        rows.append([name, m["n"], pct(m["mean"], 1) if m["mean"] is not None else "n/a",
                     f"{pct(ci[0])}-{pct(ci[1])}" if ci else "n/a", pct(m["fraction_perfect"]) if m["fraction_perfect"] is not None else "n/a",
                     m["n_not_applicable"], m["n_judge_failed"]])
    print(format_table(["metric", "n", "mean", "95% CI", "perfect", "n/a", "judge failed"], rows))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.answer_quality_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--metrics", help=f"comma separated; default all of: {', '.join(ALL_METRICS)}")
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="full", help="query handling; 'full' is what the app does")
    parser.add_argument("--judge-provider", choices=("gemini", "groq"), help="override EVAL_JUDGE_PROVIDER for this run")
    parser.add_argument("--judge-model", help="override EVAL_JUDGE_MODEL for this run")
    parser.add_argument("--limit", type=int, help="only the first N answerable questions")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions (rate limits)")
    parser.add_argument("--audit", type=int, default=0, metavar="N", help="also write a Markdown sample of N judged answers to check by hand")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    try:
        metrics = parse_metrics(args.metrics)
        entries, info = load_eval_set(args.eval_set)
    except (ValueError, EvalSetError) as exc:
        return fail(str(exc))
    questions = [e for e in entries if not e["expects_refusal"]]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        return fail("no answerable questions in the eval set.")
    with_reference = sum(1 for e in questions if e["gold_answer"])
    skipped = [m for m in metrics if m in NEEDS_REFERENCE and not with_reference]
    if skipped:
        print(f"note: no question has a `gold_answer`, so {', '.join(skipped)} cannot be scored. Add reference answers to use them.")
    elif any(m in NEEDS_REFERENCE for m in metrics) and with_reference < len(questions):
        print(f"note: only {with_reference} of {len(questions)} questions have a `gold_answer`; correctness and context_recall use those.")

    quiet_backend_logging()
    from backend import config, util
    from backend.rag.answer import REFUSAL_PHRASE, generate_answer
    from backend.util import QuotaExhausted
    from evaluation.grounding_eval import classify
    from evaluation.metering import Meter, metered

    if args.judge_provider:
        config.JUDGE_PROVIDER = args.judge_provider
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model
    judge_cfg, answer_cfg = util._settings("judge"), util._settings("default")
    if judge_cfg == answer_cfg:
        print(f"WARNING: the judge ({judge_cfg[0]} / {judge_cfg[1]}) is the same model that writes the answers. Judges favour "
              "their own style, so these scores will be optimistic. Set EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL to another model.\n")

    judge, meter, rows, errors, aborted = Judge(), Meter(), [], [], None
    print(f"Scoring {len(questions)} questions on: {', '.join(metrics)} (judge: {judge_cfg[0]} / {judge_cfg[1]})...")
    with metered(meter):
        for number, entry in enumerate(questions, 1):
            before = meter.snapshot()
            try:
                found = run_search(entry, args.enhance)
                output = generate_answer(found["standalone"], found["periods"], found["prior"])
                refused = classify(output, REFUSAL_PHRASE)["refused"]
                scored = score_question(judge, entry, output, found["periods"], metrics, refused)
            except QuotaExhausted as exc:
                aborted = f"quota exhausted at question {number} ({entry['id']}): {exc}"
                print(f"\n{aborted}\nStopping; partial results are saved.")
                break
            except Exception as exc:  # noqa: BLE001
                errors.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"  [{number}/{len(questions)}] {entry['id']}: failed ({type(exc).__name__})")
                continue
            used = meter.since(before)
            rows.append({
                "id": entry["id"], "category": entry["category"], "question": entry["question"], "refused": refused,
                "answer": "" if refused else output.get("answer", ""), "n_periods": len(found["periods"]),
                "citation_coverage": None if refused else citation_coverage(output.get("answer", "")),
                "judge_calls": used["judge"].calls if "judge" in used else 0, **scored,
            })
            print(f"  [{number}/{len(questions)}] {entry['id']}  " + ("refused" if refused else "answered") + "  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in scored["scores"].items() if v is not None))
            if args.delay and number < len(questions):
                time.sleep(args.delay)

    if not rows:
        return fail("no question completed" + (f": {aborted}" if aborted else "; first error: " + (errors[0]["error"] if errors else "?")))

    summary = summarize(rows, metrics)
    print(f"\n{summary['n_questions']} questions; answered {pct(summary['answered_rate'])} "
          f"({summary['refused_answerable']} answerable question(s) were refused); {summary['judge_errors']} judge failure(s).\n")
    _print_summary(summary, metrics)
    if summary["citation_coverage"] is not None:
        print(f"\nCitation coverage (substantive sentences that cite something): {pct(summary['citation_coverage'], 1)}")
    if len(rows) < 30:
        print(f"Note: only {len(rows)} questions; a score can move several points on one question (see the intervals).")
    for err in errors:
        print(f"  failed: {err['id']}: {err['error']}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    path = write_results("answer_quality", {
        "eval_set": info, "params": {"metrics": list(metrics), "enhance": args.enhance, "judge": list(judge_cfg),
                                      "answer_model": list(answer_cfg), "judge_is_answer_model": judge_cfg == answer_cfg},
        "summary": summary, "llm_usage": meter.by_role(), "errors": errors, "aborted": aborted, "questions": rows,
    }, out_dir)
    print(f"\nWrote {path}")
    if args.audit:
        audit = write_audit(rows, args.audit, 0, path.with_name(path.stem.replace("answer_quality", "judge_audit") + ".md"))
        print(f"Wrote {audit}: read it and check whether you agree with the judge.")
    return QUOTA_EXIT_CODE if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
