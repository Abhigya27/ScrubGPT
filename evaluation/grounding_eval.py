"""Grounding / refusal evaluation (spec 1.4): does the app decline what it cannot answer, and only that?

    python -m evaluation.grounding_eval --eval-set evaluation/eval_set.json

Every question runs through the app's /ask pipeline in-process (enhance -> retrieve -> group -> generate_answer, the
same calls backend/main.py makes). A response counts as a REFUSAL when `grounded` is False or the answer is the
refusal sentence. Two numbers matter, because a system that refuses everything would trivially score 100% on the
first one:

  refusal_accuracy   = correct_refusals / total_unanswerable      (expects_refusal: true questions that were refused)
  false_refusal_rate = wrongly_refused  / total_answerable        (answerable questions that were refused anyway)

Also reported: how many refusals came from the retrieval threshold (nothing retrieved, the LLM was never called)
versus the answer model's own guard, and a STRICT refusal accuracy that only counts the explicit refusal sentence.
`grounded: False` with a different answer text means the model answered without any valid citation, which the app
does not treat as grounded but which is not a clean "I couldn't find that".

Each question costs up to two LLM calls (query enhancement and the answer). Use --delay to stay under a rate limit.
If the provider's quota runs out the run stops, saves what it has and exits with code 3.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

from evaluation.common import (
    DEFAULT_EVAL_SET, EvalSetError, fail, format_table, load_eval_set, pct, quiet_backend_logging, run_search,
    write_results,
)

QUOTA_EXIT_CODE = 3


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", (text or "").lower()).split())


def classify(output: dict, refusal_phrase: str) -> dict:
    """What the answer function returned, as refused / not and why. Pure.

    refusal_kind: 'refusal_sentence'    the answer is the fixed refusal sentence
                  'ungrounded_answer'   grounded is False but the model wrote something else (no valid citation)
                  'answered'            grounded and cited
    """
    sentence = _norm(refusal_phrase) in _norm(output.get("answer", "")) if refusal_phrase else False
    grounded = bool(output.get("grounded"))
    if sentence:
        kind = "refusal_sentence"
    elif not grounded:
        kind = "ungrounded_answer"
    else:
        kind = "answered"
    return {"refused": kind != "answered", "strict_refusal": kind == "refusal_sentence", "refusal_kind": kind}


def summarize(rows: list[dict]) -> dict:
    """refusal_accuracy and false_refusal_rate over rows with `expects_refusal`, `refused`, `strict_refusal`, `guard`."""
    unanswerable = [r for r in rows if r["expects_refusal"]]
    answerable = [r for r in rows if not r["expects_refusal"]]
    correct = [r for r in unanswerable if r["refused"]]
    wrong = [r for r in answerable if r["refused"]]
    return {
        "total_unanswerable": len(unanswerable),
        "correct_refusals": len(correct),
        "refusal_accuracy": len(correct) / len(unanswerable) if unanswerable else None,
        "strict_refusal_accuracy": (
            sum(1 for r in unanswerable if r["strict_refusal"]) / len(unanswerable) if unanswerable else None
        ),
        "total_answerable": len(answerable),
        "wrongly_refused": len(wrong),
        "false_refusal_rate": len(wrong) / len(answerable) if answerable else None,
        "correct_refusals_by_guard": {
            "retrieval_threshold": sum(1 for r in correct if r["guard"] == "retrieval_threshold"),
            "answer_model": sum(1 for r in correct if r["guard"] == "answer_model"),
            "uncited_answer": sum(1 for r in correct if r["guard"] == "uncited_answer"),
        },
        "unanswerable_answered": [r["id"] for r in unanswerable if not r["refused"]],
        "answerable_refused": [r["id"] for r in wrong],
    }


def guard_for(refused: bool, n_periods: int, kind: str) -> str | None:
    """Which layer produced a refusal: the retrieval threshold (nothing retrieved, so no LLM call), the answer
    model's own 'not covered' sentence, or an answer that cited nothing valid."""
    if not refused:
        return None
    if n_periods == 0:
        return "retrieval_threshold"
    return "answer_model" if kind == "refusal_sentence" else "uncited_answer"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.grounding_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="full",
                        help="query handling; 'full' (default) is what the app does")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--limit", type=int, help="only the first N questions (quick trial)")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)

    try:
        entries, info = load_eval_set(args.eval_set)
    except EvalSetError as exc:
        return fail(str(exc))
    if args.limit:
        entries = entries[: args.limit]

    quiet_backend_logging()
    from backend.rag.answer import REFUSAL_PHRASE, generate_answer
    from backend.util import QuotaExhausted

    rows, errors, aborted = [], [], None
    print(f"Asking {len(entries)} questions (query handling: {args.enhance})...")
    for number, entry in enumerate(entries, 1):
        try:
            found = run_search(entry, args.enhance)
            output = generate_answer(found["standalone"], found["periods"], found["prior"])
        except QuotaExhausted as exc:
            aborted = f"quota exhausted at question {number} ({entry['id']}): {exc}"
            print(f"\n{aborted}\nStopping; partial results are saved.")
            break
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{number}/{len(entries)}] {entry['id']}: failed ({type(exc).__name__})")
            continue
        verdict = classify(output, REFUSAL_PHRASE)
        n_periods = len(found["periods"])
        row = {
            "id": entry["id"], "category": entry["category"], "question": entry["question"],
            "expects_refusal": entry["expects_refusal"], "grounded": bool(output.get("grounded")),
            "n_periods": n_periods, **verdict,
            "guard": guard_for(verdict["refused"], n_periods, verdict["refusal_kind"]),
            "n_citations": len(output.get("citations") or []), "answer": (output.get("answer") or "")[:400],
        }
        rows.append(row)
        ok = verdict["refused"] == entry["expects_refusal"]
        print(f"  [{number}/{len(entries)}] {'ok ' if ok else 'BAD'} {entry['id']}  "
              f"{'expected refusal' if entry['expects_refusal'] else 'answerable':<16} -> {verdict['refusal_kind']}")
        if args.delay and number < len(entries):
            time.sleep(args.delay)

    if not rows:
        return fail("no question completed" + (f": {aborted}" if aborted else "; first error: " + (errors[0]["error"] if errors else "?")))

    summary = summarize(rows)
    print()
    print(format_table(
        ["metric", "value", "count"],
        [
            ["refusal accuracy (unanswerable)", pct(summary["refusal_accuracy"], 1), f"{summary['correct_refusals']}/{summary['total_unanswerable']}"],
            ["  strict (refusal sentence only)", pct(summary["strict_refusal_accuracy"], 1), ""],
            ["false-refusal rate (answerable)", pct(summary["false_refusal_rate"], 1), f"{summary['wrongly_refused']}/{summary['total_answerable']}"],
        ],
    ))
    guards = summary["correct_refusals_by_guard"]
    print(f"\nCorrect refusals came from: retrieval threshold {guards['retrieval_threshold']}, "
          f"answer model {guards['answer_model']}, uncited answer {guards['uncited_answer']}.")
    if summary["unanswerable_answered"]:
        print("Answered although they should have been refused: " + ", ".join(summary["unanswerable_answered"]))
    if summary["answerable_refused"]:
        print("Refused although answerable: " + ", ".join(summary["answerable_refused"]))
    if summary["total_unanswerable"] < 10:
        print(f"Note: only {summary['total_unanswerable']} unanswerable questions; aim for 10-15 so the rate means something.")
    for err in errors:
        print(f"  failed: {err['id']}: {err['error']}")

    path = write_results("grounding", {
        "eval_set": info, "params": {"enhance": args.enhance}, "summary": summary, "errors": errors,
        "aborted": aborted, "questions": rows,
    }, args.out_dir)
    print(f"\nWrote {path}")
    return QUOTA_EXIT_CODE if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
