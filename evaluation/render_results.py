"""Turn the saved result files into the README "Results" table and the numbers for the resume bullets (spec 1.9).

    python -m evaluation.render_results                       # print the Markdown table and the bullet numbers
    python -m evaluation.render_results --write-readme        # also rewrite the section between the markers in README.md
    python -m evaluation.render_results --count-tests         # include the pytest test count (runs pytest --collect-only)

Reads the NEWEST file of each kind in evaluation/results/, so every figure comes from a script run, never from memory.
Anything not measured yet shows as "not measured" together with the command that measures it.

The README section lives between these two lines, which this script keeps:
    <!-- RESULTS:START -->
    <!-- RESULTS:END -->
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from evaluation.common import REPO_ROOT, RESULTS_DIR, fail, pct

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NOT_MEASURED = "not measured"


def latest(prefix: str, results_dir: Path = RESULTS_DIR, require: str | None = None) -> dict | None:
    """The newest `<prefix>_*.json` (file names carry a UTC timestamp), optionally one that contains key `require`."""
    for path in sorted(Path(results_dir).glob(f"{prefix}_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if require is None or data.get(require):
            data["_file"] = path.name
            return data
    return None


def count_tests(root: Path = REPO_ROOT) -> dict | None:
    """{'runnable': offline tests, 'total': including integration} from `pytest --collect-only -q`, or None."""

    def collect(*extra: str) -> tuple[int, int | None] | None:
        try:
            out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *extra], cwd=root,
                                 capture_output=True, text=True, timeout=300).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"(\d+)(?:/(\d+))? tests? (?:collected|selected)", out)
        return (int(match.group(1)), int(match.group(2)) if match.group(2) else None) if match else None

    offline = collect("-m", "not integration")
    if offline is None:
        return None
    return {"runnable": offline[0], "total": offline[1] or offline[0]}


def _r(value, fmt: str) -> str:
    return NOT_MEASURED if value is None else fmt.format(value)


def numbers(res: dict) -> dict:
    """Every value the table and the resume bullets use, pulled out of the result files. Missing -> None."""
    n: dict = {}
    ingestion_corpus, ingestion_bench = res.get("corpus_file"), res.get("bench_file")
    corpus = (ingestion_corpus or {}).get("corpus") or {}
    n["videos"], n["hours"] = corpus.get("videos"), corpus.get("hours")
    qdrant = corpus.get("qdrant") or {}
    n["chunks"], n["chunks_total"] = qdrant.get("chunks_in_scope"), qdrant.get("chunks_in_collection")
    n["languages"] = corpus.get("languages")
    bench = (ingestion_bench or {}).get("benchmark") or {}
    n["cold_s"], n["reingest_s"], n["speedup"] = bench.get("cold_seconds"), bench.get("reingest_seconds"), bench.get("reingest_speedup")

    ablation = res.get("ablation")
    if ablation:
        n["eval_questions"] = ablation["n_questions"]
        k = ablation["params"]["k"]
        n["k"] = k
        ranking = ablation["ranking"]
        n["ablation"] = {name: {"recall": s["recall_at_k"], "mrr": s["mrr"], "ci": s["recall_ci95"], "precision": s.get("precision_at_k"),
                                "ndcg": s.get("ndcg_at_k"), "chunk_recall": s.get("chunk_recall_at_k")} for name, s in ranking.items()}
        default = next((s for name, s in ranking.items() if name.endswith("(default)")), None) or {}
        n["precision"], n["ndcg"], n["chunk_recall"] = default.get("precision_at_k"), default.get("ndcg_at_k"), default.get("chunk_recall_at_k")
        n["dense_recall"] = next((s["recall_at_k"] for name, s in ranking.items() if name.startswith("dense-only")), None)
        n["hybrid_recall"] = next((s["recall_at_k"] for name, s in ranking.items() if name.endswith("(default)")), None)
    ts = (res.get("timestamp") or {}).get("summary")
    if ts:
        n["ts_n"] = ts["n"]
        n["ts"] = {t: v for t, v in ts["within"].items()}
        n["ts_start"] = {t: v for t, v in ts["start_within"].items()}
    gr = (res.get("grounding") or {}).get("summary")
    if gr:
        n["refusal_accuracy"], n["false_refusal_rate"] = gr["refusal_accuracy"], gr["false_refusal_rate"]
        n["unanswerable"], n["answerable"] = gr["total_unanswerable"], gr["total_answerable"]
    tr = (res.get("translation") or {}).get("summary")
    if tr:
        n["fallback_rate"], n["translated_chunks"], n["tr_languages"] = tr["fallback_rate"], tr["chunks_total"], tr["languages"]
        n["tr_batches"] = tr["batches_attempted"]
    aq = (res.get("answer_quality") or {}).get("summary")
    if aq:
        n["aq"] = {name: m["mean"] for name, m in aq["metrics"].items() if m["mean"] is not None}
        n["aq_n"], n["aq_answered_rate"] = aq["n_questions"], aq["answered_rate"]
        n["aq_judge_is_answerer"] = (res.get("answer_quality") or {}).get("params", {}).get("judge_is_answer_model")
    rb = res.get("robustness")
    if rb:
        drops = [(d["recall_change"], kind, name) for kind, per in rb["changes"].items() for name, d in per.items() if d["recall_change"] is not None]
        n["robust_worst"] = min(drops) if drops else None
    pf = (res.get("performance") or {}).get("summary")
    if pf:
        n["lat_p50"], n["lat_p95"] = pf["latency_seconds"]["total"]["p50"], pf["latency_seconds"]["total"]["p95"]
        n["llm_calls_per_q"], n["cost_per_100"] = pf["llm_calls_per_question"], pf["cost_per_100_questions_usd"]
    n["tests"] = res.get("tests")
    return n


def _provenance(res: dict) -> str:
    parts = []
    for key in ("ablation", "timestamp", "grounding", "answer_quality", "robustness", "performance", "ingestion", "translation"):
        data = res.get(key)
        if data:
            parts.append(f"{key}: {data['generated_at'][:10]}, `{(data.get('git') or {}).get('commit', 'unknown')[:7]}`"
                         + (" (uncommitted changes)" if (data.get("git") or {}).get("dirty") else ""))
    return "; ".join(parts) or "no results yet"


def render_markdown(res: dict) -> str:
    n = numbers(res)
    k = n.get("k", 5)
    lines = ["## Results", ""]
    size = n.get("eval_questions")
    lines += [
        f"Measured locally against the real Qdrant collection with the scripts in `evaluation/` "
        f"(eval set: {_r(size, '{} answerable questions')}"
        + (f", {n['unanswerable']} unanswerable" if n.get("unanswerable") is not None else "") + "). "
        f"Source runs: {_provenance(res)}.", "",
        "### Corpus", "", "| Videos | Hours | Chunks indexed | Languages (videos) |", "|---:|---:|---:|---|",
        f"| {_r(n['videos'], '{}')} | {_r(n['hours'], '{:.1f}')} | {_r(n['chunks'], '{}')} | "
        + (", ".join(f"{a} {b}" for a, b in (n['languages'] or {}).items()) or NOT_MEASURED) + " |", "",
        f"### Hybrid retrieval ablation (Recall@{k}, MRR)", "",
    ]
    if n.get("ablation"):
        extended = any(s["precision"] is not None or s["ndcg"] is not None for s in n["ablation"].values())
        lines += [f"| Fusion (sparse/dense) | Recall@{k} (hit rate) | 95% CI | MRR |" + (f" P@{k} | nDCG@{k} | ChunkRecall@{k} |" if extended else ""),
                  "|---|---:|---:|---:|" + ("---:|---:|---:|" if extended else "")]
        for name, s in n["ablation"].items():
            ci = f"{pct(s['ci'][0])}-{pct(s['ci'][1])}" if s["ci"] else "n/a"
            row = f"| {name} | {pct(s['recall'], 1)} | {ci} | {s['mrr']:.3f} |"
            if extended:
                row += f" {pct(s['precision'], 1)} | {'n/a' if s['ndcg'] is None else format(s['ndcg'], '.3f')} | {pct(s['chunk_recall'], 1)} |"
            lines.append(row)
    else:
        lines.append(f"{NOT_MEASURED} (`python -m evaluation.ablation_eval`)")
    lines += ["", "### Answer quality and ingestion", "", "| Metric | Value |", "|---|---:|"]
    if n.get("ts"):
        for tol, value in n["ts"].items():
            lines.append(f"| Timestamp accuracy, gold moment within ±{tol}s of a cited period | {pct(value, 1)} |")
        for tol, value in n["ts_start"].items():
            lines.append(f"| Timestamp accuracy, cited period *starts* within ±{tol}s | {pct(value, 1)} |")
    else:
        lines.append(f"| Timestamp accuracy (±15s / ±30s) | {NOT_MEASURED} (`evaluation.timestamp_accuracy_eval`) |")
    lines += [
        f"| Refusal accuracy (unanswerable questions) | {_r(n.get('refusal_accuracy'), '{:.1%}')} |",
        f"| False-refusal rate (answerable questions) | {_r(n.get('false_refusal_rate'), '{:.1%}')} |",
        f"| Cold ingest | {_r(n['cold_s'], '{:.0f}s')} |",
        f"| Re-ingest | {_r(n['reingest_s'], '{:.0f}s')}" + (f" ({n['speedup']}x faster)" if n.get("speedup") else "") + " |",
        f"| Translation fallback rate | {_r(n.get('fallback_rate'), '{:.1%}')}"
        + (f" ({n['tr_batches']} batches, {n['translated_chunks']} chunks)" if n.get("tr_batches") is not None else "") + " |",
    ]
    if n.get("lat_p50") is not None:
        lines.append(f"| Latency to first answer text, p50 / p95 | {n['lat_p50']:.2f}s / {n['lat_p95']:.2f}s |")
        lines.append(f"| LLM calls per question | {n['llm_calls_per_q']:.2f} |")
    else:
        lines.append(f"| Latency and LLM calls per question | {NOT_MEASURED} (`evaluation.performance_eval`) |")
    if n.get("robust_worst"):
        change, kind, name = n["robust_worst"]
        lines.append(f"| Robustness: worst Recall@{k} drop (under '{kind}', {name}) | {change * 100:+.1f} points |")
    tests = n.get("tests")
    lines.append(f"| Automated tests (offline) | {tests['runnable'] if tests else NOT_MEASURED} |")
    lines += ["", "### Answer quality (LLM judge)", ""]
    if n.get("aq"):
        warn = " The judge is the same model that writes the answers, so these are optimistic." if n.get("aq_judge_is_answerer") else ""
        lines += [f"{n['aq_n']} answerable questions, answered {pct(n['aq_answered_rate'])} of the time.{warn}", "",
                  "| Metric | Mean score |", "|---|---:|"]
        lines += [f"| {name.replace('_', ' ')} | {pct(value, 1)} |" for name, value in n["aq"].items()]
    else:
        lines.append(f"{NOT_MEASURED} (`python -m evaluation.answer_quality_eval`)")
    return "\n".join(lines) + "\n"


def resume_numbers(res: dict) -> str:
    """The bracketed values of the resume bullets, with what each one means."""
    n = numbers(res)
    k = n.get("k", 5)

    def show(label: str, value) -> str:
        return f"  {label:<46}{NOT_MEASURED if value is None else value}"

    ts = n.get("ts") or {}
    return "\n".join([
        "Resume bullet numbers (from the newest result files):",
        show("[N] videos", n["videos"]), show("[H] hours", n["hours"]), show("[C] chunks indexed", n["chunks"]),
        show(f"[X] dense-only Recall@{k}", pct(n.get("dense_recall"), 1) if n.get("dense_recall") is not None else None),
        show(f"[Y] hybrid (default) Recall@{k}", pct(n.get("hybrid_recall"), 1) if n.get("hybrid_recall") is not None else None),
        show("[N] questions in the eval set", n.get("eval_questions")),
        show("[T] / [Z] timestamp tolerance / within", ", ".join(f"±{t}s: {pct(v, 1)}" for t, v in ts.items()) or None),
        show("[X] refusal accuracy", pct(n.get("refusal_accuracy"), 1) if n.get("refusal_accuracy") is not None else None),
        show("false-refusal rate (quote it with the above)", pct(n.get("false_refusal_rate"), 1) if n.get("false_refusal_rate") is not None else None),
        show("[A] cold ingest / [B] re-ingest (seconds)",
             f"{n['cold_s']:.0f}s / {n['reingest_s']:.0f}s" if n.get("cold_s") and n.get("reingest_s") is not None else None),
        show("P@k / nDCG@k / ChunkRecall@k (default weights)",
             " / ".join("n/a" if v is None else f"{v:.3f}" for v in (n.get("precision"), n.get("ndcg"), n.get("chunk_recall"))) if n.get("ndcg") is not None else None),
        show("faithfulness / correctness / relevancy", " / ".join(
            "n/a" if (n.get("aq") or {}).get(m) is None else pct(n["aq"][m], 1) for m in ("faithfulness", "correctness", "answer_relevancy")) if n.get("aq") else None),
        show("latency p50 / p95", f"{n['lat_p50']:.2f}s / {n['lat_p95']:.2f}s" if n.get("lat_p50") is not None else None),
        show("tests", (n.get("tests") or {}).get("runnable")),
    ])


def replace_section(readme: str, section: str) -> str:
    """Put `section` between the markers, keeping the file's own line endings (the README is CRLF on Windows)."""
    newline = "\r\n" if "\r\n" in readme else "\n"
    body = section.rstrip("\n").replace("\r\n", "\n").replace("\n", newline)
    block = f"{START}{newline}{body}{newline}{END}"
    if START in readme and END in readme:
        head, rest = readme.split(START, 1)
        _, tail = rest.split(END, 1)
        return head + block + tail
    return readme.rstrip() + newline + newline + block + newline


def load_results(results_dir: Path = RESULTS_DIR) -> dict:
    return {
        "ablation": latest("ablation", results_dir), "timestamp": latest("timestamp", results_dir),
        "grounding": latest("grounding", results_dir), "translation": latest("translation", results_dir),
        "ingestion": latest("ingestion", results_dir), "answer_quality": latest("answer_quality", results_dir),
        "robustness": latest("robustness", results_dir), "performance": latest("performance", results_dir),
        "corpus_file": latest("ingestion", results_dir, require="corpus"),
        "bench_file": latest("ingestion", results_dir, require="benchmark"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.render_results", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--write-readme", nargs="?", const=str(REPO_ROOT / "README.md"), metavar="README",
                        help="rewrite the section between the markers (default README.md)")
    parser.add_argument("--count-tests", action="store_true", help="run pytest --collect-only to count the tests")
    args = parser.parse_args(argv)

    res = load_results(Path(args.results_dir))
    if args.count_tests:
        res["tests"] = count_tests()
        if res["tests"] is None:
            print("note: could not count tests (is pytest installed?)", file=sys.stderr)
    section = render_markdown(res)
    print(section)
    print(resume_numbers(res))
    if args.write_readme:
        path = Path(args.write_readme)
        try:
            text = path.read_bytes().decode("utf-8")
        except OSError as exc:
            return fail(f"cannot read {path}: {exc}")
        path.write_bytes(replace_section(text, section).encode("utf-8"))
        print(f"\nUpdated the Results section of {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
