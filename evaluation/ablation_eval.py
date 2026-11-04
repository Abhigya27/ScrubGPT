"""Hybrid retrieval ablation (spec 1.2): does the default SPARSE_WEIGHT=0.65 / DENSE_WEIGHT=0.35 hold up?

    python -m evaluation.ablation_eval --eval-set evaluation/eval_set.json

For every answerable question (expects_refusal false, gold_answer_contains filled in) the candidates are fetched from
Qdrant ONCE (scoped to the question's source_id) and then re-fused offline under each weight setting. That is exactly
what the app does at query time, because fusion is a pure function of the two candidate lists.

A returned chunk is RELEVANT if its text contains gold_answer_contains (case- and punctuation-insensitive).
  Recall@k      = questions with a relevant chunk in the top k / questions. Strictly a HIT RATE (any relevant chunk).
  MRR           = mean of 1/rank of the first relevant chunk (0 when none in the top --depth)
  P@k           = relevant chunks in the top k / k, averaged over questions
  nDCG@k        = like P@k but rewards relevant chunks that rank higher (binary relevance)
  ChunkRecall@k = share of ALL relevant chunks in the source that made the top k (needs the corpus scan, below)

Before scoring, the whole source is read from Qdrant (read-only; skip with --no-corpus-scan). That gives the true number
of relevant chunks per question, and catches a gold_answer_contains that matches NO chunk: that is a mistake in the eval
set (a typo, or Hindi wording where the stored text is the English translation), not a retrieval failure, so those
questions are listed and left out of the metrics (--keep-unmatched to score them anyway).

Two tables are printed:
  1. RANKING   every candidate ranked by fused score, no threshold, no precision guards. This isolates fusion quality
               and is the one to quote.
  2. PRODUCTION  the app's pipeline: absolute RETRIEVAL_THRESHOLD plus the relative-score and dense-similarity guards.
               That threshold was tuned for the default weights, so other weights are at a structural disadvantage
               here; read this table as "what the user would get", not as a fair weight comparison.

Query handling: --enhance history (default) resolves follow-ups from their `history` and does nothing else, so the
comparison is about fusion alone; --enhance full adds the app's query expansion (one LLM call per question);
--enhance none searches the question as typed. Needs the same .env as the app (Qdrant) and the embedding models.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

from evaluation.common import (
    DEFAULT_EVAL_SET, EvalSetError, chunk_recall_at_k, fail, first_relevant_rank, format_table, is_relevant,
    load_eval_set, mean, mean_reciprocal_rank, ndcg_at_k, pct, precision_at_k, prepare_query, quiet_backend_logging,
    recall_at_k, wilson_interval, write_results,
)

DEFAULT_SPEC = "0:1,1:0,0.5:0.5,0.65:0.35,0.8:0.2"  # sparse:dense


@dataclass(frozen=True)
class FusionConfig:
    name: str
    sparse: float
    dense: float

    @property
    def ratio(self) -> float:
        return self.sparse / (self.sparse + self.dense)


def _label(sparse: float, dense: float) -> str:
    if sparse == 0:
        return "dense-only"
    if dense == 0:
        return "sparse-only"
    share = sparse / (sparse + dense) * 100
    return f"hybrid {share:.0f}/{100 - share:.0f}"


def parse_configs(spec: str, default_sparse: float, default_dense: float) -> list[FusionConfig]:
    """'0:1,1:0,0.65:0.35' (sparse:dense pairs) -> configs. The one equal to the configured weights is labelled
    '(default)'; if none is, the configured weights are appended so the default is always in the comparison."""
    configs: list[FusionConfig] = []
    for part in spec.split(","):
        try:
            sparse, dense = (float(x) for x in part.strip().split(":"))
        except ValueError:
            raise ValueError(f"bad weight pair {part!r}: use sparse:dense, e.g. 0.65:0.35") from None
        if sparse < 0 or dense < 0 or sparse + dense <= 0:
            raise ValueError(f"bad weight pair {part!r}: weights must be non-negative and not both zero")
        configs.append(FusionConfig(_label(sparse, dense), sparse, dense))
    default_ratio = default_sparse / (default_sparse + default_dense)
    marked, found = [], False
    for cfg in configs:
        if math.isclose(cfg.ratio, default_ratio, abs_tol=1e-9) and not found:
            found = True
            cfg = FusionConfig(cfg.name + " (default)", cfg.sparse, cfg.dense)
        marked.append(cfg)
    if not found:
        marked.append(FusionConfig(_label(default_sparse, default_dense) + " (default)", default_sparse, default_dense))
    names = [c.name for c in marked]
    if len(set(names)) != len(names):
        raise ValueError("two weight pairs have the same sparse:dense ratio; remove the duplicate")
    return marked


def select_questions(entries: list[dict]) -> tuple[list[dict], dict]:
    """Answerable questions that have a gold string. Returns (selected, how many of each kind were left out)."""
    selected, refusals, no_gold = [], 0, 0
    for entry in entries:
        if entry["expects_refusal"]:
            refusals += 1
        elif not entry["gold_answer_contains"]:
            no_gold += 1
        else:
            selected.append(entry)
    return selected, {"expects_refusal": refusals, "missing_gold_answer_contains": no_gold}


def corpus_matches(chunks: list[dict], gold) -> dict:
    """How many chunks of a source contain the gold text, and in which language(s) they are."""
    from collections import Counter

    hits = [c for c in chunks if is_relevant(c.get("text", ""), gold)]
    languages = Counter((c.get("source_language") or "unknown").lower().split("-")[0] for c in hits)
    return {"total_relevant": len(hits), "language": languages.most_common(1)[0][0] if languages else None}


def split_unmatched(entries: list[dict], corpus_for) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Separate questions whose gold text matches at least one chunk from those that match none.

    corpus_for(source_id) -> the source's chunks. Returns (usable, unmatched, info) where info[id] is
    corpus_matches() for every entry. A source whose corpus cannot be read leaves its questions usable (and unknown)."""
    usable, unmatched, info = [], [], {}
    cache: dict[str, list[dict] | None] = {}
    for entry in entries:
        sid = entry["source_id"]
        if sid not in cache:
            try:
                cache[sid] = corpus_for(sid)
            except Exception:  # noqa: BLE001 - statistics are optional; scoring still works without them
                cache[sid] = None
        if cache[sid] is None:
            usable.append(entry)
            continue
        info[entry["id"]] = corpus_matches(cache[sid], entry["gold_answer_contains"])
        (usable if info[entry["id"]]["total_relevant"] else unmatched).append(entry)
    return usable, unmatched, info


def run_ablation(entries, fetch, rank_fn, fuse_fn, configs, depth: int = 10, k: int = 5, corpus_info=None) -> tuple[list[dict], list[dict]]:
    """Score every question under every config. Pure given the injected functions.

    fetch(entry)                      -> (dense_hits, sparse_hits, n_points)         the only I/O
    rank_fn(dense, sparse, n, entry, cfg)  -> chunk dicts, best first, no threshold
    fuse_fn(dense, sparse, n, entry, cfg)  -> chunk dicts after threshold + guards, best first
    corpus_info                        {entry id: {"total_relevant", "language"}} from split_unmatched, or None

    Returns (rows, errors). A question whose fetch fails is reported in `errors` and excluded from every metric.
    """
    rows, errors = [], []
    corpus_info = corpus_info or {}
    for entry in entries:
        try:
            dense, sparse, n_points = fetch(entry)
        except Exception as exc:  # noqa: BLE001 - one bad question must not sink the run
            errors.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        gold = entry["gold_answer_contains"]
        known = corpus_info.get(entry["id"]) or {}
        total = known.get("total_relevant")
        ranking, production, metrics = {}, {}, {}
        for cfg in configs:
            texts = [c["text"] for c in rank_fn(dense, sparse, n_points, entry, cfg)]
            flags = [is_relevant(t, gold) for t in texts[:depth]]
            ranking[cfg.name] = next((i for i, f in enumerate(flags, 1) if f), None)
            metrics[cfg.name] = {
                "precision": precision_at_k(flags, k), "ndcg": ndcg_at_k(flags, k, total),
                "chunk_recall": chunk_recall_at_k(flags, k, total),
            }
            production[cfg.name] = first_relevant_rank([c["text"] for c in fuse_fn(dense, sparse, n_points, entry, cfg)], gold, depth)
        rows.append({
            "id": entry["id"], "category": entry["category"], "question": entry["question"],
            "source_id": entry["source_id"], "total_relevant_chunks": total, "language": known.get("language"),
            "ranking_rank": ranking, "production_rank": production, "ranking_metrics": metrics,
        })
    return rows, errors


def summarize(rows: list[dict], configs, key: str, k: int, depth: int) -> dict[str, dict]:
    """Per config: n, Recall@k (+ Wilson 95% CI), MRR, P@k, nDCG@k, ChunkRecall@k, and Recall@k per category/language.

    P@k, nDCG@k and ChunkRecall@k exist only for the ranking table (key="ranking_rank"); they are None otherwise."""
    out: dict[str, dict] = {}
    for cfg in configs:
        ranks = [r[key][cfg.name] for r in rows]
        hits = sum(1 for r in ranks if r is not None and r <= k)
        ci = wilson_interval(hits, len(ranks))

        def group(field: str) -> dict:
            result = {}
            for value in sorted({r.get(field) for r in rows if r.get(field)}):
                group_ranks = [r[key][cfg.name] for r in rows if r.get(field) == value]
                result[value] = {"n": len(group_ranks), "recall_at_k": recall_at_k(group_ranks, k)}
            return result

        def metric(name: str):
            if key != "ranking_rank":
                return None
            return mean([(r.get("ranking_metrics") or {}).get(cfg.name, {}).get(name) for r in rows])

        out[cfg.name] = {
            "n": len(ranks), "recall_at_k": recall_at_k(ranks, k), "recall_ci95": list(ci) if ci else None,
            "mrr": mean_reciprocal_rank(ranks, depth), "precision_at_k": metric("precision"),
            "ndcg_at_k": metric("ndcg"), "chunk_recall_at_k": metric("chunk_recall"),
            "by_category": group("category"), "by_language": group("language"),
        }
    return out


def compare_with_default(summary: dict[str, dict], configs) -> list[str]:
    """Plain-language notes: which non-default configs match or beat the default, and whether that is meaningful."""
    default = next((c for c in configs if c.name.endswith("(default)")), None)
    if default is None or default.name not in summary:
        return []
    base = summary[default.name]
    notes = []
    for cfg in configs:
        if cfg.name == default.name:
            continue
        s = summary[cfg.name]
        better_recall = (s["recall_at_k"] or 0) > (base["recall_at_k"] or 0) + 1e-12
        better_mrr = (s["mrr"] or 0) > (base["mrr"] or 0) + 1e-12
        tied_recall = abs((s["recall_at_k"] or 0) - (base["recall_at_k"] or 0)) <= 1e-12
        if better_recall or (tied_recall and better_mrr):
            notes.append(
                f"{cfg.name} beats the default: Recall@k {pct(s['recall_at_k'])} vs {pct(base['recall_at_k'])}, "
                f"MRR {s['mrr']:.3f} vs {base['mrr']:.3f}."
            )
    return notes


def _table(summary: dict[str, dict], configs, k: int) -> str:
    rows = []
    for cfg in configs:
        m = summary[cfg.name]
        ci = m["recall_ci95"]
        rows.append([
            cfg.name, f"{cfg.sparse:g}/{cfg.dense:g}", pct(m["recall_at_k"], 1),
            f"{pct(ci[0])}-{pct(ci[1])}" if ci else "n/a", f"{m['mrr']:.3f}" if m["mrr"] is not None else "n/a",
            pct(m["precision_at_k"], 1), f"{m['ndcg_at_k']:.3f}" if m["ndcg_at_k"] is not None else "n/a",
            pct(m["chunk_recall_at_k"], 1),
        ])
    return format_table(
        ["config", "sparse/dense", f"Recall@{k}(hit)", "95% CI", "MRR", f"P@{k}", f"nDCG@{k}", f"ChunkRecall@{k}"], rows)


def _category_table(summary: dict[str, dict], configs, k: int, field: str = "by_category") -> str:
    """Recall@k per category (or per language with field="by_language"), one row per config."""
    groups = sorted({c for s in summary.values() for c in s[field]})
    rows = []
    for cfg in configs:
        row = [cfg.name]
        for group in groups:
            cell = summary[cfg.name][field].get(group)
            row.append(pct(cell["recall_at_k"]) if cell else "-")
        rows.append(row)
    counts = ["(questions)"] + [
        str(next((s[field][g]["n"] for s in summary.values() if g in s[field]), 0)) for g in groups
    ]
    return format_table(["config", *groups], [*rows, counts])


def backend_io(enhance: str, no_production: bool = False):
    """The three functions run_ablation needs, wired to the real backend (imports it, so needs the app's .env)."""
    from backend import config
    from backend.index import qdrant_store

    def fetch(entry):
        standalone, expansions, _prior, _note = prepare_query(entry, enhance)
        return qdrant_store.fetch_candidates(standalone, entry["source_id"], config.MAX_RESULTS, expansions)

    def rank_fn(dense, sparse, n_points, entry, cfg):
        return qdrant_store.rank_candidates(dense, sparse, n_points, entry["source_id"], cfg.sparse, cfg.dense)

    def fuse_fn(dense, sparse, n_points, entry, cfg):
        if no_production:
            return []
        return qdrant_store._fuse_and_threshold(
            dense, sparse, config.RETRIEVAL_THRESHOLD, config.MAX_RESULTS, entry["source_id"], n_points, cfg.sparse, cfg.dense
        )

    return fetch, rank_fn, fuse_fn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.ablation_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--configs", default=DEFAULT_SPEC, help=f"sparse:dense pairs, comma separated (default {DEFAULT_SPEC})")
    parser.add_argument("--k", type=int, default=5, help="Recall@k (default 5)")
    parser.add_argument("--depth", type=int, default=10, help="how far down the ranking MRR looks (default 10)")
    parser.add_argument("--enhance", choices=("none", "history", "full"), default="history")
    parser.add_argument("--no-production", action="store_true", help="skip the threshold + guards table")
    parser.add_argument("--no-corpus-scan", action="store_true",
                        help="do not read the source's chunks (disables ChunkRecall, language split and the gold-text check)")
    parser.add_argument("--keep-unmatched", action="store_true",
                        help="still score questions whose gold_answer_contains matches no chunk (they count as misses)")
    parser.add_argument("--limit", type=int, help="only the first N answerable questions (for a quick trial)")
    parser.add_argument("--out-dir", help="where to write the results JSON (default evaluation/results)")
    args = parser.parse_args(argv)

    try:
        entries, info = load_eval_set(args.eval_set)
    except EvalSetError as exc:
        return fail(str(exc))
    questions, left_out = select_questions(entries)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        return fail("no answerable questions with gold_answer_contains in the eval set.")

    quiet_backend_logging()
    from backend import config
    from backend.index import qdrant_store

    try:
        configs = parse_configs(args.configs, config.SPARSE_WEIGHT, config.DENSE_WEIGHT)
    except ValueError as exc:
        return fail(str(exc))

    fetch, rank_fn, fuse_fn = backend_io(args.enhance, args.no_production)

    corpus_info, unmatched = {}, []
    if not args.no_corpus_scan:
        print("Reading the indexed chunks to count the relevant ones per question...")
        usable, unmatched, corpus_info = split_unmatched(questions, qdrant_store.scan_source_chunks)
        if unmatched:
            print(f"\nWARNING: {len(unmatched)} question(s) have a gold_answer_contains that matches NO chunk of their source. "
                  "That is a mistake in the eval set (typo, or wording that differs from the stored English text), not a "
                  "retrieval failure." + ("" if args.keep_unmatched else " They are left out of the metrics:"))
            for entry in unmatched:
                print(f"  {entry['id']}: {entry['gold_answer_contains']!r}")
            print()
        if not args.keep_unmatched:
            questions = usable
        if not questions:
            return fail("no question's gold_answer_contains matches any indexed chunk; fix the eval set.")

    print(f"Running {len(questions)} questions x {len(configs)} fusion settings "
          f"(query handling: {args.enhance}; candidates fetched once per question)...")
    rows, errors = run_ablation(questions, fetch, rank_fn, fuse_fn, configs, args.depth, args.k, corpus_info)
    if not rows:
        return fail("every question failed to fetch candidates; first error: " + (errors[0]["error"] if errors else "?"))

    ranking = summarize(rows, configs, "ranking_rank", args.k, args.depth)
    production = None if args.no_production else summarize(rows, configs, "production_rank", args.k, args.depth)

    print(f"\n{len(rows)} questions scored ({len(errors)} failed, {left_out['expects_refusal']} refusal and "
          f"{left_out['missing_gold_answer_contains']} gold-less questions left out)\n")
    print(f"RANKING (fused score only, no threshold)  Recall@{args.k} is a hit rate; MRR looks {args.depth} deep")
    print(_table(ranking, configs, args.k))
    print(f"\nRecall@{args.k} (hit rate) by category")
    print(_category_table(ranking, configs, args.k))
    if any(m["by_language"] for m in ranking.values()):
        print(f"\nRecall@{args.k} (hit rate) by language of the relevant text")
        print(_category_table(ranking, configs, args.k, "by_language"))
    notes = compare_with_default(ranking, configs)
    if production is not None:
        print(f"\nPRODUCTION (RETRIEVAL_THRESHOLD={config.RETRIEVAL_THRESHOLD} + precision guards; threshold tuned for the default weights)")
        print(_table(production, configs, args.k))
    print()
    if notes:
        for note in notes:
            print(f"** {note}")
        print("   Consider whether SPARSE_WEIGHT/DENSE_WEIGHT should change, but check the 95% CIs first.")
    else:
        print("The default weights are at least as good as every other setting tried (on this eval set).")
    if len(rows) < 30:
        print(f"Note: only {len(rows)} questions. Differences of a question or two are noise (see the CIs).")
    for err in errors:
        print(f"  failed: {err['id']}: {err['error']}")

    path = write_results("ablation", {
        "eval_set": info, "params": {
            "k": args.k, "depth": args.depth, "enhance": args.enhance,
            "retrieval_threshold": config.RETRIEVAL_THRESHOLD, "max_results": config.MAX_RESULTS,
            "configured_sparse_weight": config.SPARSE_WEIGHT, "configured_dense_weight": config.DENSE_WEIGHT,
        },
        "configs": [{"name": c.name, "sparse": c.sparse, "dense": c.dense} for c in configs],
        "n_questions": len(rows), "left_out": left_out, "errors": errors,
        "unmatched_gold": [{"id": e["id"], "gold_answer_contains": e["gold_answer_contains"]} for e in unmatched],
        "corpus_scan": not args.no_corpus_scan,
        "ranking": ranking, "production": production, "beats_default": notes, "questions": rows,
    }, args.out_dir)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
