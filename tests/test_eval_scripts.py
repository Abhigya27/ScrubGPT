"""The evaluation scripts' logic, offline: ablation, timestamp accuracy, grounding, ingestion and translation stats,
the eval-set template and the results renderer. Backend calls are replaced by fakes; fusion uses the real code."""
import json
import math
from types import SimpleNamespace

import pytest

from backend import config
from backend.index import qdrant_store
from evaluation import (
    ablation_eval, build_eval_template, grounding_eval, ingestion_stats, render_results, timestamp_accuracy_eval,
    translation_stats,
)
from evaluation.common import EvalSetError, load_eval_set

N_POINTS = 50
HALF = config.SPARSE_HALF_SATURATION_TERMS * qdrant_store._TF_TYPICAL * math.log(N_POINTS)


def entry(**kw):
    base = {"id": "e1", "source_id": "PL", "video_id": "vid", "question": "q?", "category": "direct_fact",
            "gold_answer_contains": "hash table", "gold_timestamp_seconds": 100, "expects_refusal": False, "history": []}
    base.update(kw)
    return base


# =============================== ablation ===============================================================================
def hit(point_id, score, text, start=0):
    return SimpleNamespace(id=point_id, score=score, payload={
        "chunk_id": f"vid:{start}", "video_id": "vid", "video_title": "T", "start_sec": start, "end_sec": start + 90,
        "text": text, "positions": {"PL": 0}})


def real_rank(dense, sparse, n, e, cfg):
    return qdrant_store.rank_candidates(dense, sparse, n, e["source_id"], cfg.sparse, cfg.dense)


def real_fuse(dense, sparse, n, e, cfg):
    return qdrant_store._fuse_and_threshold(
        dense, sparse, config.RETRIEVAL_THRESHOLD, config.MAX_RESULTS, e["source_id"], n, cfg.sparse, cfg.dense)


def test_parse_configs_marks_the_default_and_keeps_the_spec_order():
    configs = ablation_eval.parse_configs(ablation_eval.DEFAULT_SPEC, 0.65, 0.35)
    assert [c.name for c in configs] == [
        "dense-only", "sparse-only", "hybrid 50/50", "hybrid 65/35 (default)", "hybrid 80/20"]
    assert [(c.sparse, c.dense) for c in configs][:2] == [(0.0, 1.0), (1.0, 0.0)]


def test_the_configured_default_is_added_when_the_spec_omits_it():
    configs = ablation_eval.parse_configs("0:1,1:0", 0.7, 0.3)
    assert configs[-1].name.endswith("(default)") and (configs[-1].sparse, configs[-1].dense) == (0.7, 0.3)


def test_default_is_recognised_by_ratio():
    configs = ablation_eval.parse_configs("1.3:0.7,0:1", 0.65, 0.35)  # same ratio as 0.65/0.35
    assert configs[0].name.endswith("(default)") and len(configs) == 2


@pytest.mark.parametrize("spec", ["nonsense", "0.5", "-1:2", "0:0", "0.5:0.5,1:1"])
def test_bad_config_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        ablation_eval.parse_configs(spec, 0.65, 0.35)


def test_select_questions_drops_refusals_and_gold_less_entries():
    chosen, left = ablation_eval.select_questions([
        entry(id="a"), entry(id="b", expects_refusal=True), entry(id="c", gold_answer_contains=None)])
    assert [e["id"] for e in chosen] == ["a"]
    assert left == {"expects_refusal": 1, "missing_gold_answer_contains": 1}


def candidates():
    """Dense search favours the relevant chunk; keyword search favours a distractor."""
    dense = [hit("rel", 0.9, "a hash table maps keys", 0), hit("dis", 0.3, "unrelated talk", 500)]
    sparse = [hit("dis", HALF * 20, "unrelated talk", 500), hit("rel", HALF / 10, "a hash table maps keys", 0)]
    return dense, sparse, N_POINTS


def test_ablation_ranks_the_same_question_differently_per_config():
    configs = ablation_eval.parse_configs("0:1,1:0", 0.65, 0.35)
    rows, errors = ablation_eval.run_ablation([entry()], lambda e: candidates(), real_rank, real_fuse, configs)
    assert errors == []
    ranks = rows[0]["ranking_rank"]
    assert ranks["dense-only"] == 1 and ranks["sparse-only"] == 2
    summary = ablation_eval.summarize(rows, configs, "ranking_rank", k=1, depth=10)
    assert summary["dense-only"]["recall_at_k"] == 1.0 and summary["sparse-only"]["recall_at_k"] == 0.0
    assert summary["sparse-only"]["mrr"] == 0.5 and summary["dense-only"]["mrr"] == 1.0


def test_ablation_summary_aggregates_across_questions_and_categories():
    configs = ablation_eval.parse_configs("0:1", 0.65, 0.35)[:1]  # just dense-only (the default is auto-appended)
    dense_name = configs[0].name
    rows = [
        {"category": "direct_fact", "ranking_rank": {dense_name: 1}},
        {"category": "direct_fact", "ranking_rank": {dense_name: 7}},
        {"category": "paraphrase", "ranking_rank": {dense_name: None}},
        {"category": "paraphrase", "ranking_rank": {dense_name: 2}},
    ]
    summary = ablation_eval.summarize(rows, configs, "ranking_rank", k=5, depth=10)[dense_name]
    assert summary["n"] == 4 and summary["recall_at_k"] == 0.5
    assert summary["mrr"] == pytest.approx((1 + 1 / 7 + 0 + 1 / 2) / 4)
    assert summary["by_category"]["direct_fact"] == {"n": 2, "recall_at_k": 0.5}
    assert summary["by_category"]["paraphrase"]["recall_at_k"] == 0.5
    assert summary["recall_ci95"][0] < 0.5 < summary["recall_ci95"][1]


def test_a_failing_fetch_is_reported_and_excluded():
    def fetch(e):
        if e["id"] == "bad":
            raise RuntimeError("qdrant down")
        return candidates()

    configs = ablation_eval.parse_configs("0:1", 0.65, 0.35)
    rows, errors = ablation_eval.run_ablation([entry(id="ok"), entry(id="bad")], fetch, real_rank, real_fuse, configs)
    assert [r["id"] for r in rows] == ["ok"] and errors[0]["id"] == "bad" and "qdrant down" in errors[0]["error"]


def test_production_ranking_applies_the_threshold_while_pure_ranking_does_not():
    dense = [hit("weak", 0.25, "a hash table is mentioned", 0)]  # cosine 0.25: far below the 0.55 distance cutoff
    configs = ablation_eval.parse_configs("0:1", 0.65, 0.35)
    rows, _ = ablation_eval.run_ablation([entry()], lambda e: (dense, [], N_POINTS), real_rank, real_fuse, configs)
    assert rows[0]["ranking_rank"][configs[0].name] == 1
    assert rows[0]["production_rank"][configs[0].name] is None


def test_beats_default_note_appears_only_when_something_is_better():
    configs = ablation_eval.parse_configs("0:1,0.65:0.35", 0.65, 0.35)
    default = next(c for c in configs if c.name.endswith("(default)")).name
    better = {"dense-only": {"recall_at_k": 0.9, "mrr": 0.8}, default: {"recall_at_k": 0.7, "mrr": 0.6}}
    assert any("dense-only beats the default" in n for n in ablation_eval.compare_with_default(better, configs))
    worse = {"dense-only": {"recall_at_k": 0.5, "mrr": 0.4}, default: {"recall_at_k": 0.7, "mrr": 0.6}}
    assert ablation_eval.compare_with_default(worse, configs) == []
    tied_recall = {"dense-only": {"recall_at_k": 0.7, "mrr": 0.7}, default: {"recall_at_k": 0.7, "mrr": 0.6}}
    assert ablation_eval.compare_with_default(tied_recall, configs)  # same recall, better MRR still flagged
    equal = {"dense-only": {"recall_at_k": 0.7, "mrr": 0.6}, default: {"recall_at_k": 0.7, "mrr": 0.6}}
    assert ablation_eval.compare_with_default(equal, configs) == []


# =============================== timestamp accuracy ======================================================================
@pytest.mark.parametrize("gold, start, end, tol, inside", [
    (100, 90, 180, 0, True), (90, 90, 180, 0, True), (180, 90, 180, 0, True),
    (89, 90, 180, 0, False), (181, 90, 180, 0, False),
    (75, 90, 180, 15, True),    # exactly start - tol: inside
    (74, 90, 180, 15, False),   # one second earlier: outside
    (195, 90, 180, 15, True), (196, 90, 180, 15, False),
    (60, 90, 180, 30, True), (59, 90, 180, 30, False),
])
def test_within_window_is_inclusive_at_the_edges(gold, start, end, tol, inside):
    assert timestamp_accuracy_eval.within_window(gold, start, end, tol) is inside


def test_distance_to_window():
    d = timestamp_accuracy_eval.distance_to_window
    assert d(100, 90, 180) == 0 and d(80, 90, 180) == 10 and d(200, 90, 180) == 20


def period(start, end, video="vid"):
    return {"video_id": video, "start_sec": start, "end_sec": end, "label": f"{start}-{end}", "score": 0.8}


def test_score_question_only_counts_periods_of_the_right_video():
    score = timestamp_accuracy_eval.score_question(100, [period(90, 180, "other")], "vid")
    assert score["periods_in_video"] == 0 and not any(score["within"].values()) and score["best_distance_s"] is None


def test_score_question_reports_both_definitions():
    """A wide period contains the gold moment (the spec's definition) while starting far before it."""
    score = timestamp_accuracy_eval.score_question(400, [period(100, 500)], "vid")
    assert score["within"] == {15: True, 30: True}
    assert score["start_within"] == {15: False, 30: False}
    assert score["best_distance_s"] == 0 and score["best_start_offset_s"] == 300


def test_score_question_uses_the_best_of_several_periods():
    score = timestamp_accuracy_eval.score_question(100, [period(300, 390), period(60, 120), period(500, 590)], "vid")
    assert score["within"][15] is True and score["nearest"]["start_sec"] == 60


def test_tolerance_changes_the_verdict():
    score = timestamp_accuracy_eval.score_question(100, [period(130, 220)], "vid", (15, 30))
    assert score["within"] == {15: False, 30: True}


def test_timestamp_summary_percentages_and_misses():
    rows = [
        {"periods": [period(90, 180)], "score": timestamp_accuracy_eval.score_question(100, [period(90, 180)], "vid")},
        {"periods": [period(400, 490)], "score": timestamp_accuracy_eval.score_question(100, [period(400, 490)], "vid")},
        {"periods": [], "score": timestamp_accuracy_eval.score_question(100, [], "vid")},
        {"periods": [period(110, 200)], "score": timestamp_accuracy_eval.score_question(100, [period(110, 200)], "vid")},
    ]
    summary = timestamp_accuracy_eval.summarize(rows)
    assert summary["n"] == 4 and summary["no_result"] == 1
    assert summary["within"] == {"15": 0.5, "30": 0.5}
    assert summary["start_within"]["15"] == 0.5  # period starting 10 s after the gold moment counts, 90-180 does not
    assert summary["median_period_width_s"] == 90


def test_timestamp_selection_needs_a_gold_time_and_a_video():
    chosen, left = timestamp_accuracy_eval.select_questions([
        entry(id="a"), entry(id="b", gold_timestamp_seconds=None), entry(id="c", video_id=None),
        entry(id="d", expects_refusal=True), entry(id="e", gold_timestamp_seconds=0)])
    assert [e["id"] for e in chosen] == ["a", "e"] and left["no_video_id"] == 1  # a timestamp of 0:00 is valid


# =============================== grounding ===============================================================================
REFUSAL = "I couldn't find that in the indexed videos."


@pytest.mark.parametrize("output, refused, strict, kind", [
    ({"answer": REFUSAL, "grounded": False}, True, True, "refusal_sentence"),
    ({"answer": REFUSAL.upper().rstrip("."), "grounded": False}, True, True, "refusal_sentence"),
    ({"answer": "Sure, it is X.", "grounded": False}, True, False, "ungrounded_answer"),
    ({"answer": "It is X [1].", "grounded": True}, False, False, "answered"),
    ({"answer": REFUSAL, "grounded": True}, True, True, "refusal_sentence"),
    ({}, True, False, "ungrounded_answer"),
])
def test_classify(output, refused, strict, kind):
    assert grounding_eval.classify(output, REFUSAL) == {"refused": refused, "strict_refusal": strict, "refusal_kind": kind}


def row(expects_refusal, refused, guard=None, strict=None):
    return {"id": "x", "expects_refusal": expects_refusal, "refused": refused,
            "strict_refusal": refused if strict is None else strict, "guard": guard}


def test_refusal_accuracy_and_false_refusal_rate_formulas():
    rows = (
        [row(True, True, "retrieval_threshold")] * 2 + [row(True, True, "answer_model")] + [row(True, False)]    # 3 of 4
        + [row(False, True, "answer_model")] + [row(False, False)] * 4                                          # 1 of 5
    )
    s = grounding_eval.summarize(rows)
    assert (s["total_unanswerable"], s["correct_refusals"], s["refusal_accuracy"]) == (4, 3, 0.75)
    assert (s["total_answerable"], s["wrongly_refused"], s["false_refusal_rate"]) == (5, 1, 0.2)
    assert s["correct_refusals_by_guard"] == {"retrieval_threshold": 2, "answer_model": 1, "uncited_answer": 0}


def test_a_system_that_refuses_everything_is_exposed_by_the_false_refusal_rate():
    s = grounding_eval.summarize([row(True, True, "answer_model")] * 5 + [row(False, True, "answer_model")] * 5)
    assert s["refusal_accuracy"] == 1.0 and s["false_refusal_rate"] == 1.0


def test_strict_accuracy_counts_only_the_refusal_sentence():
    s = grounding_eval.summarize([row(True, True, "uncited_answer", strict=False), row(True, True, "answer_model", strict=True)])
    assert s["refusal_accuracy"] == 1.0 and s["strict_refusal_accuracy"] == 0.5


def test_summary_handles_a_missing_class():
    s = grounding_eval.summarize([row(False, False)])
    assert s["refusal_accuracy"] is None and s["false_refusal_rate"] == 0.0


@pytest.mark.parametrize("refused, periods, kind, guard", [
    (False, 3, "answered", None), (True, 0, "refusal_sentence", "retrieval_threshold"),
    (True, 2, "refusal_sentence", "answer_model"), (True, 2, "ungrounded_answer", "uncited_answer"),
])
def test_guard_for(refused, periods, kind, guard):
    assert grounding_eval.guard_for(refused, periods, kind) == guard


# =============================== ingestion stats =========================================================================
LIBRARY = {"sources": {
    "PL1": {"indexed_videos": {
        "a": {"indexed": True, "duration": 3600, "language": "en"},
        "b": {"indexed": True, "duration": 1800, "language": "hi"},
        "c": {"indexed": False, "duration": 999, "language": "en"}}},
    "PL2": {"indexed_videos": {"a": {"indexed": True, "duration": 3600}, "d": {"indexed": True, "duration": 1800, "language": "es-MX"}}},
}}


def test_corpus_stats_dedupes_videos_and_skips_unindexed():
    stats = ingestion_stats.corpus_stats(LIBRARY, language_lookup=lambda v: None)
    assert stats["videos"] == 3 and stats["total_seconds"] == 3600 + 1800 + 1800 and stats["hours"] == 2.0
    assert stats["languages"] == {"en": 1, "hi": 1, "es": 1} and stats["non_english_videos"] == 2
    assert stats["video_ids"] == ["a", "b", "d"]


def test_corpus_stats_can_be_scoped_to_a_source():
    stats = ingestion_stats.corpus_stats(LIBRARY, ["PL1"], language_lookup=lambda v: None)
    assert stats["videos"] == 2 and stats["sources"] == ["PL1"]


def test_corpus_stats_falls_back_to_the_lookup_for_language_and_does_not_mutate_input():
    snapshot = json.dumps(LIBRARY, sort_keys=True)
    stats = ingestion_stats.corpus_stats({"sources": {"S": {"indexed_videos": {"x": {"indexed": True, "duration": 60}}}}},
                                         language_lookup=lambda v: "fr")
    assert stats["languages"] == {"fr": 1}
    ingestion_stats.corpus_stats(LIBRARY, language_lookup=lambda v: None)
    assert json.dumps(LIBRARY, sort_keys=True) == snapshot


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRunner:
    mode = "fake"

    def __init__(self, polls, clock, attached=False):
        self.polls, self.clock, self.attached, self.n = polls, clock, attached, 0

    def submit(self, url):
        return "job1", self.attached

    def status(self, job_id):
        self.n += 1
        return self.polls[min(self.n - 1, len(self.polls) - 1)]


def done_status(**kw):
    base = {"status": "done", "total_videos": 2, "skipped": [], "timings": {"total_seconds": 4.0},
            "indexed": [{"video_id": "a", "reused": False, "translation": {"batches_attempted": 2, "fallback_batches": 1}},
                        {"video_id": "b", "reused": False}]}
    base.update(kw)
    return base


def test_time_ingest_measures_wall_clock_until_done():
    clock = Clock()
    runner = FakeRunner([{"status": "processing"}, {"status": "processing"}, done_status()], clock)
    result = ingestion_stats.time_ingest(runner, "url", "cold", poll_interval=2.0, clock=clock, sleep=clock.sleep)
    assert result["elapsed_seconds"] == 4.0  # two sleeps of 2 s before the third poll saw "done"
    assert (result["videos_indexed"], result["videos_reused"], result["videos_skipped"]) == (2, 0, 0)
    assert result["translation"]["batches_attempted"] == 2


@pytest.mark.parametrize("status, message", [
    ({"status": "paused", "pause_reason": "youtube", "error": "rate limited"}, "paused"),
    ({"status": "error", "error": "boom"}, "failed"),
])
def test_a_paused_or_failed_job_is_not_a_timing(status, message):
    clock = Clock()
    with pytest.raises(ingestion_stats.BenchmarkError, match=message):
        ingestion_stats.time_ingest(FakeRunner([status], clock), "url", "cold", clock=clock, sleep=clock.sleep)


def test_a_job_that_never_finishes_times_out():
    clock = Clock()
    with pytest.raises(ingestion_stats.BenchmarkError, match="did not finish"):
        ingestion_stats.time_ingest(FakeRunner([{"status": "processing"}], clock), "url", "cold",
                                    poll_interval=5, timeout=20, clock=clock, sleep=clock.sleep)


def test_attaching_to_a_running_job_is_not_a_clean_timing():
    clock = Clock()
    with pytest.raises(ingestion_stats.BenchmarkError, match="already being indexed"):
        ingestion_stats.time_ingest(FakeRunner([done_status()], clock, attached=True), "url", "cold", clock=clock, sleep=clock.sleep)


def test_speedup():
    assert ingestion_stats.speedup(120.0, 8.0) == 15.0
    assert ingestion_stats.speedup(120.0, 0) is None


def test_http_runner_talks_to_the_ingest_endpoints():
    calls = []

    def response(code, body):
        return SimpleNamespace(status_code=code, json=lambda: body, raise_for_status=lambda: None)

    fake_requests = SimpleNamespace(
        post=lambda url, json=None, timeout=None: calls.append(("POST", url, json)) or response(200, {"job_id": "j9", "already_running": True}),
        get=lambda url, timeout=None: calls.append(("GET", url)) or response(200, {"status": "done"}))
    runner = ingestion_stats.HttpRunner("http://localhost:8000/")
    runner._requests = fake_requests
    assert runner.submit("https://youtu.be/x") == ("j9", True)
    assert runner.status("j9") == {"status": "done"}
    assert calls == [("POST", "http://localhost:8000/ingest", {"url": "https://youtu.be/x"}),
                     ("GET", "http://localhost:8000/ingest/status/j9")]
    runner._requests = SimpleNamespace(post=lambda *a, **k: response(409, {"detail": "busy"}))
    with pytest.raises(ingestion_stats.BenchmarkError, match="busy"):
        runner.submit("u")


def test_caveats_flag_skipped_videos_and_http_mode():
    phases = [{"phase": "cold", "videos_skipped": 2, "videos_indexed": 5, "videos_reused": 0},
              {"phase": "reuse", "videos_skipped": 2, "videos_indexed": 5, "videos_reused": 3}]
    text = " ".join(ingestion_stats.build_caveats(phases, "http"))
    assert "skipped" in text and "not every video" in text.lower() and "in memory" in text
    assert "in memory" not in " ".join(ingestion_stats.build_caveats(phases[:1], "inprocess"))


def test_aggregate_translation():
    total = ingestion_stats.aggregate_translation([
        {"translation": {"batches_attempted": 4, "fallback_batches": 1, "chunks_translated": 16}},
        {"translation": {"batches_attempted": 4, "fallback_batches": 0, "chunks_translated": 16}}, {}])
    assert total["batches_attempted"] == 8 and total["fallback_rate"] == 1 / 8
    assert ingestion_stats.aggregate_translation([{}]) is None


# =============================== translation stats =======================================================================
def tr(**kw):
    base = {"batches_attempted": 4, "batches_fell_back_to_smaller": 1, "batches_fell_back_to_individual": 0,
            "fallback_batches": 1, "chunks_translated": 16, "chunks_from_cache": 0, "cache_hit": False}
    base.update(kw)
    return base


TRLIB = {"sources": {
    "PL1": {"indexed_videos": {"a": {"language": "hi", "title": "A", "translation": tr()},
                               "b": {"language": "es", "title": "B", "translation": tr(batches_attempted=6, fallback_batches=0, chunks_translated=24)},
                               "c": {"language": "en", "title": "C"}}},
    "PL2": {"indexed_videos": {"a": {"language": "hi", "translation": tr()}}},
}}


def test_translation_rows_skip_untranslated_and_duplicate_videos():
    rows = translation_stats.rows_from_library(TRLIB)
    assert sorted(r["video_id"] for r in rows) == ["a", "b"]
    assert [r["video_id"] for r in translation_stats.rows_from_library(TRLIB, ["PL2"])] == ["a"]


def test_translation_aggregate_and_fallback_rate():
    stats = translation_stats.aggregate(translation_stats.rows_from_library(TRLIB))
    assert stats["videos_translated"] == 2 and stats["batches_attempted"] == 10 and stats["fallback_batches"] == 1
    assert stats["fallback_rate"] == 0.1 and stats["chunks_translated"] == 40 and stats["languages"] == {"hi": 1, "es": 1}


def test_translation_stats_with_only_cache_hits_has_no_rate():
    rows = [{"language": "hi", "translation": tr(batches_attempted=0, fallback_batches=0, chunks_translated=0, chunks_from_cache=16, cache_hit=True)}]
    stats = translation_stats.aggregate(rows)
    assert stats["fallback_rate"] is None and stats["videos_fully_cached"] == 1 and stats["chunks_total"] == 16


def test_translation_rows_from_a_job_status():
    status = {"source_id": "PL", "indexed": [{"video_id": "a", "language": "hi", "translation": tr()}, {"video_id": "b", "language": "en"}]}
    assert [r["video_id"] for r in translation_stats.rows_from_job(status)] == ["a"]


# =============================== eval-set template =======================================================================
TEMPLATE_LIB = {"sources": {
    "PL1": {"indexed_videos": {"a": {"video_id": "a", "title": "Alpha", "indexed": True},
                               "b": {"video_id": "b", "title": "Beta", "indexed": True},
                               "c": {"video_id": "c", "title": "Gamma", "indexed": False}}},
    "vid0000000X": {"indexed_videos": {"vid0000000X": {"video_id": "vid0000000X", "title": "Solo", "indexed": True}}},
}}


def test_template_has_a_stub_per_video_per_category():
    stubs = build_eval_template.build_template(TEMPLATE_LIB, cross_video=3, unanswerable=5)
    per_video = [s for s in stubs if s["category"] in build_eval_template.PER_VIDEO_CATEGORIES]
    assert len(per_video) == 3 * 4  # a, b, Solo; the unindexed video is left out
    assert {s["video_id"] for s in per_video} == {"a", "b", "vid0000000X"}
    assert sum(s["category"] == "cross_video" for s in stubs) == 3  # only PL1 has two videos
    assert sum(s["category"] == "unanswerable" for s in stubs) == 10
    assert all(s["expects_refusal"] for s in stubs if s["category"] == "unanswerable")


def test_template_stubs_are_blank_and_ids_unique():
    stubs = build_eval_template.build_template(TEMPLATE_LIB)
    assert all(s["question"] is None and s["gold_answer_contains"] is None for s in stubs)
    assert len({s["id"] for s in stubs}) == len(stubs) and stubs[0]["id"] == "e01"
    followups = [s for s in stubs if s["category"] == "followup"]
    assert followups and all("history" in s for s in followups)


def test_template_can_be_limited_to_one_source():
    stubs = build_eval_template.build_template(TEMPLATE_LIB, source="vid0000000X")
    assert {s["source_id"] for s in stubs} == {"vid0000000X"} and not any(s["category"] == "cross_video" for s in stubs)


def test_template_command_never_overwrites_hand_written_work(tmp_path):
    lib = tmp_path / "library.json"
    lib.write_text(json.dumps(TEMPLATE_LIB), encoding="utf-8")
    out = tmp_path / "eval_set.json"
    args = ["--library", str(lib), "--out", str(out)]
    assert build_eval_template.main(args) == 0 and out.exists()
    out.write_text("[]  # my questions", encoding="utf-8")
    assert build_eval_template.main(args) != 0 and out.read_text(encoding="utf-8") == "[]  # my questions"
    assert build_eval_template.main(args + ["--force"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))


def test_template_command_reports_a_missing_library(tmp_path):
    assert build_eval_template.main(["--library", str(tmp_path / "none.json"), "--out", str(tmp_path / "o.json")]) != 0


def test_a_generated_template_loads_once_one_question_is_filled_in(tmp_path):
    stubs = build_eval_template.build_template(TEMPLATE_LIB)
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(stubs), encoding="utf-8")
    with pytest.raises(EvalSetError, match="no usable questions"):
        load_eval_set(path)
    stubs[0]["question"], stubs[0]["gold_answer_contains"] = "What is Alpha?", "alpha"
    path.write_text(json.dumps(stubs), encoding="utf-8")
    entries, info = load_eval_set(path)
    assert len(entries) == 1 and info["skipped_stubs"] == len(stubs) - 1


# =============================== results rendering =======================================================================
def result_file(directory, name, payload):
    (directory / name).write_text(json.dumps({"generated_at": "2026-10-10T00:00:00+00:00", "git": {"commit": "abc1234def"}, **payload}), encoding="utf-8")


def test_latest_picks_the_newest_file_and_can_require_a_key(tmp_path):
    result_file(tmp_path, "ingestion_20261001-000000.json", {"corpus": {"videos": 3}})
    result_file(tmp_path, "ingestion_20261002-000000.json", {"benchmark": {"cold_seconds": 90}})
    assert "benchmark" in render_results.latest("ingestion", tmp_path)
    assert render_results.latest("ingestion", tmp_path, require="corpus")["corpus"]["videos"] == 3
    assert render_results.latest("ablation", tmp_path) is None


def test_replace_section_keeps_crlf_line_endings():
    readme = "# Title\r\n\r\nintro\r\n\r\n<!-- RESULTS:START -->\r\nold\r\n<!-- RESULTS:END -->\r\n\r\nmore\r\n"
    updated = render_results.replace_section(readme, "## Results\n\nnew row\n")
    assert "old" not in updated and "new row" in updated and "more" in updated
    assert "\n" not in updated.replace("\r\n", "")  # no bare LF anywhere


def test_replace_section_appends_markers_when_missing():
    updated = render_results.replace_section("# Title\n", "## Results\n")
    assert updated.count(render_results.START) == 1 and updated.startswith("# Title") and "\r" not in updated


def test_replacing_twice_is_stable():
    once = render_results.replace_section("# T\n", "## Results\nrow\n")
    assert render_results.replace_section(once, "## Results\nrow\n") == once


def test_rendering_with_no_results_says_not_measured(tmp_path):
    text = render_results.render_markdown(render_results.load_results(tmp_path))
    assert "not measured" in text and "ablation_eval" in text


def test_rendering_uses_the_measured_numbers(tmp_path):
    configs = ablation_eval.parse_configs("0:1,0.65:0.35", 0.65, 0.35)
    rows = [{"category": "direct_fact", "ranking_rank": {c.name: 1 if "dense" in c.name else 3 for c in configs}}] * 4
    ranking = ablation_eval.summarize(rows, configs, "ranking_rank", 5, 10)
    result_file(tmp_path, "ablation_20261010-000000.json", {"n_questions": 4, "params": {"k": 5}, "ranking": ranking})
    result_file(tmp_path, "grounding_20261010-000000.json", {"summary": {"refusal_accuracy": 0.9, "false_refusal_rate": 0.05, "total_unanswerable": 10, "total_answerable": 40}})
    result_file(tmp_path, "ingestion_20261010-000000.json", {"benchmark": {"cold_seconds": 120.0, "reingest_seconds": 8.0, "reingest_speedup": 15.0}})
    result_file(tmp_path, "ingestion_20261010-000001.json", {"corpus": {"videos": 12, "hours": 3.5, "languages": {"en": 10, "hi": 2}, "qdrant": {"chunks_in_scope": 480}}})
    res = render_results.load_results(tmp_path)
    text = render_results.render_markdown(res)
    assert "| 12 | 3.5 | 480 | en 10, hi 2 |" in text
    assert "90.0%" in text and "5.0%" in text and "(15.0x faster)" in text
    assert "abc1234" in text
    numbers = render_results.numbers(res)
    assert numbers["dense_recall"] == 1.0 and numbers["hybrid_recall"] == 1.0 and numbers["eval_questions"] == 4
    assert "[Y] hybrid" in render_results.resume_numbers(res)


# =============================== command-line entry points ===============================================================
def write_eval_set(tmp_path, entries):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def only_result(directory, prefix):
    [path] = list(directory.glob(f"{prefix}_*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_ablation_command_writes_a_results_file(tmp_path, monkeypatch):
    eval_set = write_eval_set(tmp_path, [
        {"id": "a", "source_id": "PL", "question": "what is a hash table", "gold_answer_contains": "hash table"},
        {"id": "b", "source_id": "PL", "question": "refuse me", "expects_refusal": True},
    ])
    monkeypatch.setattr(ablation_eval, "prepare_query", lambda e, mode: (e["question"], [], [], ""))
    monkeypatch.setattr(qdrant_store, "fetch_candidates", lambda q, s, m, x=(): candidates())
    monkeypatch.setattr(qdrant_store, "scan_source_chunks", lambda source_id: [  # no real Qdrant, no embedding model
        {"text": "a hash table maps keys", "source_language": "hi"}, {"text": "unrelated talk", "source_language": "hi"}])
    out = tmp_path / "results"
    assert ablation_eval.main(["--eval-set", eval_set, "--out-dir", str(out)]) == 0
    data = only_result(out, "ablation")
    assert data["n_questions"] == 1 and data["left_out"]["expects_refusal"] == 1
    assert {c["name"] for c in data["configs"]} >= {"dense-only", "sparse-only", "hybrid 65/35 (default)"}
    assert data["ranking"]["dense-only"]["recall_at_k"] == 1.0
    assert "commit" in data["git"] and data["generated_at"]
    assert data["questions"][0]["ranking_rank"]["dense-only"] == 1


def test_ablation_command_reports_a_bad_eval_set(tmp_path):
    assert ablation_eval.main(["--eval-set", str(tmp_path / "missing.json"), "--out-dir", str(tmp_path)]) != 0


def test_timestamp_command(tmp_path, monkeypatch):
    eval_set = write_eval_set(tmp_path, [
        {"id": "a", "source_id": "PL", "video_id": "vid", "question": "q1", "gold_timestamp_seconds": "1:40"},
        {"id": "b", "source_id": "PL", "video_id": "vid", "question": "q2", "gold_timestamp_seconds": 900},
        {"id": "c", "source_id": "PL", "video_id": "vid", "question": "q3"},  # no gold time: not used
    ])
    answers = {"q1": [{**period(95, 185), "video_title": "T"}], "q2": []}
    monkeypatch.setattr(timestamp_accuracy_eval, "run_search",
                        lambda e, mode: {"standalone": e["question"], "periods": answers[e["question"]]})
    out = tmp_path / "results"
    assert timestamp_accuracy_eval.main(["--eval-set", eval_set, "--out-dir", str(out)]) == 0
    summary = only_result(out, "timestamp")["summary"]
    assert summary["n"] == 2 and summary["within"] == {"15": 0.5, "30": 0.5} and summary["no_result"] == 1


def test_timestamp_command_needs_gold_timestamps(tmp_path):
    eval_set = write_eval_set(tmp_path, [{"id": "a", "source_id": "PL", "video_id": "vid", "question": "q"}])
    assert timestamp_accuracy_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path)]) != 0


def grounding_entries():
    return [
        {"id": "ok", "source_id": "PL", "question": "answerable one", "gold_answer_contains": "x"},
        {"id": "no", "source_id": "PL", "question": "unanswerable one", "expects_refusal": True},
        {"id": "fr", "source_id": "PL", "question": "wrongly refused", "gold_answer_contains": "x"},
    ]


def fake_answers(monkeypatch, behaviour):
    from backend.rag import answer

    monkeypatch.setattr(grounding_eval, "run_search",
                        lambda e, mode: {"standalone": e["question"], "periods": [period(0, 90)] if e["id"] != "no" else [], "prior": []})
    monkeypatch.setattr(answer, "generate_answer", lambda question, periods, prior: behaviour(question))
    return answer.REFUSAL_PHRASE


def test_grounding_command_computes_both_rates(tmp_path, monkeypatch):
    refusal = None

    def behaviour(question):
        if question.startswith("answerable"):
            return {"answer": "It is x [1].", "citations": [1], "grounded": True}
        return {"answer": refusal, "citations": [], "grounded": False}

    refusal = fake_answers(monkeypatch, behaviour)
    eval_set = write_eval_set(tmp_path, grounding_entries())
    assert grounding_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path / "r")]) == 0
    s = only_result(tmp_path / "r", "grounding")["summary"]
    assert (s["refusal_accuracy"], s["false_refusal_rate"]) == (1.0, 0.5)
    assert s["answerable_refused"] == ["fr"] and s["correct_refusals_by_guard"]["retrieval_threshold"] == 1


def test_grounding_command_stops_cleanly_when_the_quota_runs_out(tmp_path, monkeypatch):
    from backend.util import QuotaExhausted

    def behaviour(question):
        if question.startswith("unanswerable"):
            raise QuotaExhausted("daily quota", 3600)
        return {"answer": "x [1]", "citations": [1], "grounded": True}

    fake_answers(monkeypatch, behaviour)
    # Make the second question reach the (quota-limited) answer step.
    monkeypatch.setattr(grounding_eval, "run_search",
                        lambda e, mode: {"standalone": e["question"], "periods": [period(0, 90)], "prior": []})
    eval_set = write_eval_set(tmp_path, grounding_entries())
    assert grounding_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path / "r")]) == grounding_eval.QUOTA_EXIT_CODE
    data = only_result(tmp_path / "r", "grounding")
    assert "quota" in data["aborted"] and len(data["questions"]) == 1  # what finished is kept


def test_grounding_command_survives_one_failing_question(tmp_path, monkeypatch):
    def behaviour(question):
        if question.startswith("wrongly"):
            raise RuntimeError("llm exploded")
        return {"answer": "x [1]", "citations": [1], "grounded": True}

    fake_answers(monkeypatch, behaviour)
    eval_set = write_eval_set(tmp_path, grounding_entries())
    assert grounding_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path / "r")]) == 0
    data = only_result(tmp_path / "r", "grounding")
    assert [e["id"] for e in data["errors"]] == ["fr"] and len(data["questions"]) == 2


def test_translation_command(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps(TRLIB), encoding="utf-8")
    assert translation_stats.main(["--library", str(library), "--out-dir", str(tmp_path / "r")]) == 0
    assert only_result(tmp_path / "r", "translation")["summary"]["fallback_rate"] == 0.1


def test_translation_command_explains_when_there_is_nothing_to_report(tmp_path):
    library = tmp_path / "library.json"
    library.write_text(json.dumps({"sources": {"S": {"indexed_videos": {"a": {"language": "en"}}}}}), encoding="utf-8")
    assert translation_stats.main(["--library", str(library), "--out-dir", str(tmp_path / "r")]) != 0


def test_ingestion_command_reports_the_corpus(tmp_path, monkeypatch):
    library = tmp_path / "library.json"
    library.write_text(json.dumps(LIBRARY), encoding="utf-8")
    monkeypatch.setattr(ingestion_stats, "_cached_language", lambda v: None)
    monkeypatch.setattr(ingestion_stats, "qdrant_chunk_counts",
                        lambda ids: {"chunks_in_scope": 99, "chunks_in_collection": 120, "collection": "c", "per_video": {}})
    assert ingestion_stats.main(["--library", str(library), "--out-dir", str(tmp_path / "r")]) == 0
    corpus = only_result(tmp_path / "r", "ingestion")["corpus"]
    assert corpus["videos"] == 3 and corpus["qdrant"]["chunks_in_scope"] == 99 and "video_ids" not in corpus


def test_ingestion_command_still_reports_the_library_when_qdrant_is_down(tmp_path, monkeypatch):
    library = tmp_path / "library.json"
    library.write_text(json.dumps(LIBRARY), encoding="utf-8")
    monkeypatch.setattr(ingestion_stats, "_cached_language", lambda v: None)

    def down(ids):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(ingestion_stats, "qdrant_chunk_counts", down)
    assert ingestion_stats.main(["--library", str(library), "--out-dir", str(tmp_path / "r")]) == 0
    corpus = only_result(tmp_path / "r", "ingestion")["corpus"]
    assert corpus["hours"] == 2.0 and "no route" in corpus["qdrant_error"]


def test_render_command_updates_a_crlf_readme_in_place(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    readme = tmp_path / "README.md"
    readme.write_bytes(b"# Project\r\n\r\ntext\r\n\r\n<!-- RESULTS:START -->\r\nstale\r\n<!-- RESULTS:END -->\r\n")
    assert render_results.main(["--results-dir", str(results), "--write-readme", str(readme)]) == 0
    raw = readme.read_bytes()
    assert b"stale" not in raw and b"not measured" in raw
    assert raw.count(b"\r\n") == raw.count(b"\n")  # still pure CRLF


# =============================== corpus-aware ablation metrics =============================================================
CORPUS = [
    {"text": "a hash table maps keys", "source_language": "hi-IN"},
    {"text": "hash table collisions use chaining", "source_language": "hi"},
    {"text": "completely different topic", "source_language": "en"},
]


def test_corpus_matches_counts_relevant_chunks_and_their_language():
    assert ablation_eval.corpus_matches(CORPUS, "hash table") == {"total_relevant": 2, "language": "hi"}
    assert ablation_eval.corpus_matches(CORPUS, "quantum") == {"total_relevant": 0, "language": None}


def test_questions_whose_gold_matches_nothing_are_separated_out():
    entries = [entry(id="ok"), entry(id="typo", gold_answer_contains="hash tabel")]
    usable, unmatched, info = ablation_eval.split_unmatched(entries, lambda source_id: CORPUS)
    assert [e["id"] for e in usable] == ["ok"] and [e["id"] for e in unmatched] == ["typo"]
    assert info["ok"]["total_relevant"] == 2 and info["typo"]["total_relevant"] == 0


def test_each_source_is_scanned_once():
    scans = []

    def corpus_for(source_id):
        scans.append(source_id)
        return CORPUS

    ablation_eval.split_unmatched([entry(id="a"), entry(id="b"), entry(id="c", source_id="PL2")], corpus_for)
    assert sorted(scans) == ["PL", "PL2"]


def test_an_unreadable_corpus_leaves_the_questions_in_rather_than_dropping_them():
    def broken(source_id):
        raise ConnectionError("qdrant down")

    usable, unmatched, info = ablation_eval.split_unmatched([entry()], broken)
    assert len(usable) == 1 and unmatched == [] and info == {}


def test_precision_ndcg_and_chunk_recall_reach_the_rows_and_summary():
    configs = ablation_eval.parse_configs("0:1,1:0", 0.65, 0.35)
    info = {"e1": {"total_relevant": 1, "language": "hi"}}
    rows, _ = ablation_eval.run_ablation([entry()], lambda e: candidates(), real_rank, real_fuse, configs, k=5, corpus_info=info)
    dense, sparse = rows[0]["ranking_metrics"]["dense-only"], rows[0]["ranking_metrics"]["sparse-only"]
    assert dense == {"precision": 0.2, "ndcg": 1.0, "chunk_recall": 1.0}                       # relevant chunk at rank 1 of 2 returned
    assert sparse["ndcg"] == pytest.approx(1 / math.log2(3)) and sparse["chunk_recall"] == 1.0   # at rank 2
    assert rows[0]["language"] == "hi" and rows[0]["total_relevant_chunks"] == 1
    summary = ablation_eval.summarize(rows, configs, "ranking_rank", 5, 10)
    assert summary["dense-only"]["ndcg_at_k"] == 1.0 and summary["sparse-only"]["precision_at_k"] == 0.2
    assert summary["dense-only"]["by_language"] == {"hi": {"n": 1, "recall_at_k": 1.0}}
    assert ablation_eval.summarize(rows, configs, "production_rank", 5, 10)["dense-only"]["ndcg_at_k"] is None  # ranking table only


def test_chunk_recall_shows_what_the_hit_rate_hides():
    """Four chunks mention the answer; retrieval finds one. The hit rate says 100%, true recall says 25%."""
    configs = ablation_eval.parse_configs("0:1", 0.65, 0.35)[:1]
    rows, _ = ablation_eval.run_ablation([entry()], lambda e: candidates(), real_rank, real_fuse, configs, k=5,
                                         corpus_info={"e1": {"total_relevant": 4, "language": None}})
    summary = ablation_eval.summarize(rows, configs, "ranking_rank", 5, 10)["dense-only"]
    assert summary["recall_at_k"] == 1.0 and summary["chunk_recall_at_k"] == 0.25


def test_ablation_command_excludes_and_reports_unmatched_gold(tmp_path, monkeypatch):
    eval_set = write_eval_set(tmp_path, [
        {"id": "a", "source_id": "PL", "question": "what is a hash table", "gold_answer_contains": "hash table"},
        {"id": "typo", "source_id": "PL", "question": "what is a hash table", "gold_answer_contains": "hash tabel"},
    ])
    monkeypatch.setattr(ablation_eval, "prepare_query", lambda e, mode: (e["question"], [], [], ""))
    monkeypatch.setattr(qdrant_store, "fetch_candidates", lambda q, s, m, x=(): candidates())
    monkeypatch.setattr(qdrant_store, "scan_source_chunks", lambda source_id: CORPUS)
    out = tmp_path / "results"
    assert ablation_eval.main(["--eval-set", eval_set, "--out-dir", str(out)]) == 0
    data = only_result(out, "ablation")
    assert data["n_questions"] == 1 and data["unmatched_gold"] == [{"id": "typo", "gold_answer_contains": "hash tabel"}]
    assert data["questions"][0]["total_relevant_chunks"] == 2 and data["ranking"]["dense-only"]["chunk_recall_at_k"] == 0.5
    assert data["corpus_scan"] is True
    assert ablation_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path / "all"), "--keep-unmatched"]) == 0
    assert only_result(tmp_path / "all", "ablation")["n_questions"] == 2


def test_ablation_command_works_without_the_corpus_scan(tmp_path, monkeypatch):
    eval_set = write_eval_set(tmp_path, [{"id": "a", "source_id": "PL", "question": "what is a hash table", "gold_answer_contains": "hash table"}])
    monkeypatch.setattr(ablation_eval, "prepare_query", lambda e, mode: (e["question"], [], [], ""))
    monkeypatch.setattr(qdrant_store, "fetch_candidates", lambda q, s, m, x=(): candidates())
    monkeypatch.setattr(qdrant_store, "scan_source_chunks", lambda source_id: pytest.fail("must not scan"))
    assert ablation_eval.main(["--eval-set", eval_set, "--out-dir", str(tmp_path / "r"), "--no-corpus-scan"]) == 0
    assert only_result(tmp_path / "r", "ablation")["ranking"]["dense-only"]["chunk_recall_at_k"] is None


def test_template_stubs_include_a_reference_answer_field():
    stubs = build_eval_template.build_template(TEMPLATE_LIB)
    assert all("gold_answer" in s and s["gold_answer"] is None for s in stubs)


# =============================== README results for the new metrics ========================================================
def test_rendering_includes_retrieval_quality_answer_quality_performance_and_robustness(tmp_path):
    configs = ablation_eval.parse_configs("0:1,0.65:0.35", 0.65, 0.35)
    rows = [{"category": "direct_fact", "ranking_rank": {c.name: 1 for c in configs},
             "ranking_metrics": {c.name: {"precision": 0.4, "ndcg": 0.9, "chunk_recall": 0.5} for c in configs}}] * 4
    result_file(tmp_path, "ablation_20261010-000000.json", {"n_questions": 4, "params": {"k": 5}, "ranking": ablation_eval.summarize(rows, configs, "ranking_rank", 5, 10)})
    result_file(tmp_path, "answer_quality_20261010-000000.json", {
        "params": {"judge_is_answer_model": True},
        "summary": {"n_questions": 4, "answered_rate": 0.75, "metrics": {"faithfulness": {"mean": 0.92}, "correctness": {"mean": 0.8}, "answer_relevancy": {"mean": None}}}})
    result_file(tmp_path, "robustness_20261010-000000.json", {"changes": {"typos": {"dense-only": {"recall_change": -0.2}, "sparse-only": {"recall_change": -0.1}}}})
    result_file(tmp_path, "performance_20261010-000000.json", {"summary": {
        "latency_seconds": {"total": {"p50": 1.2, "p95": 3.4}}, "llm_calls_per_question": 1.8, "cost_per_100_questions_usd": None}})
    res = render_results.load_results(tmp_path)
    text = render_results.render_markdown(res)
    assert "P@5" in text and "nDCG@5" in text and "ChunkRecall@5" in text and "0.900" in text
    assert "faithfulness | 92.0%" in text and "correctness | 80.0%" in text and "answer relevancy" not in text   # unscored metric left out
    assert "same model that writes the answers" in text                                                          # the optimism warning
    assert "1.20s / 3.40s" in text and "1.80" in text
    assert "dense-only" in text and "-20.0 points" in text and "'typos'" in text
    bullets = render_results.resume_numbers(res)
    assert "0.400 / 0.900 / 0.500" in bullets and "92.0% / 80.0% / n/a" in bullets and "1.20s / 3.40s" in bullets


def test_rendering_without_the_new_results_says_so(tmp_path):
    text = render_results.render_markdown(render_results.load_results(tmp_path))
    assert "evaluation.answer_quality_eval" in text and "evaluation.performance_eval" in text
