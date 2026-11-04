"""Hybrid fusion and period grouping: pure functions, tested with hand-made hits (no Qdrant, no models)."""
import math
from types import SimpleNamespace

import pytest

from backend import config
from backend.index import qdrant_store
from backend.rag import retrieve

N_POINTS = 50
HALF = config.SPARSE_HALF_SATURATION_TERMS * qdrant_store._TF_TYPICAL * math.log(N_POINTS)  # BM25 score that maps to 0.5


def hit(point_id, score, text="chunk", video="v1", start=0, **payload):
    base = {"chunk_id": f"{video}:{start}", "video_id": video, "video_title": "T", "start_sec": start,
            "end_sec": start + 90, "text": text, "positions": {"PL": 0}}
    return SimpleNamespace(id=point_id, score=score, payload={**base, **payload})


def fuse(dense, sparse, threshold=1.0, **kwargs):
    return qdrant_store._fuse_and_threshold(dense, sparse, threshold, 50, "PL", N_POINTS, **kwargs)


@pytest.fixture(autouse=True)
def plain_guards(monkeypatch):
    """Switch the precision guards off by default so fusion arithmetic is tested on its own."""
    monkeypatch.setattr(config, "RELATIVE_SCORE_FLOOR", 0.0)
    monkeypatch.setattr(config, "MIN_DENSE_SIMILARITY", 0.0)
    monkeypatch.setattr(config, "SPARSE_WEIGHT", 0.65)
    monkeypatch.setattr(config, "DENSE_WEIGHT", 0.35)


# --- the fusion formula ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("sparse_w, dense_w, expected", [
    (0.65, 0.35, 0.65 * 0.5 + 0.35 * 0.8),           # the defaults
    (0.5, 0.5, (0.5 * 0.5 + 0.5 * 0.8) / 1.0),
    (2.0, 1.0, (2.0 * 0.5 + 1.0 * 0.8) / 3.0),       # weights are normalised by their sum, not assumed to add to 1
    (0.0, 1.0, 0.8),                                  # dense-only
    (1.0, 0.0, 0.5),                                  # sparse-only
])
def test_fusion_formula(sparse_w, dense_w, expected):
    """fused = (sparse_weight*sparse + dense_weight*dense) / (sparse_weight + dense_weight)"""
    [result] = fuse([hit("a", 0.8)], [hit("a", HALF)], sparse_weight=sparse_w, dense_weight=dense_w)
    assert result["score"] == pytest.approx(expected, abs=1e-4)
    assert result["distance"] == pytest.approx(1 - expected, abs=1e-4)


def test_defaults_come_from_the_config(monkeypatch):
    monkeypatch.setattr(config, "SPARSE_WEIGHT", 0.2)
    monkeypatch.setattr(config, "DENSE_WEIGHT", 0.8)
    [result] = fuse([hit("a", 0.8)], [hit("a", HALF)])
    assert result["score"] == pytest.approx(0.2 * 0.5 + 0.8 * 0.8, abs=1e-4)


def test_explicit_weights_override_the_config_for_one_call_only():
    dense, sparse = [hit("a", 0.8)], [hit("a", HALF)]
    default = fuse(dense, sparse)[0]["score"]
    overridden = fuse(dense, sparse, sparse_weight=0.0, dense_weight=1.0)[0]["score"]
    assert overridden != default
    assert (config.SPARSE_WEIGHT, config.DENSE_WEIGHT) == (0.65, 0.35)  # untouched
    assert fuse(dense, sparse)[0]["score"] == default


def test_passing_the_configured_weights_equals_passing_none():
    dense, sparse = [hit("a", 0.7), hit("b", 0.4)], [hit("b", HALF * 2), hit("c", HALF)]
    assert fuse(dense, sparse) == fuse(dense, sparse, sparse_weight=0.65, dense_weight=0.35)


def test_zero_total_weight_is_an_error():
    with pytest.raises(ValueError):
        fuse([hit("a", 0.5)], [], sparse_weight=0.0, dense_weight=0.0)


def test_sparse_score_saturates_below_one():
    [low] = fuse([], [hit("a", HALF / 10)], sparse_weight=1.0, dense_weight=0.0)
    [high] = fuse([], [hit("b", HALF * 1000)], sparse_weight=1.0, dense_weight=0.0)
    assert 0 < low["score"] < 0.5 < high["score"] < 1.0


def test_negative_scores_are_clamped_to_zero():
    [result] = fuse([hit("a", -0.3)], [hit("a", -5.0)], sparse_weight=0.5, dense_weight=0.5)
    assert result["score"] == 0.0


def test_chunks_found_by_only_one_leg_are_scored_with_zero_for_the_other():
    dense = [hit("dense_only", 0.9, start=0)]
    sparse = [hit("sparse_only", HALF, start=500)]
    results = fuse(dense, sparse, sparse_weight=0.5, dense_weight=0.5)
    scores = {r["start_sec"]: r["score"] for r in results}
    assert scores[0] == pytest.approx(0.5 * 0.9, abs=1e-4)    # no lexical match: sparse counts as 0
    assert scores[500] == pytest.approx(0.5 * 0.5, abs=1e-4)  # not in the dense pool: dense counts as 0


# --- threshold and guards ---------------------------------------------------------------------------------------
def test_results_are_best_first_and_cut_by_the_absolute_threshold():
    dense = [hit("a", 0.9, start=0), hit("b", 0.6, start=100), hit("c", 0.1, start=200)]
    out = fuse(dense, [hit("a", HALF * 4), hit("b", HALF)], threshold=0.6)
    assert [r["start_sec"] for r in out] == [0, 100]
    assert [r["distance"] for r in out] == sorted(r["distance"] for r in out)


def test_an_irrelevant_query_returns_nothing():
    assert fuse([hit("a", 0.05)], [], threshold=0.55) == []


def test_relative_score_floor_drops_weak_stragglers(monkeypatch):
    monkeypatch.setattr(config, "RELATIVE_SCORE_FLOOR", 0.8)
    dense = [hit("strong", 1.0, start=0), hit("weak", 0.45, start=500)]
    sparse = [hit("strong", HALF * 50)]
    assert [r["start_sec"] for r in fuse(dense, sparse, threshold=0.9)] == [0]


def test_the_best_chunk_always_survives_the_floor(monkeypatch):
    monkeypatch.setattr(config, "RELATIVE_SCORE_FLOOR", 0.99)
    assert len(fuse([hit("a", 0.7)], [], threshold=0.9)) == 1


def test_min_dense_similarity_removes_off_topic_keyword_matches(monkeypatch):
    monkeypatch.setattr(config, "MIN_DENSE_SIMILARITY", 0.2)
    dense = [hit("topical", 0.6, start=0), hit("offtopic", 0.05, start=300)]
    sparse = [hit("topical", HALF * 10), hit("offtopic", HALF * 10)]
    assert [r["start_sec"] for r in fuse(dense, sparse, threshold=0.9)] == [0]


def test_a_sparse_only_candidate_is_not_judged_by_the_dense_gate(monkeypatch):
    """It has dense=0 only because it fell outside the dense candidate pool, which says nothing about topic."""
    monkeypatch.setattr(config, "MIN_DENSE_SIMILARITY", 0.2)
    out = fuse([], [hit("lexical", HALF * 40)], threshold=0.9)
    assert len(out) == 1


def test_max_results_is_only_a_safety_cap():
    dense = [hit(f"p{i}", 0.9, start=i * 200) for i in range(10)]
    assert len(qdrant_store._fuse_and_threshold(dense, [], 0.9, 3, "PL", N_POINTS)) == 3


# --- ranking helper used by the ablation ------------------------------------------------------------------------
def test_rank_candidates_has_no_threshold_and_is_best_first():
    dense = [hit("a", 0.2, start=0), hit("b", 0.9, start=100), hit("c", 0.5, start=200)]
    ranked = qdrant_store.rank_candidates(dense, [], N_POINTS, "PL", sparse_weight=0.0, dense_weight=1.0)
    assert [r["start_sec"] for r in ranked] == [100, 200, 0]  # even the 0.2 candidate is returned


def test_rank_candidates_leaves_out_candidates_with_no_signal():
    """Sparse-only weights: a chunk only the dense leg found has fused score 0 and must not be 'ranked'."""
    dense = [hit("dense_only", 0.99, start=0)]
    sparse = [hit("lexical", HALF, start=100)]
    ranked = qdrant_store.rank_candidates(dense, sparse, N_POINTS, "PL", sparse_weight=1.0, dense_weight=0.0)
    assert [r["start_sec"] for r in ranked] == [100]


def test_dense_only_and_sparse_only_rank_differently():
    dense = [hit("a", 0.9, start=0), hit("b", 0.3, start=100)]
    sparse = [hit("b", HALF * 20, start=100), hit("a", HALF / 10, start=0)]
    by_dense = qdrant_store.rank_candidates(dense, sparse, N_POINTS, "PL", 0.0, 1.0)
    by_sparse = qdrant_store.rank_candidates(dense, sparse, N_POINTS, "PL", 1.0, 0.0)
    assert [r["start_sec"] for r in by_dense] == [0, 100]
    assert [r["start_sec"] for r in by_sparse] == [100, 0]


def test_result_dicts_carry_what_the_app_needs():
    [r] = fuse([hit("a", 0.9, start=75, source_language="hi", transcript_source="youtube_generated")], [], threshold=0.9)
    assert r["timestamp"] == "1:15" and r["position"] == 0
    assert (r["source_language"], r["transcript_source"]) == ("hi", "youtube_generated")


def test_hybrid_search_passes_weights_through(monkeypatch):
    monkeypatch.setattr(qdrant_store, "fetch_candidates",
                        lambda query, source_id, max_results, expansions=(): ([hit("a", 0.8)], [hit("a", HALF)], N_POINTS))
    default = qdrant_store.hybrid_search("q", "PL", 1.0, 10)[0]["score"]
    dense_only = qdrant_store.hybrid_search("q", "PL", 1.0, 10, sparse_weight=0.0, dense_weight=1.0)[0]["score"]
    assert dense_only == pytest.approx(0.8, abs=1e-4) and default != dense_only


def test_retrieve_chunks_forwards_weights_to_hybrid_search(monkeypatch):
    seen = {}

    def fake(query, source_id, threshold, max_results, expansions=(), sparse_weight=None, dense_weight=None):
        seen.update(sparse=sparse_weight, dense=dense_weight, expansions=expansions)
        return []

    monkeypatch.setattr(qdrant_store, "hybrid_search", fake)
    retrieve.retrieve_chunks("q", "PL", ["alt"], sparse_weight=0.2, dense_weight=0.8)
    assert seen == {"sparse": 0.2, "dense": 0.8, "expansions": ["alt"]}
    retrieve.retrieve_chunks("q", "PL")  # the app's normal call: no overrides
    assert seen["sparse"] is None and seen["dense"] is None


# --- period grouping ------------------------------------------------------------------------------------------
def chunk(start, video="v1", position=0, score=0.8, text=None):
    return {"video_id": video, "video_title": f"Video {video}", "position": position, "start_sec": start,
            "end_sec": start + 90, "score": score, "distance": 1 - score, "text": text or f"text at {start}"}


@pytest.fixture(autouse=True)
def period_settings(monkeypatch):
    monkeypatch.setattr(config, "PERIOD_GAP_SECONDS", 180)
    monkeypatch.setattr(config, "CHUNK_SECONDS", 90)


def test_nearby_chunks_merge_into_one_period():
    [period] = retrieve.group_into_periods([chunk(0), chunk(70), chunk(140)])
    assert (period["start_sec"], period["end_sec"], period["chunks"]) == (0, 230, 3)
    assert period["is_range"] and period["label"] == "0:00 - 3:50"


def test_distant_chunks_stay_separate_periods():
    periods = retrieve.group_into_periods([chunk(0), chunk(500)])
    assert [(p["start_sec"], p["end_sec"]) for p in periods] == [(0, 90), (500, 590)]


@pytest.mark.parametrize("gap, merged", [(179, True), (180, False), (181, False)])
def test_the_gap_is_measured_between_hit_starts(gap, merged):
    periods = retrieve.group_into_periods([chunk(0), chunk(gap)])
    assert (len(periods) == 1) is merged


def test_a_topic_that_recurs_gives_one_period_per_recurrence():
    periods = retrieve.group_into_periods([chunk(0), chunk(60), chunk(600), chunk(660), chunk(1500)])
    assert [(p["start_sec"], p["chunks"]) for p in periods] == [(0, 2), (600, 2), (1500, 1)]


def test_input_order_does_not_matter_within_a_video():
    forward = retrieve.group_into_periods([chunk(0), chunk(70), chunk(600)])
    shuffled = retrieve.group_into_periods([chunk(600), chunk(0), chunk(70)])
    assert [(p["start_sec"], p["end_sec"]) for p in forward] == [(p["start_sec"], p["end_sec"]) for p in shuffled]


def test_periods_are_sorted_by_playlist_position_then_time():
    chunks = [chunk(900, "late", position=2), chunk(10, "early", position=0), chunk(600, "mid", position=1),
              chunk(800, "early", position=0)]
    periods = retrieve.group_into_periods(chunks)
    assert [(p["video_id"], p["start_sec"]) for p in periods] == [("early", 10), ("early", 800), ("mid", 600), ("late", 900)]


def test_chunks_of_different_videos_are_never_merged():
    periods = retrieve.group_into_periods([chunk(0, "a", 0), chunk(30, "b", 1)])
    assert len(periods) == 2


def test_a_single_chunk_is_a_plain_timestamp_not_a_range():
    [period] = retrieve.group_into_periods([chunk(70)])
    assert not period["is_range"] and period["label"] == period["timestamp"] == "1:10"


def test_period_text_is_time_ordered_and_the_snippet_comes_from_the_best_chunk():
    [period] = retrieve.group_into_periods([
        chunk(70, score=0.9, text="the best passage"), chunk(0, score=0.5, text="an earlier passage"),
    ])
    assert period["text"].index("earlier") < period["text"].index("best")
    assert period["snippet"] == "the best passage" and period["score"] == 0.9


def test_a_custom_gap_overrides_the_config():
    assert len(retrieve.group_into_periods([chunk(0), chunk(100)], gap_seconds=50)) == 2
    assert len(retrieve.group_into_periods([chunk(0), chunk(100)], gap_seconds=500)) == 1


def test_public_view_hides_the_excerpt():
    [period] = retrieve.group_into_periods([chunk(0)])
    assert "text" in period and "text" not in retrieve.public_view(period)


def test_no_chunks_no_periods():
    assert retrieve.group_into_periods([]) == []
