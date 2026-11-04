"""Precision@k, nDCG, true chunk recall, percentiles and bootstrap intervals, plus the backend pieces they rely on."""
import json
import math
from types import SimpleNamespace

import pytest

from backend import config, util
from backend.index import qdrant_store
from evaluation import common
from evaluation.common import EvalSetError, load_eval_set

T, F = True, False


def test_precision_at_k():
    assert common.precision_at_k([T, F, T, F, F], 5) == 0.4
    assert common.precision_at_k([T, F, T, F, F], 3) == pytest.approx(2 / 3)
    assert common.precision_at_k([T], 5) == 0.2  # a short list is penalised: the denominator is k
    assert common.precision_at_k([], 5) == 0.0
    assert common.precision_at_k([T], 0) is None


def test_ndcg_rewards_earlier_hits():
    first = common.ndcg_at_k([T, F, F], 3)
    second = common.ndcg_at_k([F, T, F], 3)
    third = common.ndcg_at_k([F, F, T], 3)
    assert first == 1.0 and second == pytest.approx(1 / math.log2(3)) and third == pytest.approx(0.5)
    assert first > second > third


def test_ndcg_with_a_known_number_of_relevant_chunks():
    dcg = 1 + 1 / math.log2(4)                                # relevant at ranks 1 and 3
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)            # three relevant chunks exist, so ranks 1-3 would all be hits
    assert common.ndcg_at_k([T, F, T], 3, total_relevant=3) == pytest.approx(dcg / ideal)


def test_ndcg_edge_cases():
    assert common.ndcg_at_k([F, F, F], 3) == 0.0                # relevant chunks may exist but none were retrieved
    assert common.ndcg_at_k([T, F, F], 3, total_relevant=0) is None   # nothing could be relevant: undefined
    assert common.ndcg_at_k([T, T, T, T], 2, total_relevant=10) == 1.0  # a perfect top k, however many exist


def test_chunk_recall_is_the_share_of_all_relevant_chunks_found():
    assert common.chunk_recall_at_k([T, T, F], 3, 4) == 0.5
    assert common.chunk_recall_at_k([T, T, F], 2, 4) == 0.5
    assert common.chunk_recall_at_k([T, F, F], 3, 1) == 1.0
    assert common.chunk_recall_at_k([T, T, T], 3, 2) == 1.0     # never above 1, even if the list holds duplicates
    assert common.chunk_recall_at_k([T], 3, None) is None and common.chunk_recall_at_k([T], 3, 0) is None


def test_hit_rate_and_chunk_recall_differ():
    """Recall@k as the spec defines it is satisfied by ONE relevant chunk; true recall is not."""
    flags, total = [T, F, F, F, F], 4
    assert common.recall_at_k([1], 5) == 1.0
    assert common.chunk_recall_at_k(flags, 5, total) == 0.25


def test_mean_skips_missing_values():
    assert common.mean([1.0, None, 3.0]) == 2.0 and common.mean([None]) is None and common.mean([]) is None


def test_percentile_interpolates():
    values = [1, 2, 3, 4]
    assert common.percentile(values, 0) == 1 and common.percentile(values, 100) == 4
    assert common.percentile(values, 50) == 2.5 and common.percentile(values, 95) == pytest.approx(3.85)
    assert common.percentile([7], 95) == 7 and common.percentile([], 50) is None
    assert common.percentile([4, 1, 3, 2], 50) == 2.5  # input order does not matter


def test_bootstrap_interval_is_seeded_and_brackets_the_mean():
    values = [0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    low, high = common.bootstrap_ci(values)
    assert (low, high) == common.bootstrap_ci(values)           # same seed, same answer
    assert low <= common.mean(values) <= high and 0.0 <= low and high <= 1.0
    assert common.bootstrap_ci([0.7, 0.7, 0.7]) == (pytest.approx(0.7), pytest.approx(0.7))
    assert common.bootstrap_ci([0.5]) is None and common.bootstrap_ci([None, 0.5]) is None


def test_more_data_gives_a_tighter_interval():
    small = common.bootstrap_ci([0.0, 1.0] * 4)
    large = common.bootstrap_ci([0.0, 1.0] * 40)
    assert (large[1] - large[0]) < (small[1] - small[0])


# --- eval-set loader additions ------------------------------------------------------------------------------------
def write(tmp_path, entries):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def entry(**kw):
    base = {"id": "e1", "source_id": "PL", "question": "What is X?", "gold_answer_contains": "x is"}
    base.update(kw)
    return base


def test_gold_answer_is_loaded_and_blank_means_none(tmp_path):
    entries, _ = load_eval_set(write(tmp_path, [entry(gold_answer="  X is a thing. "), entry(id="e2", gold_answer="   "), entry(id="e3")]))
    assert [e["gold_answer"] for e in entries] == ["X is a thing.", None, None]


def test_gold_answer_must_be_text(tmp_path):
    with pytest.raises(EvalSetError, match="gold_answer"):
        load_eval_set(write(tmp_path, [entry(gold_answer=["a list"])]))


def test_unreviewed_drafts_are_ignored_and_counted(tmp_path):
    entries, info = load_eval_set(write(tmp_path, [entry(), entry(id="d1", needs_review=True), entry(id="d2", needs_review=True)]))
    assert [e["id"] for e in entries] == ["e1"] and info["skipped_unreviewed"] == 2


def test_a_file_of_only_drafts_says_to_review_them(tmp_path):
    with pytest.raises(EvalSetError, match="1 unreviewed draft"):
        load_eval_set(write(tmp_path, [entry(needs_review=True)]))


def test_a_reviewed_draft_is_used_once_the_flag_is_removed(tmp_path):
    draft = entry(needs_review=True)
    assert load_eval_set(write(tmp_path, [entry(id="e0"), draft]))[1]["skipped_unreviewed"] == 1
    del draft["needs_review"]
    entries, info = load_eval_set(write(tmp_path, [entry(id="e0"), draft]))
    assert len(entries) == 2 and info["skipped_unreviewed"] == 0


def test_results_files_are_tracked_for_run_all(tmp_path):
    before = len(common.WRITTEN)
    path = common.write_results("demo", {}, tmp_path)
    assert common.WRITTEN[before:] == [path]


# --- the judge LLM role ------------------------------------------------------------------------------------------
def test_judge_role_uses_its_own_provider_and_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "gemini")
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-model")
    assert util._settings("judge") == ("gemini", "judge-model")


def test_judge_role_falls_back_to_the_answer_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", None)
    monkeypatch.setattr(config, "JUDGE_MODEL", None)
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    assert util._settings("judge") == util._settings("default") == ("groq", config.GROQ_MODEL)


def test_judge_provider_without_a_model_uses_that_providers_default(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "groq")
    monkeypatch.setattr(config, "JUDGE_MODEL", None)
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    assert util._settings("judge") == ("groq", config.GROQ_MODEL)


def test_other_roles_are_unchanged(monkeypatch):
    assert util._settings("translate") == (config.TRANSLATE_PROVIDER, config.TRANSLATE_MODEL)
    assert util._settings("enhance") == (config.ENHANCE_PROVIDER, config.ENHANCE_MODEL)


# --- corpus scan -----------------------------------------------------------------------------------------------
class FakeClient:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.calls.append({"offset": offset, "with_vectors": with_vectors, "limit": limit})
        index = 0 if offset is None else offset
        points = self.pages[index]
        return points, (index + 1 if index + 1 < len(self.pages) else None)


def point(chunk_id, text, positions=None):
    return SimpleNamespace(id=chunk_id, payload={"chunk_id": chunk_id, "video_id": "v", "text": text, "positions": positions or {"PL": 2}})


def test_scan_reads_every_page_without_vectors(monkeypatch):
    client = FakeClient([[point("a", "one"), point("b", "two")], [point("c", "three")]])
    monkeypatch.setattr(qdrant_store, "ensure_ready", lambda: None)
    monkeypatch.setattr(qdrant_store, "collection_name", lambda: "test")  # the real one loads the embedding model
    monkeypatch.setattr(qdrant_store, "_client", lambda: client)
    chunks = qdrant_store.scan_source_chunks("PL")
    assert [c["chunk_id"] for c in chunks] == ["a", "b", "c"]
    assert all(c["position"] == 2 for c in chunks)
    assert len(client.calls) == 2 and not any(call["with_vectors"] for call in client.calls)


def test_scan_of_an_empty_source(monkeypatch):
    monkeypatch.setattr(qdrant_store, "ensure_ready", lambda: None)
    monkeypatch.setattr(qdrant_store, "collection_name", lambda: "test")
    monkeypatch.setattr(qdrant_store, "_client", lambda: FakeClient([[]]))
    assert qdrant_store.scan_source_chunks("PL") == []
