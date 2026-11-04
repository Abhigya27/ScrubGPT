"""robustness_eval: the perturbations and how retrieval is compared across them."""
import json

import pytest

from backend import config
from backend.index import qdrant_store
from evaluation import ablation_eval, robustness_eval as R

Q = "How does the speaker explain hash tables and collisions?"


def test_clean_is_unchanged_and_unknown_kinds_are_rejected():
    assert R.perturb(Q, "clean") == Q
    with pytest.raises(ValueError, match="unknown perturbation"):
        R.perturb(Q, "sarcasm")


@pytest.mark.parametrize("kind", R.KINDS)
def test_perturbations_are_deterministic(kind):
    assert R.perturb(Q, kind, seed=3) == R.perturb(Q, kind, seed=3)


def test_typos_swap_neighbouring_letters_in_long_words_only():
    questions = [f"{Q} Please also explain how {w} work" for w in ("arrays", "linked lists", "binary trees")]
    changed_any = False
    for question in questions:
        altered = R.perturb(question, "typos", seed=1)
        for before, after in zip(question.split(), altered.split()):
            if before != after:
                changed_any = True
                assert len(before.rstrip("?.,")) >= 4 and sorted(before) == sorted(after)  # a transposition, nothing else
            if len(before.rstrip("?.,")) < 4:
                assert before == after
        assert len(altered.split()) == len(question.split())
    assert changed_any


def test_a_different_seed_gives_different_typos():
    seeds = {R.perturb(Q * 3, "typos", seed=s) for s in range(6)}
    assert len(seeds) > 1


def test_typos_keep_trailing_punctuation():
    assert R.perturb("Explain hashing?", "typos", seed=0).endswith("?")


def test_uppercase_filler_distractor_and_truncation():
    assert R.perturb(Q, "uppercase") == Q.upper()
    filler = R.perturb(Q, "filler")
    assert filler.startswith("hey so um") and "hash tables and collisions" in filler and filler.endswith("thanks!!")
    assert R.perturb(Q, "distractor").startswith(Q) and "bananas" in R.perturb(Q, "distractor")
    truncated = R.perturb(Q, "truncated")
    assert Q.startswith(truncated) and 2 <= len(truncated.split()) < len(Q.split())
    assert R.perturb("Why?", "truncated") == "Why?"   # too short to cut


# --- comparing retrieval across the variants -----------------------------------------------------------------------
HALF = config.SPARSE_HALF_SATURATION_TERMS * qdrant_store._TF_TYPICAL * __import__("math").log(50)


def hit(pid, score, text):
    from types import SimpleNamespace

    return SimpleNamespace(id=pid, score=score, payload={"chunk_id": pid, "video_id": "v", "video_title": "T", "start_sec": 0,
                                                         "end_sec": 90, "text": text, "positions": {"PL": 0}})


def fetch(entry):
    """The relevant chunk is only found when the question still contains the word 'hash'."""
    if "hash" in entry["question"].lower():
        return [hit("rel", 0.9, "a hash table")], [hit("rel", HALF * 5, "a hash table")], 50
    return [hit("other", 0.9, "unrelated")], [], 50


def rank_fn(dense, sparse, n, entry, cfg):
    return qdrant_store.rank_candidates(dense, sparse, n, entry["source_id"], cfg.sparse, cfg.dense)


ENTRIES = [{"id": f"e{i}", "source_id": "PL", "question": f"Could you please tell me about the hash table number {i}", "category": "direct_fact",
            "gold_answer_contains": "hash table"} for i in range(4)]


def test_run_robustness_reports_each_variant():
    configs = ablation_eval.parse_configs("0:1,1:0", 0.65, 0.35)
    results = R.run_robustness(ENTRIES, ("uppercase", "truncated"), fetch, rank_fn, configs)
    assert set(results) == {"clean", "uppercase", "truncated"}
    assert results["clean"]["summary"]["dense-only"]["recall_at_k"] == 1.0
    assert results["uppercase"]["summary"]["dense-only"]["recall_at_k"] == 1.0   # 'HASH'.lower() still matches
    assert results["truncated"]["summary"]["dense-only"]["recall_at_k"] == 0.0   # "What is a" lost the topic word


def test_degradation_is_perturbed_minus_clean():
    configs = ablation_eval.parse_configs("0:1", 0.65, 0.35)
    changes = R.degradation(R.run_robustness(ENTRIES, ("uppercase", "truncated"), fetch, rank_fn, configs), configs)
    assert changes["uppercase"]["dense-only"]["recall_change"] == 0.0
    assert changes["truncated"]["dense-only"]["recall_change"] == -1.0
    assert changes["truncated"]["dense-only"]["mrr_change"] == -1.0


def test_the_questions_are_not_mutated():
    before = json.dumps(ENTRIES)
    R.run_robustness(ENTRIES, ("typos",), fetch, rank_fn, ablation_eval.parse_configs("0:1", 0.65, 0.35))
    assert json.dumps(ENTRIES) == before


def test_command_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation_eval, "backend_io", lambda enhance, no_production=False: (fetch, rank_fn, None))
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([{**e, "gold_answer_contains": "hash table"} for e in ENTRIES] + [{"id": "r", "source_id": "PL", "question": "?", "expects_refusal": True}]), encoding="utf-8")
    out = tmp_path / "r"
    assert R.main(["--eval-set", str(path), "--out-dir", str(out), "--perturbations", "uppercase,truncated"]) == 0
    data = json.loads(next(out.glob("robustness_*.json")).read_text(encoding="utf-8"))
    assert data["changes"]["truncated"]["dense-only"]["recall_change"] == -1.0
    assert data["left_out"]["expects_refusal"] == 1 and set(data["results"]) == {"clean", "uppercase", "truncated"}


def test_command_rejects_an_unknown_perturbation(tmp_path):
    assert R.main(["--eval-set", str(tmp_path / "x.json"), "--perturbations", "sarcasm"]) != 0
