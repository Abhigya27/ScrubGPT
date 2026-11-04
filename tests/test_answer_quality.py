"""answer_quality_eval: per-question judging, aggregation, the audit sample and the command, with a scripted judge."""
import json

import pytest

from backend import util
from backend.util import QuotaExhausted
from evaluation import answer_quality_eval as AQ
from evaluation import judge as J
from evaluation.judge import Judge


def router(overrides=None, default_ok=True):
    """An `ask` that answers each judge prompt according to which prompt it is; `overrides` maps a system-prompt constant
    to a reply (dict, str, or callable). Counts calls per prompt in .counts."""
    replies = {
        J.FAITHFULNESS_SYSTEM: {"claims": [{"claim": "c1", "supported": True}, {"claim": "c2", "supported": True}]},
        J.CITATION_SYSTEM: {"results": [{"id": 1, "supported": True}, {"id": 2, "supported": True}]},
        J.RELEVANCY_SYSTEM: {"verdict": "fully", "reason": "answers it"},
        J.CORRECTNESS_SYSTEM: {"verdict": "correct", "reason": "matches"},
        J.CONTEXT_PRECISION_SYSTEM: {"results": [{"id": 1, "useful": True}, {"id": 2, "useful": True}]},
        J.CONTEXT_RECALL_SYSTEM: {"statements": [{"statement": "s", "attributable": True}]},
        **(overrides or {}),
    }
    names = {J.FAITHFULNESS_SYSTEM: "faithfulness", J.CITATION_SYSTEM: "citation", J.RELEVANCY_SYSTEM: "relevancy",
             J.CORRECTNESS_SYSTEM: "correctness", J.CONTEXT_PRECISION_SYSTEM: "precision", J.CONTEXT_RECALL_SYSTEM: "recall"}
    counts = {}

    def ask(system, user):
        key = names[system]
        counts[key] = counts.get(key, 0) + 1
        reply = replies[system]
        reply = reply(user) if callable(reply) else reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    ask.counts = counts
    return ask


PERIODS = [
    {"video_title": "V", "label": "0:00 - 1:30", "text": "hash tables are fast", "video_id": "v", "position": 0},
    {"video_title": "V", "label": "5:00", "text": "they use buckets", "video_id": "v", "position": 0},
]
OUTPUT = {"answer": "Hash tables are fast [1]. They use buckets [2].", "grounded": True,
          "citations": [{"n": 1, "text": "hash tables are fast"}, {"n": 2, "text": "they use buckets"}]}


def entry(**kw):
    base = {"id": "e1", "category": "direct_fact", "question": "How do hash tables work?", "gold_answer": "They are fast and use buckets."}
    base.update(kw)
    return base


def score(ask=None, e=None, output=OUTPUT, periods=PERIODS, metrics=AQ.ALL_METRICS, refused=False):
    return AQ.score_question(Judge(ask or router()), e or entry(), output, periods, metrics, refused)


def test_a_good_answer_scores_one_on_every_metric():
    result = score()
    assert result["scores"] == {m: 1.0 for m in AQ.ALL_METRICS} and result["judge_errors"] == []
    assert set(result["details"]) == set(AQ.ALL_METRICS)


def test_the_judge_sees_the_excerpts_the_answer_was_built_from():
    seen = []
    ask = router({J.FAITHFULNESS_SYSTEM: lambda user: seen.append(user) or {"claims": []}})
    score(ask, metrics=("faithfulness",))
    assert "hash tables are fast" in seen[0] and "they use buckets" in seen[0] and "[2]" in seen[0]


def test_unsupported_claims_lower_faithfulness():
    ask = router({J.FAITHFULNESS_SYSTEM: {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}})
    result = score(ask)
    assert result["scores"]["faithfulness"] == 0.5 and result["details"]["faithfulness"]["unsupported"] == ["b"]


def test_a_refusal_skips_the_answer_metrics_but_still_scores_the_retrieved_context():
    ask = router()
    result = score(ask, refused=True)
    assert set(result["scores"]) == {"context_precision", "context_recall"}
    assert not {"faithfulness", "citation", "relevancy", "correctness"} & set(ask.counts)


def test_without_a_reference_answer_correctness_and_context_recall_are_not_scored():
    result = score(e=entry(gold_answer=None))
    assert set(result["scores"]) == {"faithfulness", "citation_support", "answer_relevancy", "context_precision"}


def test_nothing_retrieved_means_zero_context_recall_and_no_precision():
    result = score(output={"answer": "", "grounded": False, "citations": []}, periods=[], refused=True)
    assert result["scores"] == {"context_recall": 0.0}


def test_an_answer_with_no_citations_has_no_citation_score():
    result = score(output={**OUTPUT, "answer": "Hash tables are fast and use buckets.", "citations": []})
    assert "citation_support" not in result["scores"]


def test_a_judge_failure_is_recorded_and_the_other_metrics_still_run():
    result = score(router({J.RELEVANCY_SYSTEM: "I cannot decide."}))
    assert result["scores"]["answer_relevancy"] is None and result["judge_errors"] == ["answer_relevancy"]
    assert "error" in result["details"]["answer_relevancy"]
    assert result["scores"]["faithfulness"] == 1.0 and result["scores"]["correctness"] == 1.0


def test_only_the_requested_metrics_are_judged():
    ask = router()
    result = score(ask, metrics=("answer_relevancy",))
    assert list(result["scores"]) == ["answer_relevancy"] and ask.counts == {"relevancy": 1}


def test_quota_exhaustion_propagates_out_of_scoring():
    def ask(system, user):
        raise QuotaExhausted("no quota", 60)

    with pytest.raises(QuotaExhausted):
        score(ask)


def test_parse_metrics():
    assert AQ.parse_metrics(None) == AQ.ALL_METRICS
    assert AQ.parse_metrics("faithfulness, answer_relevancy") == ("faithfulness", "answer_relevancy")
    with pytest.raises(ValueError, match="unknown metric"):
        AQ.parse_metrics("faithfulness,vibes")


def test_citation_coverage():
    assert AQ.citation_coverage("This sentence is cited properly [1]. This one has no citation at all.") == 0.5
    assert AQ.citation_coverage("Sure. Yes.") is None   # too short to be expected to cite
    assert AQ.citation_coverage("Every substantive sentence here is cited [1].") == 1.0


def test_build_context_numbers_the_excerpts_like_the_answer_prompt():
    text = AQ.build_context(PERIODS)
    assert text.startswith('[1] "V" (0:00 - 1:30)\nhash tables are fast') and '[2] "V" (5:00)' in text


def row(id, category, scores, refused=False, errors=(), coverage=None):
    return {"id": id, "category": category, "question": f"q {id}", "refused": refused, "answer": "" if refused else "an answer",
            "scores": scores, "details": {k: {} for k in scores}, "judge_errors": list(errors), "citation_coverage": coverage}


def test_summary_means_intervals_and_rates():
    rows = [
        row("a", "direct_fact", {"faithfulness": 1.0, "answer_relevancy": 1.0}, coverage=1.0),
        row("b", "direct_fact", {"faithfulness": 0.5, "answer_relevancy": 0.5}, coverage=0.5),
        row("c", "paraphrase", {"faithfulness": 0.0, "answer_relevancy": None}, errors=["answer_relevancy"]),
        row("d", "paraphrase", {}, refused=True),
    ]
    s = AQ.summarize(rows, ("faithfulness", "answer_relevancy"))
    assert s["n_questions"] == 4 and s["answered_rate"] == 0.75 and s["refused_answerable"] == 1 and s["judge_errors"] == 1
    assert s["citation_coverage"] == 0.75
    f = s["metrics"]["faithfulness"]
    assert f["n"] == 3 and f["mean"] == 0.5 and f["fraction_perfect"] == pytest.approx(1 / 3)
    assert f["n_not_applicable"] == 1 and f["ci95"][0] <= 0.5 <= f["ci95"][1]
    assert f["by_category"] == {"direct_fact": {"n": 2, "mean": 0.75}, "paraphrase": {"n": 1, "mean": 0.0}}
    r = s["metrics"]["answer_relevancy"]
    assert r["n"] == 2 and r["n_judge_failed"] == 1 and r["mean"] == 0.75


def test_an_unscored_metric_summarises_as_none():
    s = AQ.summarize([row("a", "direct_fact", {})], ("correctness",))
    assert s["metrics"]["correctness"]["mean"] is None and s["metrics"]["correctness"]["n"] == 0


def test_audit_file_lists_questions_answers_and_judge_reasons(tmp_path):
    rows = [row(f"q{i}", "direct_fact", {"answer_relevancy": 1.0}) for i in range(5)]
    rows[0]["details"]["answer_relevancy"] = {"reason": "directly answers"}
    path = AQ.write_audit(rows, 3, 0, tmp_path / "audit.md")
    text = path.read_text(encoding="utf-8")
    assert text.count("## q") == 3 and "**Question:**" in text and "| answer_relevancy | 1.00 |" in text
    assert AQ.write_audit(rows, 3, 0, tmp_path / "again.md").read_text(encoding="utf-8") == text  # seeded: repeatable
    assert AQ.write_audit(rows, 99, 0, tmp_path / "all.md").read_text(encoding="utf-8").count("## q") == 5


# --- the command -------------------------------------------------------------------------------------------------
def eval_set(tmp_path):
    entries = [
        {"id": "a", "source_id": "PL", "question": "How do hash tables work?", "gold_answer_contains": "hash", "gold_answer": "They are fast."},
        {"id": "b", "source_id": "PL", "question": "Something uncovered?", "expects_refusal": True},
        {"id": "c", "source_id": "PL", "question": "And buckets?", "gold_answer_contains": "buckets"},
    ]
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


@pytest.fixture
def fake_pipeline(monkeypatch):
    from backend.rag import answer

    monkeypatch.setattr(AQ, "run_search", lambda e, mode: {"standalone": e["question"], "periods": PERIODS, "prior": []})
    monkeypatch.setattr(answer, "generate_answer", lambda q, periods, prior: OUTPUT)
    return answer


def judge_chat(monkeypatch, ask):
    """Replace the LLM itself (not _chat), so the app's real call path and the metering both run."""
    from types import SimpleNamespace

    class FakeModel:
        def invoke(self, messages):
            (_, system), (_, user) = messages
            return SimpleNamespace(content=ask(system, user), usage_metadata={"input_tokens": 10, "output_tokens": 2})

    monkeypatch.setattr(util, "get_llm", lambda role="default": FakeModel())


def test_command_scores_answerable_questions_and_writes_results(tmp_path, monkeypatch, fake_pipeline):
    judge_chat(monkeypatch, router())
    out = tmp_path / "r"
    assert AQ.main(["--eval-set", eval_set(tmp_path), "--out-dir", str(out), "--audit", "2"]) == 0
    [result] = list(out.glob("answer_quality_*.json"))
    data = json.loads(result.read_text(encoding="utf-8"))
    assert [q["id"] for q in data["questions"]] == ["a", "c"]          # the unanswerable question is the grounding eval's job
    assert data["summary"]["metrics"]["faithfulness"]["mean"] == 1.0
    assert data["summary"]["metrics"]["correctness"]["n"] == 1         # only question "a" has a reference answer
    assert data["params"]["judge_is_answer_model"] is True             # no separate judge configured in the test
    assert data["questions"][0]["judge_calls"] == 6 and data["questions"][1]["judge_calls"] == 4
    assert list(out.glob("judge_audit_*.md"))
    assert data["llm_usage"]["judge"]["calls"] == 10 and data["llm_usage"]["judge"]["input_tokens"] == 100


def test_command_stops_cleanly_when_the_judge_runs_out_of_quota(tmp_path, monkeypatch, fake_pipeline):
    calls = []

    def ask(system, user):
        calls.append(1)
        if len(calls) > 3:
            raise QuotaExhausted("daily quota", 3600)
        return router()(system, user)

    judge_chat(monkeypatch, ask)
    out = tmp_path / "r"
    assert AQ.main(["--eval-set", eval_set(tmp_path), "--out-dir", str(out), "--metrics", "faithfulness,answer_relevancy"]) == AQ.QUOTA_EXIT_CODE
    data = json.loads(next(out.glob("answer_quality_*.json")).read_text(encoding="utf-8"))
    assert "quota" in data["aborted"]


def test_command_rejects_unknown_metrics(tmp_path):
    assert AQ.main(["--eval-set", eval_set(tmp_path), "--metrics", "vibes", "--out-dir", str(tmp_path)]) != 0
