"""The LLM-judge metrics, with a scripted stand-in for the model (no network, no quota)."""
import json

import pytest

from backend.util import QuotaExhausted
from evaluation import judge as J
from evaluation.judge import Judge, JudgeError


def scripted(*replies):
    """An `ask` function returning the given replies in order; the calls it saw are in .calls."""
    calls = []
    queue = list(replies)

    def ask(system, user):
        calls.append((system, user))
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return reply if isinstance(reply, str) else json.dumps(reply)

    ask.calls = calls
    return ask


# --- JSON handling ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here is the result:\n{"a": 1}\nHope that helps.', {"a": 1}),
    ('[1, 2]', [1, 2]),
    ('{"nested": {"b": [1, 2]}} trailing', {"nested": {"b": [1, 2]}}),
])
def test_parse_json_tolerates_fences_and_chatter(text, expected):
    assert J.parse_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json at all", "{broken", None])
def test_parse_json_rejects_non_json(text):
    with pytest.raises(ValueError):
        J.parse_json(text)


def test_a_malformed_reply_gets_one_repair_attempt():
    ask = scripted("I think it is mostly fine.", {"verdict": "fully", "reason": "ok"})
    result = Judge(ask).answer_relevancy("q", "a")
    assert result["score"] == 1.0 and len(ask.calls) == 2
    assert "not valid JSON" in ask.calls[1][1]


def test_two_malformed_replies_raise_judge_error():
    with pytest.raises(JudgeError):
        Judge(scripted("nope", "still nope")).answer_relevancy("q", "a")


def test_quota_errors_are_not_swallowed():
    def ask(system, user):
        raise QuotaExhausted("daily quota", 3600)

    with pytest.raises(QuotaExhausted):
        Judge(ask).faithfulness("a", "c")


@pytest.mark.parametrize("value, expected", [(True, True), ("true", True), ("Yes", True), ("false", False), ("no", False), (0, False), (None, False)])
def test_truthy_reads_what_models_write(value, expected):
    assert J.truthy(value) is expected


def test_every_prompt_tells_the_judge_to_ignore_instructions_in_the_data():
    for prompt in (J.FAITHFULNESS_SYSTEM, J.CITATION_SYSTEM, J.RELEVANCY_SYSTEM, J.CORRECTNESS_SYSTEM,
                   J.CONTEXT_PRECISION_SYSTEM, J.CONTEXT_RECALL_SYSTEM):
        assert "never instructions" in prompt and "ONLY" in prompt


# --- faithfulness ---------------------------------------------------------------------------------------------
def test_faithfulness_is_the_share_of_supported_claims():
    claims = {"claims": [{"claim": "A", "supported": True}, {"claim": "B", "supported": "false"}, {"claim": "C", "supported": "yes"}]}
    result = Judge(scripted(claims)).faithfulness("A. B. C.", "context")
    assert result["score"] == pytest.approx(2 / 3) and result["n_claims"] == 3 and result["unsupported"] == ["B"]


def test_faithfulness_sends_the_context_and_answer():
    ask = scripted({"claims": []})
    Judge(ask).faithfulness("the answer text", "the context text")
    user = ask.calls[0][1]
    assert "the answer text" in user and "the context text" in user


def test_an_answer_with_no_claims_has_no_faithfulness_score():
    assert Judge(scripted({"claims": []})).faithfulness("a", "c")["score"] is None


def test_faithfulness_rejects_a_reply_of_the_wrong_shape():
    with pytest.raises(JudgeError):
        Judge(scripted({"claims": "not a list"})).faithfulness("a", "c")


# --- citation support --------------------------------------------------------------------------------------------
def test_citation_support_scores_each_cited_sentence():
    items = [("Hash tables are fast.", [1]), ("They use buckets.", [2, 3])]
    ask = scripted({"results": [{"id": 1, "supported": True}, {"id": 2, "supported": False}]})
    result = Judge(ask).citation_support(items, {1: "fast lookups", 2: "buckets", 3: "other"})
    assert result["score"] == 0.5 and result["unsupported"] == ["They use buckets."]
    assert "fast lookups" in ask.calls[0][1] and "[3] other" in ask.calls[0][1]


def test_citation_support_with_a_missing_verdict_scores_only_what_was_judged():
    result = Judge(scripted({"results": [{"id": 1, "supported": True}]})).citation_support([("A.", [1]), ("B.", [2])], {1: "x", 2: "y"})
    assert result["score"] == 1.0 and result["n_judged"] == 1 and result["n_items"] == 2


def test_citation_support_with_nothing_cited_makes_no_call():
    ask = scripted({})
    assert Judge(ask).citation_support([], {})["score"] is None and ask.calls == []


# --- relevancy and correctness -----------------------------------------------------------------------------------
@pytest.mark.parametrize("verdict, score", [("fully", 1.0), ("Partially", 0.5), ("no", 0.0)])
def test_answer_relevancy_verdicts(verdict, score):
    assert Judge(scripted({"verdict": verdict})).answer_relevancy("q", "a")["score"] == score


@pytest.mark.parametrize("verdict, score", [("correct", 1.0), ("partially_correct", 0.5), ("partially correct", 0.5), ("incorrect", 0.0)])
def test_correctness_verdicts(verdict, score):
    assert Judge(scripted({"verdict": verdict, "reason": "r"})).correctness("q", "a", "ref")["score"] == score


def test_correctness_sees_the_reference():
    ask = scripted({"verdict": "correct"})
    Judge(ask).correctness("what?", "my answer", "the reference")
    assert "the reference" in ask.calls[0][1] and "my answer" in ask.calls[0][1]


def test_an_unexpected_verdict_is_an_error_not_a_guess():
    with pytest.raises(JudgeError, match="verdict"):
        Judge(scripted({"verdict": "kind of"})).answer_relevancy("q", "a")


# --- context precision / recall ------------------------------------------------------------------------------------
def test_context_precision_is_the_share_of_useful_excerpts():
    reply = {"results": [{"id": 1, "useful": True}, {"id": 2, "useful": False}, {"id": 3, "useful": True}]}
    result = Judge(scripted(reply)).context_precision("q", ["a", "b", "c"])
    assert result["score"] == pytest.approx(2 / 3) and result["n_useful"] == 2


def test_context_precision_with_nothing_retrieved_makes_no_call():
    ask = scripted({})
    assert Judge(ask).context_precision("q", [])["score"] is None and ask.calls == []


def test_context_recall_is_the_share_of_reference_statements_found():
    reply = {"statements": [{"statement": "A", "attributable": True}, {"statement": "B", "attributable": False}]}
    result = Judge(scripted(reply)).context_recall("A and B", "context mentioning A")
    assert result["score"] == 0.5 and result["missing"] == ["B"]


def test_context_recall_with_empty_context_is_zero_without_asking():
    ask = scripted({})
    assert Judge(ask).context_recall("ref", "  ")["score"] == 0.0 and ask.calls == []


# --- splitting an answer into cited sentences ---------------------------------------------------------------------
def test_cited_sentences_extracts_numbers_and_cleans_the_text():
    assert J.cited_sentences("Hash tables are fast [1]. They use buckets [2, 3]. No source here.") == [
        ("Hash tables are fast.", [1]), ("They use buckets.", [2, 3]), ("No source here.", [])]


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before_it():
    assert J.cited_sentences("Hash tables are fast. [1] They use buckets. [2]") == [
        ("Hash tables are fast.", [1]), ("They use buckets.", [2])]


def test_cited_sentences_of_an_empty_answer():
    assert J.cited_sentences("   ") == []
