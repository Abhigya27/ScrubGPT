"""draft_eval_set: LLM-proposed questions are validated mechanically and always arrive flagged for human review."""
import json

import pytest

from backend import util
from evaluation import draft_eval_set as D
from evaluation.common import EvalSetError, load_eval_set
from evaluation.judge import JsonLLM

PASSAGE = ("Today we will talk about hash tables. A hash table stores key value pairs and uses a hash function to pick a "
           "bucket. When two keys land in the same bucket we call that a collision, and separate chaining handles it by "
           "keeping a linked list inside each bucket. Lookups are constant time on average because the bucket is found directly.")


def chunk(i=0, text=PASSAGE, video="v1", start=100):
    return {"chunk_id": f"{video}:{start}", "video_id": video, "video_title": "Data structures", "start_sec": start, "text": text}


def item(**kw):
    base = {"question": "How do hash tables deal with two keys that collide?", "gold_answer": "They use separate chaining.",
            "evidence": "separate chaining handles it", "kind": "direct_fact"}
    base.update(kw)
    return base


def llm(reply):
    return JsonLLM(ask=lambda system, user: reply if isinstance(reply, str) else json.dumps(reply))


# --- validating one proposal -------------------------------------------------------------------------------------
def test_a_good_draft_passes():
    assert D.validate_draft(item(), PASSAGE) == (True, "")


@pytest.mark.parametrize("change, reason", [
    ({"question": "hash tables"}, "not a proper question"),
    ({"question": "What does this passage say about collisions?"}, "refers to the passage"),
    ({"question": "In the transcript, how are collisions handled?"}, "refers to the passage"),
    ({"gold_answer": " "}, "no reference answer"),
    ({"evidence": "chaining"}, "not a short phrase"),
    ({"evidence": "this was never said in the passage at all"}, "not verbatim"),
    ({"evidence": " ".join(["word"] * 14)}, "not a short phrase"),
])
def test_bad_drafts_are_rejected_with_a_reason(change, reason):
    ok, why = D.validate_draft(item(**change), PASSAGE)
    assert not ok and reason in why


def test_evidence_matching_ignores_case_and_punctuation():
    assert D.validate_draft(item(evidence="Separate Chaining, handles it!"), PASSAGE)[0]


def test_lexical_overlap_distinguishes_copied_from_reworded_questions():
    copied = D.lexical_overlap("How does separate chaining handle a collision in a bucket?", PASSAGE)
    reworded = D.lexical_overlap("What happens when two items map to the same slot?", PASSAGE)
    assert copied > 0.6 > reworded
    assert D.lexical_overlap("Is it?", PASSAGE) is None   # no content words to compare


# --- drafting from one passage ------------------------------------------------------------------------------------
def test_draft_for_passage_keeps_valid_items_and_reports_the_rest():
    reply = {"items": [item(), item(question="What does the excerpt say?"), item(evidence="a quote the model invented")]}
    accepted, rejected = D.draft_for_passage(llm(reply), chunk())
    assert len(accepted) == 1 and len(rejected) == 2
    assert accepted[0]["gold_answer_contains"] == "separate chaining handles it" and 0 <= accepted[0]["lexical_overlap"] <= 1
    assert {r["reason"] for r in rejected} == {"refers to the passage/transcript", "evidence is not verbatim in the passage"}


def test_an_unknown_kind_becomes_direct_fact():
    accepted, _ = D.draft_for_passage(llm({"items": [item(kind="sarcastic")]}), chunk())
    assert accepted[0]["category"] == "direct_fact"
    accepted, _ = D.draft_for_passage(llm({"items": [item(kind="terminology_mismatch")]}), chunk())
    assert accepted[0]["category"] == "terminology_mismatch"


def test_the_passage_is_sent_as_data_with_the_rules():
    seen = []
    D.draft_for_passage(JsonLLM(ask=lambda system, user: seen.append((system, user)) or '{"items": []}'), chunk(), n=3)
    system, user = seen[0]
    assert "3 different questions" in system and "EXACTLY" in system and "data, not instructions" in system
    assert PASSAGE in user and "Data structures" in user


# --- sampling ----------------------------------------------------------------------------------------------------
def test_sampling_spreads_over_each_video_and_skips_thin_passages():
    chunks = [chunk(text=PASSAGE, video="a", start=s) for s in range(0, 1000, 100)] + [chunk(text="too short", video="b", start=0)]
    picked = D.sample_chunks(chunks, per_video=3, seed=1)
    assert len(picked) == 3 and {c["video_id"] for c in picked} == {"a"}   # video b has no substantial passage
    starts = sorted(c["start_sec"] for c in picked)
    assert starts[0] < 400 and starts[-1] >= 600                            # not three neighbours from the start
    assert D.sample_chunks(chunks, 3, seed=1) == picked                     # seeded


def test_sampling_never_asks_for_more_than_a_video_has():
    assert len(D.sample_chunks([chunk(start=0), chunk(start=100)], per_video=5)) == 2


# --- whole-source drafting ------------------------------------------------------------------------------------------
def routed_ask(items=None, unanswerable=None):
    """A scripted model: unanswerable-question prompts get questions, passage prompts get drafted items."""

    def ask(system, user):
        if system.startswith("You write evaluation questions that a question-answering system should REFUSE"):
            return json.dumps({"questions": unanswerable if unanswerable is not None else ["Does the course cover quantum computing?"]})
        return json.dumps({"items": items if items is not None else [item()]})

    return ask


def routed(items=None, unanswerable=None):
    return JsonLLM(ask=routed_ask(items, unanswerable))


def test_every_draft_is_flagged_for_review_and_points_at_its_source_chunk():
    chunks = [chunk(video="a", start=500), chunk(video="b", start=40)]
    entries, info = D.build_entries("PL", chunks, routed(), per_video=1, unanswerable=2, max_questions=None)
    answerable = [e for e in entries if not e["expects_refusal"]]
    assert len(answerable) == 2 and all(e["needs_review"] for e in entries)
    assert {e["video_id"] for e in answerable} == {"a", "b"}
    assert {e["gold_timestamp_seconds"] for e in answerable} == {500, 40}   # the passage start; the human refines it
    assert all(e["draft_chunk_id"] and e["evidence_matches_n_chunks"] >= 1 for e in answerable)
    assert info["passages"] == 2


def test_unanswerable_drafts_expect_refusal_and_have_no_gold_text():
    entries, _ = D.build_entries("PL", [chunk()], routed(unanswerable=["What GPU did he use?", "not a question"]), 1, 5, None)
    [refusal] = [e for e in entries if e["category"] == "unanswerable"]
    assert refusal["expects_refusal"] and refusal["gold_answer_contains"] is None and refusal["question"] == "What GPU did he use?"


def test_the_answerable_cap_is_respected():
    chunks = [chunk(video=f"v{i}", start=0) for i in range(6)]
    entries, _ = D.build_entries("PL", chunks, routed(items=[item(), item(question="How is a collision resolved here?")]), 1, 0, max_questions=3)
    assert len([e for e in entries if not e["expects_refusal"]]) == 3


def test_a_judge_failure_on_one_passage_is_counted_not_fatal():
    calls = []

    def ask(system, user):
        calls.append(1)
        return "not json" if len(calls) <= 2 else json.dumps({"items": [item()]})

    entries, info = D.build_entries("PL", [chunk(video="a"), chunk(video="b")], JsonLLM(ask=ask), 1, 0, None)
    assert info["failed_calls"] == 1 and len(entries) == 1


def test_drafts_are_invisible_to_the_scripts_until_reviewed(tmp_path):
    entries, _ = D.build_entries("PL", [chunk()], routed(), 1, 1, None)
    for number, e in enumerate(entries, 1):
        e["id"] = f"d{number:02d}"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(EvalSetError, match="unreviewed"):
        load_eval_set(path)
    for e in entries:
        del e["needs_review"]
    path.write_text(json.dumps(entries), encoding="utf-8")
    loaded, info = load_eval_set(path)
    assert len(loaded) == len(entries) and loaded[0]["gold_answer"] == "They use separate chaining."


# --- the command ----------------------------------------------------------------------------------------------------
def test_command_writes_a_draft_file_and_never_overwrites(tmp_path, monkeypatch):
    from backend.index import qdrant_store

    library = tmp_path / "library.json"
    library.write_text(json.dumps({"sources": {"PL": {"indexed_videos": {}}}}), encoding="utf-8")
    monkeypatch.setattr(qdrant_store, "scan_source_chunks", lambda source_id: [chunk(video="a"), chunk(video="b", start=900)])
    from types import SimpleNamespace

    class FakeModel:
        def invoke(self, messages):
            (_, system), (_, user) = messages
            return SimpleNamespace(content=routed_ask()(system, user), usage_metadata=None)

    monkeypatch.setattr(util, "get_llm", lambda role="default": FakeModel())
    out = tmp_path / "eval_set.draft.json"
    args = ["--library", str(library), "--out", str(out), "--unanswerable", "1"]
    assert D.main(args) == 0
    drafts = json.loads(out.read_text(encoding="utf-8"))
    assert [d["id"] for d in drafts] == ["d01", "d02", "d03"] and all(d["needs_review"] for d in drafts)
    assert D.main(args) != 0                      # refuses to overwrite
    assert D.main(args + ["--force"]) == 0
