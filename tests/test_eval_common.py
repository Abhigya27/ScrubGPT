"""Shared evaluation helpers: eval-set loading and validation, relevance matching, metric formulas, result files."""
import json

import pytest

from evaluation import common
from evaluation.common import EvalSetError, load_eval_set

VID = "dQw4w9WgXcQ"


def write(tmp_path, entries, name="eval.json"):
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def entry(**kwargs):
    base = {"id": "e1", "source_id": VID, "question": "What is a hashmap?", "category": "direct_fact",
            "gold_answer_contains": "hash table", "gold_timestamp_seconds": 342, "expects_refusal": False}
    base.update(kwargs)
    return base


# --- loading ----------------------------------------------------------------------------------------------------
def test_a_filled_entry_loads_with_defaults(tmp_path):
    [e], info = load_eval_set(write(tmp_path, [entry(gold_timestamp_seconds=None)]))
    assert e["video_id"] == VID  # a single-video source id doubles as the video id
    assert e["history"] == [] and e["expects_refusal"] is False and e["gold_timestamp_seconds"] is None
    assert info["entries"] == 1 and info["skipped_stubs"] == 0


def test_a_playlist_source_has_no_default_video_id(tmp_path):
    [e], _ = load_eval_set(write(tmp_path, [entry(source_id="PLabcdefghijk123")]))
    assert e["video_id"] is None
    [e], _ = load_eval_set(write(tmp_path, [entry(source_id="PLabcdefghijk123", video_id=VID)], "b.json"))
    assert e["video_id"] == VID


def test_unfilled_stubs_are_ignored_and_counted(tmp_path):
    stubs = [entry(id="s1", question=None), entry(id="s2", question="   "), entry(id="e2")]
    entries, info = load_eval_set(write(tmp_path, stubs))
    assert [e["id"] for e in entries] == ["e2"] and info["skipped_stubs"] == 2


def test_a_file_of_only_stubs_is_an_error_that_says_what_to_do(tmp_path):
    with pytest.raises(EvalSetError, match="no usable questions"):
        load_eval_set(write(tmp_path, [entry(question=None)]))


def test_the_wrapped_questions_form_is_accepted(tmp_path):
    entries, _ = load_eval_set(write(tmp_path, {"questions": [entry()]}))
    assert len(entries) == 1


def test_example_format_loads_with_top_level_source_and_expected_ranges(tmp_path):
    payload = {
        "source_id": "PLabcdefghijk123",
        "questions": [{
            "q": "What does the video say about Python?",
            "expected": [{"video_id": VID, "start": "5:42", "end": "6:30"}],
        }],
    }
    [loaded], _ = load_eval_set(write(tmp_path, payload))
    assert loaded["question"] == payload["questions"][0]["q"]
    assert loaded["source_id"] == payload["source_id"]
    assert loaded["video_id"] == VID
    assert loaded["gold_timestamp_seconds"] is None
    assert loaded["expected"][0]["end"] == "6:30"


def test_example_format_marks_empty_expected_ranges_as_refusal(tmp_path):
    payload = {
        "source_id": "PLabcdefghijk123",
        "questions": [{"q": "What is not covered?", "expected": []}],
    }
    [loaded], _ = load_eval_set(write(tmp_path, payload))
    assert loaded["expects_refusal"] is True
    assert loaded["video_id"] is None


@pytest.mark.parametrize("value, seconds", [(342, 342), ("5:42", 342), ("1:02:03", 3723), ("90", 90), (12.0, 12)])
def test_timestamps_are_normalised_to_seconds(tmp_path, value, seconds):
    [e], _ = load_eval_set(write(tmp_path, [entry(gold_timestamp_seconds=value)]))
    assert e["gold_timestamp_seconds"] == seconds


def test_a_zero_timestamp_is_kept_not_treated_as_missing(tmp_path):
    [e], _ = load_eval_set(write(tmp_path, [entry(gold_timestamp_seconds=0)]))
    assert e["gold_timestamp_seconds"] == 0


@pytest.mark.parametrize("bad, message", [
    (entry(category="trivia"), "unknown category"),
    (entry(source_id=""), "source_id"),
    (entry(gold_timestamp_seconds="soon"), "bad timestamp"),
    (entry(gold_answer_contains=5), "gold_answer_contains"),
    (entry(history="not a list"), "history"),
    (entry(history=[{"question": None}]), "history turn 1"),
])
def test_malformed_entries_are_rejected_with_their_id(tmp_path, bad, message):
    with pytest.raises(EvalSetError, match=message):
        load_eval_set(write(tmp_path, [bad]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(EvalSetError, match="duplicate"):
        load_eval_set(write(tmp_path, [entry(), entry()]))


def test_history_is_normalised(tmp_path):
    hist = [{"question": "What is Python?", "answer": "A language."}, "And its creator?"]
    [e], _ = load_eval_set(write(tmp_path, [entry(category="followup", history=hist)]))
    assert e["history"] == [{"question": "What is Python?", "answer": "A language."},
                            {"question": "And its creator?", "answer": None}]


def test_missing_and_invalid_files_give_helpful_errors(tmp_path):
    with pytest.raises(EvalSetError, match="build_eval_template"):
        load_eval_set(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(EvalSetError, match="not valid JSON"):
        load_eval_set(bad)


def test_blank_gold_strings_count_as_unset(tmp_path):
    [e], _ = load_eval_set(write(tmp_path, [entry(gold_answer_contains="  ")]))
    assert e["gold_answer_contains"] is None


# --- relevance --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, gold, expected", [
    ("A Hash Table maps keys to values.", "hash table", True),            # case-insensitive
    ("lookups are O(1) on average", "o 1", True),                          # punctuation-insensitive
    ("the   hash\ntable", "hash table", True),                             # whitespace-insensitive
    ("completely unrelated", "hash table", False),
    ("a hash table", ["tree", "hash table"], True),                        # any of several
    ("a hash table", ["tree", "heap"], False),
    ("anything", None, False), ("anything", "", False), ("anything", [], False),
    ("naïve bayes classifier", "NAÏVE BAYES", True),                       # unicode
])
def test_is_relevant(text, gold, expected):
    assert common.is_relevant(text, gold) is expected


def test_first_relevant_rank_respects_depth():
    texts = ["no", "no", "yes here", "no", "yes again"]
    assert common.first_relevant_rank(texts, "yes") == 3
    assert common.first_relevant_rank(texts, "yes", depth=2) is None
    assert common.first_relevant_rank(texts, "absent") is None
    assert common.first_relevant_rank([], "x") is None


# --- metric formulas ---------------------------------------------------------------------------------------------
def test_recall_at_k_formula():
    ranks = [1, 3, 5, 6, None]
    assert common.recall_at_k(ranks, 5) == 3 / 5   # ranks 1, 3, 5 are within the top 5
    assert common.recall_at_k(ranks, 1) == 1 / 5
    assert common.recall_at_k([], 5) is None


def test_mrr_formula():
    assert common.mean_reciprocal_rank([1, 2, 4, None]) == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4)
    assert common.mean_reciprocal_rank([1, 12], depth=10) == pytest.approx(1 / 2)  # rank 12 is beyond the depth: 0
    assert common.mean_reciprocal_rank([]) is None


def test_wilson_interval():
    assert common.wilson_interval(0, 0) is None
    low, high = common.wilson_interval(5, 10)
    assert low < 0.5 < high and 0 <= low and high <= 1
    assert common.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert common.wilson_interval(0, 10)[0] == 0.0
    narrow, wide = common.wilson_interval(50, 100), common.wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])  # more questions, tighter interval


def test_median():
    assert common.median([]) is None
    assert common.median([3, 1, 2]) == 2
    assert common.median([1, 2, 3, 4]) == 2.5


def test_pct():
    assert common.pct(None) == "n/a" and common.pct(0.5) == "50%" and common.pct(0.1234, 1) == "12.3%"


def test_format_table_aligns_columns():
    text = common.format_table(["name", "n"], [["alpha", 1], ["b", 1000]])
    lines = text.splitlines()
    assert len(lines) == 4 and len({len(line) for line in lines}) == 1


# --- result files -----------------------------------------------------------------------------------------------
def test_results_carry_a_timestamp_and_the_commit(tmp_path):
    path = common.write_results("demo", {"value": 1}, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.startswith("demo_") and path.suffix == ".json"
    assert data["value"] == 1 and "T" in data["generated_at"] and "commit" in data["git"]


def test_git_info_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(common.subprocess, "run", boom)
    assert common.git_info() == {"commit": "unknown", "dirty": None}
