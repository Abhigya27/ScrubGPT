"""run_all: choosing steps, passing arguments, surviving failures, and the exit code."""
import json

import pytest

from evaluation import common, run_all as RA


def test_all_steps_by_default_in_canonical_order():
    assert RA.select_steps(None, None, False) == list(RA.STEPS)
    assert list(RA.STEPS)[0] == "ablation" and "ingestion_stats" not in RA.STEPS   # the destructive benchmark is not a step


def test_only_skip_and_fast():
    assert RA.select_steps("timestamp,ablation", None, False) == ["ablation", "timestamp"]   # canonical order, not typed order
    assert "robustness" not in RA.select_steps(None, "robustness", False)
    fast = RA.select_steps(None, None, True)
    assert "answer_quality" not in fast and "performance" not in fast and "grounding" in fast
    assert RA.select_steps("answer_quality,ablation", None, True) == ["ablation"]


def test_unknown_step_names_are_an_error():
    with pytest.raises(ValueError, match="unknown step"):
        RA.select_steps("ablation,nonsense", None, False)
    with pytest.raises(ValueError, match="--skip"):
        RA.select_steps(None, "nope", False)


def test_arguments_depend_on_what_each_step_accepts():
    assert RA.step_args("ablation", "set.json", "out", 2.0) == ["--eval-set", "set.json", "--out-dir", "out"]     # no --delay
    assert RA.step_args("grounding", "set.json", None, 2.0) == ["--eval-set", "set.json", "--delay", "2.0"]
    assert RA.step_args("grounding", "set.json", None, 0) == ["--eval-set", "set.json"]
    assert RA.step_args("translation", "set.json", "out", 2.0) == ["--out-dir", "out"]                           # needs no eval set
    assert RA.step_args("corpus", "set.json", None, 0) == []


def test_every_step_module_exists_and_has_a_main():
    import importlib

    for name, (module, *_rest) in RA.STEPS.items():
        assert callable(importlib.import_module(f"evaluation.{module}").main), name


def test_a_failing_step_does_not_stop_the_others(tmp_path):
    seen = []

    def runner(name, args):
        seen.append(name)
        if name == "timestamp":
            raise RuntimeError("boom")
        if name == "grounding":
            return 3
        if name == "robustness":
            raise SystemExit(2)
        common.write_results(name, {}, tmp_path)
        return 0

    results = RA.run_steps(["ablation", "robustness", "timestamp", "grounding", "translation"], "set.json", str(tmp_path), runner=runner)
    assert seen == ["ablation", "robustness", "timestamp", "grounding", "translation"]
    by_name = {r["step"]: r for r in results}
    assert by_name["ablation"]["code"] == 0 and len(by_name["ablation"]["files"]) == 1 and by_name["ablation"]["files"][0].startswith("ablation_")
    assert by_name["timestamp"]["code"] == 1 and "boom" in by_name["timestamp"]["error"]
    assert by_name["robustness"]["code"] == 2
    assert by_name["grounding"]["code"] == 3
    assert by_name["translation"]["files"][0].startswith("translation_")


def test_exit_codes():
    assert RA.overall_code([{"code": 0}, {"code": 0}]) == 0
    assert RA.overall_code([{"code": 0}, {"code": 3}]) == 3
    assert RA.overall_code([{"code": 3}, {"code": 1}]) == 1
    assert RA.overall_code([{"code": 2}]) == 1


def write_set(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps([{"id": "a", "source_id": "PL", "question": "q?", "gold_answer": "x"},
                                {"id": "b", "source_id": "PL", "question": "r?", "expects_refusal": True}]), encoding="utf-8")
    return str(path)


def test_command_runs_the_selected_steps_and_returns_their_status(tmp_path, monkeypatch):
    ran = []

    def fake_run_steps(names, eval_set, out_dir, delay=0.0, runner=None):
        ran.append((names, eval_set, out_dir, delay))
        return [{"step": n, "code": 0, "seconds": 0.1, "error": None, "files": [f"{n}_x.json"]} for n in names]

    monkeypatch.setattr(RA, "run_steps", fake_run_steps)
    assert RA.main(["--eval-set", write_set(tmp_path), "--only", "ablation,corpus", "--out-dir", str(tmp_path), "--delay", "1.5"]) == 0
    assert ran == [(["ablation", "corpus"], write_set(tmp_path), str(tmp_path), 1.5)]
    monkeypatch.setattr(RA, "run_steps", lambda *a, **k: [{"step": "ablation", "code": 1, "seconds": 0, "error": "x", "files": []}])
    assert RA.main(["--eval-set", write_set(tmp_path), "--only", "ablation"]) == 1


def test_command_fails_fast_on_a_bad_eval_set_before_spending_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(RA, "run_steps", lambda *a, **k: pytest.fail("no step should run"))
    assert RA.main(["--eval-set", str(tmp_path / "missing.json"), "--only", "ablation"]) != 0
    assert RA.main(["--eval-set", write_set(tmp_path), "--only", "nonsense"]) != 0


def test_steps_that_need_no_eval_set_do_not_require_one(tmp_path, monkeypatch):
    monkeypatch.setattr(RA, "run_steps", lambda names, *a, **k: [{"step": n, "code": 0, "seconds": 0, "error": None, "files": []} for n in names])
    assert RA.main(["--eval-set", str(tmp_path / "missing.json"), "--only", "translation,corpus"]) == 0


def test_command_can_refresh_the_readme(tmp_path, monkeypatch):
    from evaluation import render_results

    monkeypatch.setattr(RA, "run_steps", lambda names, *a, **k: [{"step": n, "code": 0, "seconds": 0, "error": None, "files": []} for n in names])
    seen = []
    monkeypatch.setattr(render_results, "main", lambda argv=None: seen.append(argv) or 0)
    assert RA.main(["--eval-set", write_set(tmp_path), "--only", "ablation", "--out-dir", str(tmp_path), "--write-readme", "--count-tests"]) == 0
    assert seen == [["--results-dir", str(tmp_path), "--write-readme", "--count-tests"]]
