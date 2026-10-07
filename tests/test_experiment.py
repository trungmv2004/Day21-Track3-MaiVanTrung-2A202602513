"""Integrity checks and paired qualitative examples; test data is never lab evidence."""
import dataclasses
import hashlib
import json

import pytest

from labkit import evaluate as ev, experiment
from labkit.config import OPTIMIZED_PROMPT, get_tier
from labkit.submission import load_bundle, render_report, table


@pytest.fixture
def frozen_workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    for name in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl"):
        (tmp_path / "data" / name).write_text('{}\n', encoding="utf-8")
    tier = dataclasses.replace(get_tier("CPU"), model_id="unit-test/model")
    record = {"tier": tier.name, "model": tier.model_id,
              "optimized_prompt_sha": hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16],
              "dataset_sha256": experiment.dataset_hashes(tmp_path),
              "frozen_at_utc": "2026-10-07T00:00:00+00:00"}
    (tmp_path / "results/baselines_frozen.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path, tier


def test_frozen_baseline_requires_prior_measurement(tmp_path):
    with pytest.raises(ValueError, match="NB2"):
        experiment.require_frozen_baseline(tmp_path, get_tier("CPU"))


def test_frozen_baseline_accepts_same_experiment(frozen_workspace):
    root, tier = frozen_workspace
    assert experiment.require_frozen_baseline(root, tier)["model"] == tier.model_id


def test_frozen_baseline_rejects_changed_dataset(frozen_workspace):
    root, tier = frozen_workspace
    (root / "data/eval_target.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="Checksum"):
        experiment.require_frozen_baseline(root, tier)


def test_frozen_baseline_rejects_changed_model(frozen_workspace):
    root, tier = frozen_workspace
    with pytest.raises(ValueError, match="Model"):
        experiment.require_frozen_baseline(root, dataclasses.replace(tier, model_id="another/model"))


def test_baseline_cannot_be_overwritten_after_training(tmp_path):
    adapter = tmp_path / "adapters/correct"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"test fixture")
    with pytest.raises(ValueError, match="adapter"):
        experiment.ensure_baseline_not_trained(tmp_path)


def test_a_wrong_ft_answer_is_not_necessarily_a_loss():
    label = {"intent": "doi_tra", "urgency": "cao", "product": "balo", "sentiment": "tieu_cuc"}
    partial = json.dumps({"intent": "doi_tra"})
    row = experiment.paired_examples([{"input": "balo", "label": label}], ["broken"], [partial])[0]
    assert row["ft_score"] == 0.25 and row["outcome"] == "win"
    assert row["ft_wrong_fields"] == ["urgency", "product", "sentiment"]


def test_qualitative_preserves_full_prediction_and_true_losses():
    label = {"intent": "doi_tra", "urgency": "cao", "product": "balo", "sentiment": "tieu_cuc"}
    correct = json.dumps(label)
    wrong = '{"intent":"hoan_tien"}' + " " * 200
    examples = experiment.paired_examples([{"input": "x" * 200, "label": label}], [correct], [wrong])
    assert examples[0]["outcome"] == "loss"
    assert examples[0]["ft_pred"] == wrong
    assert len(examples[0]["ticket"]) == 200


def test_pairing_refuses_misaligned_predictions():
    with pytest.raises(ValueError, match="Prediction"):
        experiment.paired_examples([{"input": "x", "label": {}}], [], ["x"])


def test_selection_includes_two_losses_and_does_not_duplicate():
    rows = [{"i": i, "outcome": "loss" if i < 3 else "win", "delta": -1 if i < 3 else 1}
            for i in range(6)]
    selected = experiment.select_examples(rows)
    assert len(selected) == 5 and len({row["i"] for row in selected}) == 5
    assert sum(row["outcome"] == "loss" for row in selected) >= 2


def test_selection_does_not_fabricate_losses():
    rows = [{"i": i, "outcome": "win", "delta": 1} for i in range(5)]
    assert all(row["outcome"] == "win" for row in experiment.select_examples(rows))


def test_report_refuses_incomplete_artifacts(tmp_path):
    with pytest.raises(ValueError, match="baselines_frozen"):
        load_bundle(tmp_path)


def test_report_table_keeps_pipes_and_thinking_tags_readable():
    result = table([{"output": '<think>a|b</think>\nJSON'}], ["output"])
    assert "&#124;" in result and "&lt;think&gt;" in result and "<br>" in result


@pytest.fixture
def complete_fixture(tmp_path):
    """Synthetic integration fixture in pytest temp; never written into results/."""
    import csv

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    label = {"intent": "doi_tra", "urgency": "cao", "product": "balo", "sentiment": "tieu_cuc"}
    target = [{"input": f"ticket {i}", "label": label} for i in range(6)]
    (tmp_path / "data/train_seed.jsonl").write_text('{}\n', encoding="utf-8")
    (tmp_path / "data/eval_target.jsonl").write_text(
        "\n".join(json.dumps(row) for row in target), encoding="utf-8")
    (tmp_path / "data/eval_regression.jsonl").write_text(
        json.dumps({"instruction": "fixture", "keywords": ["answer"]}), encoding="utf-8")
    correct = json.dumps(label)
    b_preds = [correct, correct, "broken", "broken", correct, correct]
    ft_preds = ["broken", "broken", correct, correct, correct, correct]
    scores = ev.GroupScores(target=4/6, regression=1, format=4/6, latency_ms=12, n=6)
    def write(name, value):
        (tmp_path / "results" / name).write_text(json.dumps(value), encoding="utf-8")
    model = "synthetic-test/model"
    frozen = {
        "model": model, "tier": "CPU", "smoke_mode": False, "eval_limit": None,
        "frozen_at_utc": "2026-10-07T00:00:00+00:00",
        "dataset_sha256": experiment.dataset_hashes(tmp_path),
        "optimized_prompt_sha": hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16],
        "baseline_a": ev.GroupScores(n=6).as_dict(), "baseline_b": scores.as_dict(),
        "n_target": 6, "n_regression": 1, "target_predictions_b": b_preds,
    }
    write("baselines_frozen.json", frozen)
    baseline_sha = experiment.file_hash(tmp_path / "results/baselines_frozen.json")
    rows = []
    for key in ("correct", "attn_only", "wrong_lr", "qlora"):
        row = dict(run=key, model=model, tier="CPU", placement="text-linear", r=16,
                   lora_alpha=32, trainable_params=100, learning_rate=0.0001,
                   load_in_4bit=False, max_steps=6, mask_mode="assistant-only", max_length=512,
                   epochs=2, started_at_utc="2026-10-07T01:00:00+00:00",
                   baseline_sha256=baseline_sha, final_loss=0.3, train_seconds=10, peak_vram_gb=4)
        if key == "attn_only":
            row.update(placement="attn-only", r=32, lora_alpha=64, trainable_params=102)
        if key == "wrong_lr":
            row["learning_rate"] = 0.00001
        if key == "qlora":
            row.update(load_in_4bit=True, peak_vram_gb=2)
        rows.append(row)
        adapter = tmp_path / "adapters" / key
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"unit-test dummy")
        (adapter / "adapter_config.json").write_text('{}')
        write(f"training_{key}.json", {"log_history": [{"loss": 1}, {"loss": 0.3}],
                                      "device": {"name": "unit-test fixture"}})
    with (tmp_path / "results/runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write("evaluation_correct.json", {"model": model, "baseline_sha256": baseline_sha,
                                       "scores": scores.as_dict(), "target_predictions": ft_preds,
                                       "regression_predictions": ["answer"]})
    write("autopsy.json", [{"run": row["run"], "target": round(scores.target, 4), "n": 6} for row in rows])
    write("qualitative.json", experiment.paired_examples(target, b_preds, ft_preds))
    write("verdict.json", {"verdict": ev.regression_gate(scores, scores).as_dict(), "valid_trace_rate": 0})
    write("mask_proof.json", {"answer_is_supervised": True, "question_is_masked": True,
                              "supervised_fraction": 0.4, "n_supervised": 4, "n_total": 10,
                              "supervised_preview": correct})
    write("template_check.json", {"verdict": "unit-test template"})
    write("token_stats.json", {"p95": 98, "suggested_max_length": 256})
    return tmp_path


def test_complete_failed_verdict_produces_honest_report(complete_fixture):
    bundle = load_bundle(complete_fixture)
    report = render_report(bundle, name="test learner", student_id="test ID")
    assert "**FAILED**" in report
    assert "2 ca FT thua" in report
    assert "Chưa có phản tư cá nhân" in report
    conclusion = report.split("## Kết luận\n", 1)[1].split("## Phản tư", 1)[0]
    assert len(conclusion.split()) >= 150


@pytest.mark.parametrize("artifact,field,value", [
    ("evaluation_correct.json", "baseline_sha256", "changed"),
    ("evaluation_correct.json", "model", "another/model"),
])
def test_report_refuses_mismatched_experiment(complete_fixture, artifact, field, value):
    path = complete_fixture / "results" / artifact
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Evaluation"):
        load_bundle(complete_fixture)


def test_report_refuses_falsified_gate(complete_fixture):
    path = complete_fixture / "results/verdict.json"
    record = json.loads(path.read_text())
    record["verdict"]["passed"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Verdict"):
        load_bundle(complete_fixture)


def test_report_refuses_truncated_qualitative_output(complete_fixture):
    path = complete_fixture / "results/qualitative.json"
    record = json.loads(path.read_text())
    record[0]["baseline_b_pred"] = "cut off"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="qualitative"):
        load_bundle(complete_fixture)


def test_option_a_keeps_contrast_evidence_without_exporting_contrast_weights(complete_fixture):
    for key in ("attn_only", "wrong_lr", "qlora"):
        folder = complete_fixture / "adapters" / key
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            (folder / filename).unlink()
        folder.rmdir()
    assert load_bundle(complete_fixture, require_all_adapters=False)["runs"]["attn_only"]
    with pytest.raises(ValueError, match="adapter"):
        load_bundle(complete_fixture)


def test_option_a_still_requires_the_main_adapter(complete_fixture):
    (complete_fixture / "adapters/correct/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="correct"):
        load_bundle(complete_fixture, require_all_adapters=False)
