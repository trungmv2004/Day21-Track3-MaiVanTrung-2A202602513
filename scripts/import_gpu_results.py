#!/usr/bin/env python3
"""Validate a Colab export, then import measured files without replacing source/data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import struct
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import experiment
from labkit.submission import load_bundle

GPU_RESULTS = {
    "autopsy.json", "baselines_frozen.json", "evaluation_correct.json",
    "mask_proof.json", "qualitative.json", "runs.csv", "template_check.json",
    "token_stats.json", "verdict.json", "training_correct.json",
    "training_attn_only.json", "training_wrong_lr.json", "training_qlora.json",
}
ADAPTER_FILES = {
    "adapter_config.json", "adapter_model.safetensors", "README.md",
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
}


def import_archive(archive_path: pathlib.Path, root: pathlib.Path) -> dict:
    cache = root / ".cache/imports"
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=cache) as temporary:
        staging = pathlib.Path(temporary)
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP có tên file trùng lặp.")
        for name in names:
            if not (staging / name).resolve().is_relative_to(staging.resolve()):
                raise ValueError("ZIP có đường dẫn ra ngoài workspace.")
        if archive.testzip() is not None:
            raise ValueError("ZIP hỏng CRC.")
        (staging / "data").mkdir()
        for filename in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl"):
            data = archive.read("data/" + filename)
            if data != (root / "data" / filename).read_bytes():
                raise ValueError(f"Corpus khác bản local: {filename}; dừng nhập để kiểm tra.")
            (staging / "data" / filename).write_bytes(data)
        wanted = ["results/" + name for name in sorted(GPU_RESULTS)]
        wanted += ["adapters/correct/" + name for name in sorted(ADAPTER_FILES)
                   if "adapters/correct/" + name in names]
        wanted += ["submission/" + name for name in
                   ("REPORT.md", "REPORT-CPU.md", "requirements-gpu.lock.txt")]
        for name in wanted:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
        bundle = load_bundle(staging, require_all_adapters=False)
        adapter = staging / "adapters/correct"
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        correct = bundle["runs"]["correct"]
        if (config["base_model_name_or_path"] != bundle["frozen"]["model"]
                or config["r"] != int(correct["r"])
                or config["lora_alpha"] != int(correct["lora_alpha"])):
            raise ValueError("Adapter config không khớp model/rank/alpha của run.")
        weight_file = adapter / "adapter_model.safetensors"
        with weight_file.open("rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            if header_size > weight_file.stat().st_size - 8:
                raise ValueError("Header safetensors không hợp lệ.")
            header = json.loads(fh.read(header_size))
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        n_parameters = sum(math.prod(tensor["shape"]) for tensor in tensors)
        if n_parameters != int(correct["trainable_params"]):
            raise ValueError("Số tham số adapter không khớp runs.csv.")
        # Score every regression prediction again; the exported text is the evidence.
        regression = [json.loads(line) for line in (staging / "data/eval_regression.jsonl")
                      .read_text(encoding="utf-8").splitlines() if line.strip()]
        from labkit.evaluate import keyword_recall
        paired = []
        for index, (row, baseline, tuned) in enumerate(zip(
                regression, bundle["frozen"]["regression_predictions_b"],
                bundle["measured"]["regression_predictions"])):
            b_score, ft_score = keyword_recall(baseline, row["keywords"]), keyword_recall(tuned, row["keywords"])
            paired.append(dict(i=index, question=row["instruction"], keywords=row["keywords"],
                               baseline_b_pred=baseline, ft_pred=tuned,
                               baseline_b_score=b_score, ft_score=ft_score,
                               delta=ft_score-b_score,
                               outcome="loss" if ft_score < b_score else "win" if ft_score > b_score else "tie"))
        if len(paired) != len(regression):
            raise ValueError("Thiếu prediction regression trong baseline.")
        if abs(sum(row["baseline_b_score"] for row in paired)/len(paired) - bundle["frozen"]["baseline_b"]["regression"]) > 1e-8:
            raise ValueError("Regression baseline lệch output.")
        backup = pathlib.Path(tempfile.mkdtemp(prefix="before-gpu-import-", dir=cache))
        for name in wanted:
            target = root / name
            if target.is_file():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / name, target)
        (root / "results/qualitative_regression.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        record = dict(imported_at_utc=experiment.utc_now(), archive=str(archive_path.name),
                      archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                      original_archive_preserved=True, source_and_corpus_unchanged=True,
                      validated_model=bundle["frozen"]["model"], adapter_parameters=n_parameters,
                      adapter_tensors=len(tensors), all_four_runs_validated=True,
                      verdict_recomputed=True, regression_scores_recomputed=True,
                      backup=str(backup.relative_to(root)), gpu_inference_rerun_locally=False)
        (root / "results/gpu_import_check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=pathlib.Path,
                        default=ROOT / "submission/lab21_gpu_results.zip")
    args = parser.parse_args()
    try:
        record = import_archive(args.archive.resolve(), ROOT)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print(f"Không nhập kết quả GPU: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
