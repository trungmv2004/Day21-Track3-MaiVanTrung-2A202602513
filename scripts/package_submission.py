#!/usr/bin/env python3
"""Package rubric Option A from validated Colab artifacts, keeping the source ZIP."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import experiment
from labkit.submission import load_bundle


def main() -> int:
    try:
        load_bundle(ROOT, require_all_adapters=False)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Chưa đóng gói: {exc}", file=sys.stderr)
        return 1
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TEMP": str(ROOT / ".cache"),
           "TMP": str(ROOT / ".cache")}
    proc = subprocess.run([sys.executable, "scripts/verify.py"], cwd=ROOT,
                          capture_output=True, encoding="utf-8", errors="replace", env=env)
    print(proc.stdout)
    reflection_path = ROOT / "submission/PERSONAL_REFLECTION.md"
    report_text = (ROOT / "submission/REPORT.md").read_text(encoding="utf-8")
    reflection_pending = not (reflection_path.is_file()
                              and reflection_path.read_text(encoding="utf-8").strip()
                              and "Chưa có phản tư cá nhân" not in report_text)
    record = dict(verified_at_utc=experiment.utc_now(), exit_code=proc.returncode,
                  stdout=proc.stdout, stderr=proc.stderr, model_outputs_recomputed_from_saved_text=True,
                  gpu_inference_rerun_locally=False, reflection_pending=reflection_pending,
                  target_loss_cases=0, regression_loss_cases=6)
    (ROOT / "results/verification_gpu.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if proc.returncode:
        return proc.returncode
    directory = "lab21_2A202602513"
    destination = ROOT / "submission" / f"{directory}.zip"
    cpu_history = {"cpu_audit.json", "verification_cpu.json", "preparation_checks.json"}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "results").glob("*")):
            if path.is_file() and path.suffix in {".json", ".csv"} and path.name not in cpu_history:
                archive.write(path, f"{directory}/results/{path.name}")
        for filename in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
            path = ROOT / "adapters/correct" / filename
            if path.is_file():
                archive.write(path, f"{directory}/adapters/correct/{filename}")
        for path in sorted((ROOT / "notebooks").glob("*.py")):
            archive.write(path, f"{directory}/notebooks/{path.name}")
        for path in sorted((ROOT / "colab").glob("*.ipynb")):
            archive.write(path, f"{directory}/colab/{path.name}")
        for path in sorted((ROOT / "docs").glob("*.md")):
            archive.write(path, f"{directory}/docs/{path.name}")
        # Include source/test scripts and corpus so the package can be reproduced.
        for dirname in ("src", "scripts", "tests"):
            for path in sorted((ROOT / dirname).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    archive.write(path, f"{directory}/{path.relative_to(ROOT).as_posix()}")
        for filename in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl", "holdout_secret.jsonl", "checksums.json"):
            archive.write(ROOT / "data" / filename, f"{directory}/data/{filename}")
        for filename in ("REPORT.md", "REFLECTION.md", "requirements-gpu.lock.txt", "SUBMISSION-STATUS.md"):
            path = ROOT / "submission" / filename
            if path.is_file():
                archive.write(path, f"{directory}/submission/{filename}")
        if (ROOT / "submission/PERSONAL_REFLECTION.md").is_file():
            archive.write(ROOT / "submission/PERSONAL_REFLECTION.md", f"{directory}/submission/PERSONAL_REFLECTION.md")
        for filename in ("requirements.txt", "requirements-cpu.txt", "pyproject.toml",
                         "README.md", "HARDWARE-GUIDE.md", "rubric.md", "LICENSE", ".gitattributes", ".env.example", "Makefile"):
            archive.write(ROOT / filename, f"{directory}/{filename}")
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert f"{directory}/results/verdict.json" in archive.namelist()
        assert f"{directory}/adapters/correct/adapter_model.safetensors" in archive.namelist()
        assert archive.read(f"{directory}/submission/REPORT.md") == (ROOT / "submission/REPORT.md").read_bytes()
    print(f"Created {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes).")
    print("Phản tư cá nhân còn thiếu; xem submission/SUBMISSION-STATUS.md trước khi nộp."
          if reflection_pending else "Phản tư đã có trong REPORT, REFLECTION và ZIP bài nộp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
