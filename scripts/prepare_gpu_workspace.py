#!/usr/bin/env python3
"""Create a local-upload Colab notebook and workspace ZIP; no Git/network writes."""
from __future__ import annotations

import hashlib
import pathlib
import zipfile

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    cells = [nbformat.v4.new_markdown_cell(
        "# Lab 21 — chạy workspace cục bộ trên T4\n\n"
        "Chọn Runtime → Change runtime type → **T4 GPU**. Chạy lần lượt các ô. "
        "Tải lên `submission/lab21_gpu_workspace.zip`; không cần commit hoặc push. "
        "Core NB1–NB5 khoảng 100–130 phút theo README. "
        "Ô 4 tạo báo cáo từ kết quả GPU. Bổ sung phản tư cá nhân trước khi nộp."),
        nbformat.v4.new_code_cell('''# 1. Upload workspace ZIP từ máy
import io, os, pathlib, subprocess, sys, time, zipfile
from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 1, "Chọn đúng một workspace ZIP."
name, payload = next(iter(uploaded.items()))
assert name.endswith(".zip"), "Cần file ZIP."
LAB_ROOT = pathlib.Path(f"/content/lab21-local-{time.time_ns()}")
LAB_ROOT.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert (LAB_ROOT / member.filename).resolve().is_relative_to(LAB_ROOT), "Đường dẫn ZIP không hợp lệ."
    archive.extractall(LAB_ROOT)
assert (LAB_ROOT / "scripts/colab_run.py").is_file(), "ZIP thiếu source lab."
os.chdir(LAB_ROOT)
print("Workspace:", LAB_ROOT)
'''), nbformat.v4.new_code_cell('''# 2. Setup GPU và kiểm tra code
import torch
assert torch.cuda.is_available(), "Chọn Runtime → Change runtime type → T4 GPU."
os.environ.update(COMPUTE_TIER="T4", MASK_MODE="assistant-only", EPOCHS="2",
                  HF_HOME=str(LAB_ROOT / ".cache/huggingface"))
for key in ("BASE_MODEL", "EVAL_LIMIT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "ONLY", "FORCE_RETRAIN"):
    os.environ.pop(key, None)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, str(LAB_ROOT / "src"))
from labkit import device
print(device.banner())
subprocess.run([sys.executable, "scripts/verify.py", "--smoke"], check=True)
'''), nbformat.v4.new_code_cell('''# 3. Core: đo baseline TRƯỚC train, đầy đủ eval, cùng ngân sách step
STAGES = ["nb1", "nb2", "nb3", "nb4", "nb5"]
# Nếu run bị ngắt: trong cùng workspace, chỉ chạy các stage chưa hoàn thành.
# NB4 tự bỏ qua adapter đã lưu. Không chạy lại NB2 để sửa baseline sau khi thấy FT.
subprocess.run([sys.executable, "-u", "scripts/colab_run.py", *STAGES], check=True)
'''), nbformat.v4.new_code_cell('''# 4. Tạo báo cáo từ số đo thật và kiểm tra artefact
import json
report_args = [sys.executable, "scripts/write_gpu_report.py"]
# Nếu đã có phản tư cá nhân, thêm file này vào workspace trước khi chạy ô 4.
reflection = LAB_ROOT / "submission/PERSONAL_REFLECTION.md"
if reflection.is_file():
    report_args += ["--reflection", str(reflection)]
subprocess.run(report_args, check=True)
for artifact in ("mask_proof.json", "token_stats.json", "baselines_frozen.json", "autopsy.json", "verdict.json"):
    print("\\n", artifact)
    print((LAB_ROOT / "results" / artifact).read_text(encoding="utf-8"))
print("\\nruns.csv\\n", (LAB_ROOT / "results/runs.csv").read_text(encoding="utf-8"))
verification = subprocess.run([sys.executable, "scripts/verify.py"], check=False)
print("verify exit:", verification.returncode)
print("Đọc REPORT.md, bổ sung phản tư cá nhân, kiểm tra ca thua và chạy verify lần cuối trước khi nộp.")
'''), nbformat.v4.new_code_cell('''# 5. Tải source + report + toàn bộ results + adapter chính về máy
import importlib.metadata
(LAB_ROOT / "submission/requirements-gpu.lock.txt").write_text(
    "\\n".join(sorted(f"{dist.metadata['Name']}=={dist.version}"
                     for dist in importlib.metadata.distributions())) + "\\n", encoding="utf-8")
output = pathlib.Path("/content/lab21_gpu_results.zip")
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory in ("results", "submission", "notebooks", "src", "scripts", "data", "tests", "docs", "adapters/correct"):
        for path in sorted((LAB_ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".zip":
                archive.write(path, path.relative_to(LAB_ROOT))
    for filename in ("requirements.txt", "requirements-cpu.txt", "pyproject.toml", "README.md", "rubric.md", ".gitattributes"):
        archive.write(LAB_ROOT / filename, filename)
files.download(str(output))
''')]
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(f"{index}\0{cell.source}".encode()).hexdigest()[:8]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
    })
    nbformat.validate(notebook)
    nb_path = ROOT / "colab" / "Lab21_LOCAL_UPLOAD.ipynb"
    nbformat.write(notebook, nb_path)

    archive_path = ROOT / "submission" / "lab21_gpu_workspace.zip"
    allowed = {".py", ".ipynb", ".md", ".json", ".jsonl", ".csv", ".txt"}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("colab", "notebooks", "src", "scripts", "tests", "data", "submission", "results", "docs"):
            for path in sorted((ROOT / directory).rglob("*")):
                rel = path.relative_to(ROOT)
                if (path.is_file() and path.suffix in allowed
                        and "__pycache__" not in rel.parts
                        and not str(rel).replace("\\", "/").startswith("data/split/")):
                    archive.write(path, rel)
        for filename in ("README.md", "HARDWARE-GUIDE.md", "rubric.md", "requirements.txt",
                         "requirements-cpu.txt", "pyproject.toml", "Makefile", "LICENSE",
                         ".gitattributes", ".gitignore", ".env.example"):
            archive.write(ROOT / filename, filename)
    print(f"Created {nb_path.relative_to(ROOT)}")
    print(f"Created {archive_path.relative_to(ROOT)} ({archive_path.stat().st_size:,} bytes)")
    print("This ZIP is for continuing on GPU; the core lab is not complete yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
