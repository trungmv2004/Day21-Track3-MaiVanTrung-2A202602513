"""Keep frozen baselines tied to the model, prompts and dataset used for training."""
from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone

from .config import OPTIMIZED_PROMPT, Tier


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_hashes(root: pathlib.Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
            for name in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl")}


def file_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_frozen_baseline(root: pathlib.Path, tier: Tier) -> dict:
    path = root / "results" / "baselines_frozen.json"
    if not path.is_file():
        raise ValueError("Chưa có baseline đóng băng: chạy NB2 trước NB3/NB4/NB5.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("model") != tier.model_id or frozen.get("tier") != tier.name:
        raise ValueError("Model/tier khác baseline: dùng cùng model và cấu hình đã đóng băng.")
    expected_prompt = hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16]
    if frozen.get("optimized_prompt_sha") != expected_prompt:
        raise ValueError("Prompt (b) đã đổi sau đóng băng; không thể so sánh với baseline cũ.")
    if frozen.get("dataset_sha256") != dataset_hashes(root):
        raise ValueError("Checksum corpus không khớp baseline; cần NB2 mới trong workspace thí nghiệm riêng.")
    if not frozen.get("frozen_at_utc"):
        raise ValueError("Baseline thiếu thời điểm đóng băng; cần chạy NB2 trước train.")
    return frozen


def ensure_baseline_not_trained(root: pathlib.Path) -> None:
    # Do not remeasure a baseline after seeing the adapters from this experiment.
    if any((root / "adapters").glob("*/adapter_model.safetensors")):
        raise ValueError("Workspace đã có adapter: không ghi đè NB2 sau train. Dùng workspace mới cho thí nghiệm mới.")


def paired_examples(target: list[dict], baseline: list[str], tuned: list[str]) -> list[dict]:
    from .evaluate import TRIAGE_KEYS, triage_field_accuracy

    if not target or len(target) != len(baseline) or len(target) != len(tuned):
        raise ValueError("Prediction baseline/FT phải đủ và khớp từng mẫu target.")
    rows = []
    for index, (record, b_pred, ft_pred) in enumerate(zip(target, baseline, tuned)):
        label = record["label"]
        b_score = triage_field_accuracy(b_pred, label)
        ft_score = triage_field_accuracy(ft_pred, label)
        delta = ft_score - b_score
        rows.append({
            "i": index, "ticket": record["input"], "label": label,
            "baseline_b_pred": b_pred, "ft_pred": ft_pred,
            "baseline_b_score": b_score, "ft_score": ft_score,
            "delta": delta, "outcome": "loss" if delta < 0 else "win" if delta > 0 else "tie",
            "ft_wrong_fields": [key for key in TRIAGE_KEYS
                                if triage_field_accuracy(ft_pred, label, keys=[key]) == 0],
        })
    return sorted(rows, key=lambda row: (row["delta"], row["ft_score"], row["i"]))


def select_examples(rows: list[dict], limit: int = 5) -> list[dict]:
    selected = [row for row in rows if row["outcome"] == "loss"][:2]
    selected += sorted((row for row in rows if row["outcome"] == "win"),
                       key=lambda row: (-row["delta"], row["i"]))[:2]
    seen = {row["i"] for row in selected}
    for row in rows:
        if len(selected) >= limit:
            break
        if row["i"] not in seen:
            selected.append(row)
            seen.add(row["i"])
    return selected[:limit]
