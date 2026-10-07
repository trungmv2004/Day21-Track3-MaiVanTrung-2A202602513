#!/usr/bin/env python3
"""Record corpus and mask checks using the real tokenizer, without model weights.

Run after NB1: python scripts/audit_cpu.py
This is additional NB1 evidence; it does not replace GPU baselines or evaluation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, report
from labkit.config import get_tier


def load_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / name).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    from transformers import AutoTokenizer

    tier = get_tier()
    # NB1 already downloads this tokenizer; the audit is reproducible offline.
    tok = AutoTokenizer.from_pretrained(tier.model_id, local_files_only=True)
    corpus = load_rows("train_seed.jsonl")
    target = load_rows("eval_target.jsonl")
    regression = load_rows("eval_regression.jsonl")
    train, val = data.split(corpus, train_frac=0.9, seed=42)
    refs = json.loads((ROOT / "data" / "checksums.json").read_text(encoding="utf-8"))
    # Do not inspect the secret holdout's contents for training or analysis.
    hashes = {name: hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()[:16]
              for name in ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl")}
    problems: list[str] = []
    for name, rows in (("train_seed", corpus), ("eval_target", target)):
        for index, row in enumerate(rows):
            label = row["label"]
            valid = (
                set(label) == {"intent", "urgency", "product", "sentiment"}
                and label["intent"] in {"doi_tra", "van_chuyen", "hoan_tien", "san_pham_loi", "hoi_thong_tin"}
                and label["urgency"] in {"cao", "trung_binh", "thap"}
                and label["sentiment"] in {"tieu_cuc", "trung_tinh", "tich_cuc"}
                and label["product"] in row["input"]
                and json.loads(row["output"]) == label
            )
            if not valid:
                problems.append(f"{name}[{index}] has inconsistent labels")

    counts = Counter()
    for row in corpus:
        messages = data.to_messages(row)
        full = data.build_example(tok, messages, max_length=8192)
        example = data.build_example(tok, messages, max_length=tier.max_length)
        supervised = data.decode_supervised(tok, example)
        counts["answer_is_supervised"] += messages[-1]["content"] in supervised
        counts["question_is_masked"] += row["input"] not in supervised
        counts["eos_is_supervised"] += tok.eos_token in supervised
        counts["nonempty_loss"] += example.n_supervised > 0
        counts["not_truncated"] += example.n_total == full.n_total
        counts["prompt_alignment"] += data.prompt_alignment(
            tok, row)["eval_prompt_is_prefix_of_training"]
    train_inputs = {row["input"] for row in train}
    val_inputs = {row["input"] for row in val}
    target_inputs = {row["input"] for row in target}
    overlaps = {
        "train_val": len(train_inputs & val_inputs),
        "corpus_target": len({row["input"] for row in corpus} & target_inputs),
    }
    checksums_match = all(hashes[name] == refs[name] for name in hashes)
    passed = (not problems and checksums_match and not any(overlaps.values())
              and all(counts[key] == len(corpus) for key in (
                  "answer_is_supervised", "question_is_masked", "eos_is_supervised",
                  "nonempty_loss", "not_truncated", "prompt_alignment")))
    result = {
        "scope": "CPU data and masking checks only; no training or model scoring",
        "passed": passed,
        "tier": tier.name,
        "model": tier.model_id,
        "max_length": tier.max_length,
        "seed": 42,
        "n_corpus": len(corpus), "n_train": len(train), "n_val": len(val),
        "n_target": len(target), "n_regression": len(regression),
        "checksums": hashes, "checksums_match": checksums_match,
        "label_errors": problems, "exact_input_overlap": overlaps,
        "mask_checks": dict(counts),
        "python_version": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in (
            "transformers", "tokenizers", "jinja2", "jupytext", "pytest")},
    }
    report.write_json(result, "cpu_audit.json", results_dir=ROOT / "results")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
