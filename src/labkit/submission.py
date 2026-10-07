"""Build the experiment report from measured artifacts, refusing incomplete runs."""
from __future__ import annotations

import csv
import html
import json
import pathlib
from datetime import datetime

from . import evaluate as ev, experiment
from .config import get_tier

RUNS = ("correct", "attn_only", "wrong_lr", "qlora")


def read_json(root: pathlib.Path, name: str):
    path = root / "results" / name
    if not path.is_file():
        raise ValueError(f"Thiếu results/{name}; chưa thể viết report thực nghiệm.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(root: pathlib.Path, *, require_all_adapters: bool = True) -> dict:
    frozen = read_json(root, "baselines_frozen.json")
    tier = get_tier(frozen["tier"])
    # Do not resolve BASE_MODEL from the report machine; take the frozen model ID.
    from dataclasses import replace
    tier = replace(tier, model_id=frozen["model"])
    experiment.require_frozen_baseline(root, tier)
    if frozen.get("smoke_mode") or frozen.get("eval_limit"):
        raise ValueError("EVAL_LIMIT đang rút gọn dữ liệu; cần đủ eval để nộp.")
    runs_path = root / "results" / "runs.csv"
    if not runs_path.is_file():
        raise ValueError("Thiếu runs.csv: chưa có bằng chứng train.")
    with runs_path.open(encoding="utf-8", newline="") as fh:
        latest = {row["run"]: row for row in csv.DictReader(fh)}
    if any(key not in latest for key in RUNS):
        raise ValueError("Phải có đủ correct, attn_only, wrong_lr, qlora.")
    runs = {key: latest[key] for key in RUNS}
    if len({int(row["max_steps"]) for row in runs.values()}) != 1:
        raise ValueError("Bốn run khác ngân sách step.")
    if len({row["mask_mode"] for row in runs.values()}) != 1:
        raise ValueError("Bốn run khác loss mask.")
    if len({row["max_length"] for row in runs.values()}) != 1:
        raise ValueError("Bốn run khác max_length.")
    for key in ("wrong_lr", "qlora"):
        for field in ("placement", "r", "lora_alpha", "trainable_params"):
            if runs[key][field] != runs["correct"][field]:
                raise ValueError(f"{key} đổi thêm {field}; đối chứng không còn một biến.")
    if runs["wrong_lr"]["load_in_4bit"] != runs["correct"]["load_in_4bit"]:
        raise ValueError("wrong_lr đổi thêm lượng tử hóa.")
    if runs["qlora"]["learning_rate"] != runs["correct"]["learning_rate"]:
        raise ValueError("qlora đổi thêm LR.")
    wanted = int(runs["correct"]["trainable_params"])
    relative_gap = abs(int(runs["attn_only"]["trainable_params"]) - wanted) / max(1, wanted)
    if relative_gap >= 0.05:
        raise ValueError("attn_only lệch ngân sách tham số từ 5% trở lên.")
    baseline_sha = experiment.file_hash(root / "results" / "baselines_frozen.json")
    frozen_at = datetime.fromisoformat(frozen["frozen_at_utc"])
    for key, row in runs.items():
        if row["model"] != frozen["model"] or row["tier"] != frozen["tier"]:
            raise ValueError(f"{key}: model/tier khác baseline.")
        if row.get("baseline_sha256") != baseline_sha:
            raise ValueError(f"{key}: baseline bị ghi đè hoặc khác thí nghiệm.")
        if datetime.fromisoformat(row["started_at_utc"]) < frozen_at:
            raise ValueError(f"{key}: train trước khi đóng băng baseline.")
        # Rubric Option A exports only the main adapter. All measured contrast rows,
        # logs and task scores remain mandatory when reviewing such an export.
        if require_all_adapters or key == "correct":
            adapter = root / "adapters" / key
            if not (adapter / "adapter_model.safetensors").is_file() or not (adapter / "adapter_config.json").is_file():
                raise ValueError(f"Thiếu adapter hoàn chỉnh của {key}.")
    measured = read_json(root, "evaluation_correct.json")
    if measured["model"] != frozen["model"] or measured["baseline_sha256"] != baseline_sha:
        raise ValueError("Evaluation khác model hoặc mốc đóng băng.")
    autopsy = {row["run"]: row for row in read_json(root, "autopsy.json")}
    if any(key not in autopsy for key in RUNS):
        raise ValueError("NB5 chưa chấm đủ bốn adapter.")
    target = [json.loads(line) for line in (root / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    regression = [json.loads(line) for line in (root / "data/eval_regression.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if frozen["n_target"] != len(target) or frozen["n_regression"] != len(regression):
        raise ValueError("Baseline chưa dùng đủ tập eval.")
    if any(row["n"] != len(target) for row in autopsy.values()):
        raise ValueError("Các adapter dùng khác tập target.")
    examples = experiment.paired_examples(target, frozen["target_predictions_b"], measured["target_predictions"])
    if read_json(root, "qualitative.json") != examples:
        raise ValueError("qualitative.json không khớp nhãn/output baseline/FT.")
    def group(raw):
        return ev.GroupScores(**raw)
    base_b, tuned = group(frozen["baseline_b"]), group(measured["scores"])
    recomputed_target = sum(row["ft_score"] for row in examples) / len(examples)
    if abs(recomputed_target - tuned.target) > 1e-8:
        raise ValueError("Điểm target không khớp output FT.")
    if abs(autopsy["correct"]["target"] - round(tuned.target, 4)) > 1e-8:
        raise ValueError("Autopsy correct không khớp điểm FT.")
    if abs(sum(row["baseline_b_score"] for row in examples) / len(examples) - base_b.target) > 1e-8:
        raise ValueError("Điểm baseline (b) không khớp output.")
    if len(measured["regression_predictions"]) != len(regression):
        raise ValueError("Thiếu output regression FT.")
    recomputed_regression = sum(ev.keyword_recall(pred, row["keywords"]) for pred, row in
                                zip(measured["regression_predictions"], regression)) / len(regression)
    recomputed_format = sum(ev.has_required_keys(pred, ev.TRIAGE_KEYS) for pred in
                            measured["target_predictions"]) / len(target)
    if abs(recomputed_regression - tuned.regression) > 1e-8 or abs(recomputed_format - tuned.format) > 1e-8:
        raise ValueError("Điểm regression/format không khớp output FT.")
    computed_gate = ev.regression_gate(tuned, base_b).as_dict()
    verdict = read_json(root, "verdict.json")
    if (verdict["verdict"]["passed"] != computed_gate["passed"]
            or abs(verdict["verdict"]["target_delta"] - computed_gate["target_delta"]) > 1e-8
            or abs(verdict["verdict"]["regression_delta"] - computed_gate["regression_delta"]) > 1e-8):
        raise ValueError("Verdict không khớp điểm thật; không tự đổi cổng.")
    proof = read_json(root, "mask_proof.json")
    if not proof["answer_is_supervised"] or not proof["question_is_masked"] or proof["supervised_fraction"] >= 0.95:
        raise ValueError("Mask proof không đạt.")
    histories = {key: read_json(root, f"training_{key}.json") for key in RUNS}
    return dict(frozen=frozen, tier=tier, runs=runs, autopsy=autopsy, verdict=verdict,
                measured=measured, examples=examples, relative_gap=relative_gap,
                proof=proof, template=read_json(root, "template_check.json"),
                stats=read_json(root, "token_stats.json"), histories=histories)


def table(rows: list[dict], columns: list[str]) -> str:
    def cell(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", "<br>")
    return "\n".join([
        "| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|",
        *("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in rows),
    ])


def render_report(bundle: dict, *, name: str, student_id: str, reflection: str = "") -> str:
    b = bundle
    frozen, runs, autopsy = b["frozen"], b["runs"], b["autopsy"]
    correct, attention, wrong, quant = (runs[key] for key in RUNS)
    verdict = b["verdict"]["verdict"]
    target_delta, regression_delta = verdict["target_delta"], verdict["regression_delta"]
    comparison = ev.comparison_table({"(a) naive": ev.GroupScores(**frozen["baseline_a"]),
                                      "(b) optimized": ev.GroupScores(**frozen["baseline_b"]),
                                      "(c) fine-tune": ev.GroupScores(**b["measured"]["scores"])})
    rows = [{**runs[key], "target": autopsy[key]["target"]} for key in RUNS]
    by_target = sorted(RUNS, key=lambda key: (-autopsy[key]["target"], key))
    by_loss = sorted(RUNS, key=lambda key: (float(runs[key]["final_loss"]), key))
    trace_rows = []
    for key in RUNS:
        logged = [item for item in b["histories"][key]["log_history"] if "loss" in item]
        trace_rows.append({"run": key, "loss đầu": logged[0]["loss"] if logged else "Không có log",
                           "loss cuối log": logged[-1]["loss"] if logged else "Không có log",
                           "nguồn": f"results/training_{key}.json"})
    selected = experiment.select_examples(b["examples"])
    losses = sum(row["outcome"] == "loss" for row in b["examples"])
    wins = sum(row["outcome"] == "win" for row in b["examples"])
    qualitative = [{"#": row["i"], "Ticket": row["ticket"], "Nhãn": row["label"],
                    "(b) output": row["baseline_b_pred"], "(c) output": row["ft_pred"],
                    "Điểm b/c": f"{row['baseline_b_score']}/{row['ft_score']}",
                    "Kết quả": row["outcome"], "FT sai": row["ft_wrong_fields"]} for row in selected]
    relationship = "thắng" if autopsy["attn_only"]["target"] > autopsy["correct"]["target"] else "thua" if autopsy["attn_only"]["target"] < autopsy["correct"]["target"] else "hòa"
    vram_saved = float(correct["peak_vram_gb"]) - float(quant["peak_vram_gb"])
    q_target_delta = autopsy["qlora"]["target"] - autopsy["correct"]["target"]
    reflection_text = reflection.strip() or "Chưa có phản tư cá nhân do người học cung cấp. Cần bổ sung trước khi nộp; phần phân tích kỹ thuật này do AI soạn từ số đo."
    return f'''# Lab 21 — Báo cáo thí nghiệm từ số đo thực tế

Họ tên: **{name}**. MSSV: **{student_id}**. Tier: **{frozen['tier']}**.
Base model: **{frozen['model']}**. Thiết bị thực tế: **{b['histories']['correct']['device']['name']}**.
Báo cáo được AI biên soạn từ artefact; phản tư cá nhân được tách riêng phía cuối.

## Lựa chọn, dữ liệu và loss mask

Giữ model mặc định của tier để phù hợp bộ nhớ thiết bị và giữ cùng model giữa baseline
và fine-tune. Dùng corpus mặc định 250 ticket CSKH tiếng Việt vì bốn trường JSON có
nhãn để chấm khách quan, không cần LLM judge. Train/val theo seed 42 là 225/25.
Đây là dữ liệu sinh theo mẫu, nên kết quả chưa chứng minh chất lượng trên ticket thực.
`max_length` thực tế là {correct['max_length']}; p95 đo được {b['stats']['p95']},
độ dài gợi ý {b['stats']['suggested_max_length']}. Giữ độ dài của tier với biên độ
so với corpus; lựa chọn này có thể tốn padding hơn mức gợi ý.

Mask `{correct['mask_mode']}`: giám sát {b['proof']['n_supervised']}/{b['proof']['n_total']}
token, tỷ lệ {b['proof']['supervised_fraction']}; hai assert đáp án trong loss và
câu hỏi được che đều true. Template: **{b['template']['verdict']}**.
Nội dung thực sự tính loss:

```text
{b['proof']['supervised_preview']}
```

## Baseline đóng băng trước train

Thời điểm đóng băng (UTC): `{frozen['frozen_at_utc']}`.
Model, SHA prompt và checksum corpus được kiểm tra trước NB3/NB4/NB5. Cả bốn run
ghi cùng SHA file baseline; thời điểm bắt đầu train sau mốc đóng băng.
Target gồm {frozen['n_target']} mẫu, regression {frozen['n_regression']} mẫu, không rút gọn eval.
SHA prompt (b): `{frozen['optimized_prompt_sha']}`. Prompt tối ưu giữ nguyên.

{table(comparison, ['run', 'target', 'regression', 'format', 'latency_ms', 'n'])}

Baseline (b) so với (a): target chênh
{frozen['baseline_b']['target'] - frozen['baseline_a']['target']:+.4f}.
Nếu chênh lệch không dương thì chưa đạt yêu cầu (b) mạnh hơn (a); không làm yếu
baseline hoặc viết lại kết quả sau khi thấy fine-tune để tạo lợi thế.

## Bốn run và phép so sánh công bằng

{table(rows, ['run', 'placement', 'r', 'trainable_params', 'learning_rate', 'load_in_4bit', 'max_steps', 'final_loss', 'target', 'train_seconds', 'peak_vram_gb'])}

Các run dùng cùng step, mask, độ dài, dữ liệu, seed và ngân sách epoch.
`attn_only` đổi vị trí từ text-linear sang attention q/v; rank được giải để khớp
ngân sách tham số, sai lệch {b['relative_gap']:.4%}. Rank thay đổi là điều kiện kiểm
soát ngân sách, không phải phép quét rank độc lập. `wrong_lr` chỉ đổi LR từ
{correct['learning_rate']} sang {wrong['learning_rate']}. `qlora` chỉ đổi base sang
4-bit; lúc chấm cũng dùng 4-bit. Vị trí, LR và lượng tử hóa được xét bằng các đối chứng riêng.

Theo target: {', '.join(f"{key} ({autopsy[key]['target']})" for key in by_target)}.
Theo loss: {', '.join(f"{key} ({runs[key]['final_loss']})" for key in by_loss)}.
Các điểm bằng nhau là hòa; thứ tự hiển thị trong nhóm hòa không tạo thứ hạng mới.

**Vị trí/rank:** attn_only {relationship} correct trên target. Vì ngân sách tham số
đã khớp, phép đo phản ánh thay đổi vị trí có điều kiện theo cấu hình này. Không thể
từ một cặp run suy ra tăng rank luôn tốt hoặc luôn vô ích trên mọi task/model.

**LR:** chênh target wrong_lr so với correct là
{autopsy['wrong_lr']['target'] - autopsy['correct']['target']:+.4f}; loss trung bình
đã nằm trong bảng trên. Log đầu/cuối dưới đây giúp xem đường loss thực tế, nhưng
không thay thế target để kết luận. Không gán thất bại do LR nếu số đo không ủng hộ.

{table(trace_rows, ['run', 'loss đầu', 'loss cuối log', 'nguồn'])}

**QLoRA:** tiết kiệm VRAM theo phép trừ correct − qlora là {vram_saved:+.2f} GB;
chênh target qlora − correct {q_target_delta:+.4f}. Giá trị tiết kiệm âm nghĩa là
run này không tiết kiệm theo peak đo được. Đây là đánh đổi thực nghiệm của một
model/task; không tự khái quát thành khuyến nghị cho toàn bộ dòng model.

## Phán quyết và diễn giải

**{'PASSED' if verdict['passed'] else 'FAILED'}**; target Δ={target_delta:+.6f},
regression Δ={regression_delta:+.6f}, valid_trace_rate={b['verdict']['valid_trace_rate']}.
Lý do từ cổng: {' '.join(verdict['reasons'])}

Cổng yêu cầu target vượt baseline (b) và mức giảm regression không quá 0,02.
Các delta được tính từ điểm đầy đủ của run, không suy ra từ loss hay chỉ từ ví dụ
được chọn. Một verdict FAILED vẫn là kết quả thực nghiệm hợp lệ: cần phân biệt
không vượt target với quên năng lực phổ thông. Bảng format giúp nhận diện lỗi
JSON; bảng latency cho biết chi phí suy luận. Hai nhóm này được báo cáo nhưng
không tự thêm hoặc bỏ điều kiện của cổng hiện có. valid_trace_rate chỉ mô tả output;
corpus JSON không có trace nên chưa phải thí nghiệm reasoning-trace collapse.
Muốn đổi LR, dữ liệu replay hoặc prompt cho vòng kế tiếp thì phải mở thí nghiệm
mới, đo baseline trước train, giữ tập đánh giá và ghi rõ thay đổi.

## Ví dụ định tính từ output đầy đủ

Toàn bộ target có {wins} ca FT thắng, {losses} ca FT thua, còn lại hòa với (b).
{'Đã chọn ít nhất hai ca thua.' if losses >= 2 else 'Không có đủ hai ca FT thua trong tập eval; ghi rõ giới hạn này, không bịa ca thua hoặc sửa eval.'}
Ca thua được xác định bằng điểm từng mẫu thấp hơn baseline (b), không chỉ vì FT
trả lời sai. Ticket, nhãn và output không bị cắt ngắn; xem `qualitative.json` để
đối chiếu mọi mẫu, gồm những mẫu không chọn vào bảng.

{table(qualitative, ['#', 'Ticket', 'Nhãn', '(b) output', '(c) output', 'Điểm b/c', 'Kết quả', 'FT sai'])}

## Kết luận

Quyết định hiện tại là **{'có thể xem xét thử nghiệm triển khai có kiểm soát' if verdict['passed'] else 'chưa triển khai bản fine-tune này'}**
dựa trên cổng đã đo. Lợi ích target chỉ có nghĩa khi được so với prompt tối ưu
trên cùng base model và cùng tập đánh giá. Loss huấn luyện thấp cho biết model
phù hợp dữ liệu train hơn, nhưng không bảo đảm generalization hoặc giữ năng lực
phổ thông. Bởi vậy thứ hạng tác vụ trong autopsy và điểm regression quyết định
kết luận; loss được dùng để chẩn đoán hành vi học và đối chiếu với thứ hạng đó.

Phép đối chứng vị trí giữ gần bằng số tham số và cùng ngân sách step, giúp tránh
nhầm tác động ngân sách với vị trí adapter. Rank attention được điều chỉnh để
đạt điều kiện này, nên kết quả không phải bằng chứng của một phép quét rank độc
lập. Đối chứng LR giữ vị trí và lượng tử hóa cố định, còn QLoRA giữ cấu hình
LoRA và thay lượng tử hóa của base. Số đo VRAM, thời gian và target của từng run
giúp mô tả cái giá cụ thể của lựa chọn đó thay vì lặp lại kỳ vọng trong tài liệu.

Ngay cả khi cổng PASSED, dữ liệu ticket sinh theo mẫu và regression có kích thước
giới hạn chưa đủ để khẳng định an toàn triển khai rộng. Cần kiểm tra thêm dữ liệu
khách hàng đại diện, yêu cầu format thực tế và độ trễ trong môi trường phục vụ.
Nếu FAILED thì giữ nguyên kết quả, giải thích nhóm điểm gây thất bại và thiết kế
một vòng tiếp theo trước khi thấy output mới. Không sửa baseline hay eval để
đổi một thất bại thành chiến thắng. Bằng chứng mask và việc đóng băng mốc là nền
tảng để những kết luận này có thể kiểm tra lại.

## Phản tư cá nhân

{reflection_text}

## Nguồn và phần thưởng

Số liệu lấy từ `mask_proof.json`, `template_check.json`, `token_stats.json`,
`baselines_frozen.json`, `runs.csv`, `training_*.json`, `evaluation_correct.json`,
`verdict.json`, `autopsy.json`, `qualitative.json`. Không có số liệu model mô phỏng.
Chưa làm NB6, dataset riêng, reasoning-trace collapse, rank sweep hoặc HF Hub.
Không commit, không push. Thiếu phản tư hoặc thiếu hai ca thua cần được ghi nhận
riêng khi chấm rubric, dù cổng kỹ thuật có thể đạt.
'''
