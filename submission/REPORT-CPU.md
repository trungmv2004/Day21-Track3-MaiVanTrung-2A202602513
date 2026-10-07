# Lab 21 — Bằng chứng CPU và trạng thái thí nghiệm

Ngày thực hiện: **07/10/2026** (Asia/Bangkok). Tên và MSSV suy từ thư mục dự án:
**Mai Van Trung — 2A202602513**, chưa được người học xác nhận. AI assistant soạn
báo cáo từ artefact thực tế; người học cần bổ sung phản tư cá nhân.

**Hoàn thành NB1 và kiểm tra CPU; chưa hoàn thành core NB1–NB5.** Chưa có baseline,
adapter được train hoặc verdict. Không thể kết luận fine-tune thắng hay thua.
`verify --smoke` kiểm tra môi trường và code; cổng `verify` đầy đủ vẫn báo FAIL
khi thiếu artefact GPU. Đây là báo cáo tiến độ, chưa phải bài nộp hoàn chỉnh.

## Lựa chọn và lý do

Chọn tier **CPU**, tokenizer **Qwen/Qwen3.5-0.8B**, model mặc định của CPU trong
README. Máy Windows không cung cấp `nvidia-smi` hoặc driver `nvcuda.dll` tại đường
dẫn hệ thống đã kiểm tra. NB1 chỉ tải tokenizer, không tải trọng số hay chạy inference.
Khi chuyển T4, phải chạy lại NB1 với model T4 và dùng cùng model đó cho NB2–NB5.

Giữ corpus mặc định **250 ticket CSKH tiếng Việt → JSON** (`intent`, `urgency`,
`product`, `sentiment`) vì có nhãn chấm khách quan, không cần LLM judge. Không thay
dataset hoặc prompt. Dữ liệu sinh theo mẫu nên chưa đại diện mọi ticket khách hàng.

| Thành phần | Số đo / cấu hình | Nguồn |
|---|---|---|
| Corpus / train / validation | 250 / 225 / 25 mẫu | `results/cpu_audit.json` |
| Seed | 42 | `results/cpu_audit.json` |
| Target / regression | 50 / 15 mẫu | `results/cpu_audit.json` |
| Mask | `assistant-only` | `results/mask_proof.json` |
| Tier `max_length` | 512 | `results/cpu_audit.json` |
| p50 / p95 / p99 / max | 93 / 98 / 100 / 101 token | `results/token_stats.json` |
| Mean | 93,1 token | `results/token_stats.json` |
| `suggested_max_length` | 256 token | `results/token_stats.json` |
| Epochs | 2, cấu hình chưa chạy train | `.env` |

Giữ `max_length=512` theo cấu hình tier của README. p95=98 và max=101 cho thấy 256
đã đủ cho corpus hiện tại; 512 dành thêm biên độ nhưng có thể tốn padding hơn khi
train. Audit xác nhận không mẫu nào bị cắt ở 512. Chưa khẳng định 512 là tối ưu.

## Bằng chứng loss mask và template

NB1 dùng tokenizer thật để giải mã token có label khác `-100`:

| Kiểm tra trên mẫu đầu | Kết quả |
|---|---|
| Token giám sát / tổng số | 37 / 94 |
| `supervised_fraction` | 0,3936 |
| `answer_is_supervised` | true |
| `question_is_masked` | true |

Phần được tính loss, lấy từ `results/mask_proof.json`:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chứng `everything` ở NB1 giám sát 94/94 token, gồm cả câu hỏi và system prompt.
Loss giảm trong chế độ đó chưa chứng minh model học đúng nhiệm vụ trả lời.

`scripts/audit_cpu.py` kiểm tra **250/250** mẫu: toàn bộ câu trả lời nằm trong loss,
ticket được che, EOS nằm trong loss, loss không rỗng, không truncation, prompt đánh
giá là tiền tố của chuỗi train. Không phát hiện nhãn sai schema hay output lệch
label. Không có input trùng chính xác giữa train/validation hoặc corpus/target.
Kiểm tra này chưa loại trừ tương đồng ngữ nghĩa giữa ticket sinh theo mẫu.

`results/template_check.json` có verdict **reasoning preserved — safe to train on
traces**: nội dung thử nghiệm trong `<think>` được giữ sau render. Corpus hiện tại
chứa đáp án JSON, không có reasoning trace để train; chưa làm thưởng B3.

Chạy `scripts/check_mask_agreement.py` cho thấy template thiếu
`{% generation %}`. API mask assistant của tokenizer trả **0/31** token, còn labkit
giám sát **9/31** token gồm đáp án và EOS. Script trả exit 1 để cảnh báo khác biệt.
TRL chưa được cài ở môi trường CPU, nên nhánh trainer TRL chưa được kiểm tra thực tế.
Code train hiện có dùng `data.to_training_dataset()` với label tính sẵn từ
`build_example()`, đúng mask NB1 đã chứng minh, không dựa vào mask tokenizer rỗng.

## Tính toàn vẹn dữ liệu và kiểm tra

Git có `core.autocrlf=true`, làm checkout Windows dùng CRLF dù Git lưu LF.
Đã xác nhận đổi riêng CRLF về LF khôi phục đúng **cả bốn checksum gốc**, rồi mới
ghi lại file. Thêm `.gitattributes` với `data/*.jsonl text eol=lf` để giữ LF khi
checkout. Không thay ticket, nhãn hoặc checksum tham chiếu; không khai báo custom
dataset để bỏ qua lỗi. Holdout chỉ được kiểm tra checksum, không dùng phân tích.

| File | SHA-256 rút gọn sau khôi phục LF |
|---|---|
| `train_seed.jsonl` | `2a58fe45d7315da0` |
| `eval_target.jsonl` | `2991050fefb848c8` |
| `eval_regression.jsonl` | `9c6ba90ea86e28fb` |
| `holdout_secret.jsonl` | `be3dd38d058871a5` |

NB1 chạy với Python **3.11.9**, transformers **5.19.0**, tokenizers **0.23.2**,
Jinja2 **3.1.6**, jupytext **1.19.6**, pytest **9.1.1**. Thư viện cài trong `.venv`;
phiên bản đầy đủ lưu ở `submission/requirements-cpu.lock.txt`. PyTorch CPU được
cài thêm để chạy các test tùy chọn, không dùng để thay thế thí nghiệm GPU.
Test suite ở lần CPU đầu: **119 test đạt, không có test bỏ qua**. `pip check` không
phát hiện dependency hỏng. Kết quả các cổng lưu ở `results/verification_cpu.json`.

## Phần chưa đo: baseline, đối chứng, phán quyết

Đã bổ sung cơ chế lưu output đầy đủ của baseline từ NB2, kiểm tra fingerprint của
model/prompt/corpus trước train và chấm, ghi thời điểm train cùng SHA baseline,
lưu lịch sử loss từng run, và so sánh định tính từng mẫu với baseline (b).
`scripts/write_gpu_report.py` tạo báo cáo từ artefact sau NB5, kiểm tra lại số đo
và giữ báo cáo CPU. Những thay đổi này chuẩn bị cho phép đo; chưa tạo thêm kết quả
model hoặc làm hoàn thành các mục rubric yêu cầu thực nghiệm GPU.
Kết quả kiểm tra phần bổ sung lưu ở `results/preparation_checks.json`.

| Run | target | regression | format | latency |
|---|---|---|---|---|
| (a) base + naive prompt | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| (b) base + optimized prompt | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| (c) LoRA fine-tune | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Chưa có `baselines_frozen.json`. Checksum đúng không đồng nghĩa baseline đã được
đo và đóng băng. Cần hoàn tất NB2 trước NB3, dùng đầy đủ eval, giữ cùng base model,
và kiểm tra (b) thực sự mạnh hơn (a) trước khi đưa ra kết luận.

**Vị trí so với rank:** chưa có target hoặc loss của `attn_only` và `correct`, nên
chưa thể xếp hạng. NB4 phải dùng `matched_rank()` để trainable params lệch dưới 5%,
cùng số step. NB5 chấm target mới có bằng chứng vị trí adapter là đòn bẩy hay không.

**Learning rate:** code cấu hình LoRA 1e-4 và đối chứng `wrong_lr` 1e-5. Đây là cấu
hình dự kiến, không phải số đo train. Chưa có đường loss nên không thể suy ra tốc
độ hội tụ hoặc chất lượng cuối; chỉ nhìn loss có thể dẫn tới xếp hạng sai theo proxy.

**QLoRA:** chưa có VRAM, thời gian hoặc target. Không thể suy ra mức tiết kiệm VRAM
và mức giảm chất lượng từ hướng dẫn phần cứng. Run phải được train/chấm với base
4-bit, cùng số step và các cấu hình còn lại của `correct`.

**Phán quyết chưa xác định**, khác với verdict FAILED đã đo. Chưa có output sinh
để so sánh target, regression, format, latency, vì vậy chưa có bằng chứng deploy
hay khẳng định fine-tune không hiệu quả. Sau NB5, cần lấy delta và `passed` từ
`verdict.json`, đối chiếu `autopsy.json`, và giải thích tại sao thắng/thua. Nếu target
cải thiện nhưng regression tụt qua cổng, đó là lý do không deploy. Nếu baseline
prompt đã tốt, cần xem lợi ích fine-tune có đủ bù chi phí train và vận hành adapter.
Đây là nguyên tắc đọc kết quả sắp tới, chưa phải kết luận từ run hiện tại. Cổng phải
được giữ nguyên, kể cả khi một verdict FAILED khiến kết quả không như mong đợi.

**Định tính chưa hoàn thành:** chưa có prediction (b)/(c), nên chưa thể viết 5 ca
so sánh hoặc gán 2 ca FT thua. Phải lấy output thật từ `qualitative.json` sau NB5,
đối chiếu nhãn và phân tích mẫu lỗi. Không dùng nhãn đúng làm prediction giả.

## Kết luận và điều rút ra

Hiện chưa có căn cứ để deploy một bản fine-tune, bởi chưa có adapter được huấn luyện
và chưa so sánh với baseline đã tối ưu prompt. Phần CPU trả lời được câu hỏi đầu
tiên của README: loss mask labkit thực sự giám sát câu trả lời và che câu hỏi.
Bằng chứng không chỉ là một cờ cấu hình; token được giải mã thành JSON và EOS,
và kiểm tra toàn bộ corpus đều thỏa điều kiện đó. Đây là điều kiện cần, chưa đủ
để bảo đảm model sau train tốt hơn. Model vẫn có thể học quá khớp, trả JSON sai
hoặc suy giảm năng lực trả lời phổ thông dù mask đúng.

Một phát hiện có quan hệ nhân quả rõ là checksum phụ thuộc byte: Windows đổi xuống
dòng khiến cổng báo dữ liệu bị sửa dù ticket và nhãn không đổi. Khôi phục LF làm
checksum trở về giá trị gốc; quy tắc Git giúp tránh lỗi lặp lại khi checkout.
Phát hiện khác là không thể thay mask labkit bằng mask assistant tokenizer vô điều
kiện. Template thiếu generation marker dẫn đến mask rỗng; nếu dựng batch từ mask
đó thì không còn đáp án để tính loss. Giữ label đã kiểm chứng trong train là điều
kiện tiên quyết trước khi bàn đến tăng rank hoặc chỉnh LR.

Đòn bẩy nào quyết định chất lượng cuối cùng vẫn phải được đo trên GPU. Bốn run cần
cùng ngân sách step; đối chứng vị trí cần khớp ngân sách tham số; baseline phải đóng
băng trước train; thứ hạng phải lấy từ target. Hoàn thành các bước này cùng phân
tích ca thua mới trả lời được câu hỏi thứ hai của README có căn cứ. Báo cáo này
ghi tiến độ và bằng chứng có thật, chưa đủ để nộp như bài core hoàn chỉnh.

Ba điều rút ra: audit cả corpus phát hiện được truncation mà một mẫu chưa phản ánh;
checksum cần xuống dòng ổn định; prompt alignment cần kiểm tra riêng vì mask đúng
chưa bảo đảm train/eval hỏi cùng nhiệm vụ. Bước tiếp theo là NB1–NB5 trên GPU theo
`RUN-GPU.md`, rồi cập nhật báo cáo từ số đo mới và bổ sung reflection cá nhân.
Chưa thực hiện NB6/B1–B5. Không commit, không push, không đăng adapter bên ngoài.
