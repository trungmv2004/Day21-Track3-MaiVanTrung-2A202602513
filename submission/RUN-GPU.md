# Chạy tiếp phần core trên GPU, không commit hoặc push

Phần CPU đã có trong `results/`; NB2–NB5 còn thiếu. Notebook
`colab/Lab21_LOCAL_UPLOAD.ipynb` chạy bản ZIP workspace mà không cần đưa thay đổi lên
GitHub. ZIP phục vụ chạy tiếp, chưa phải bài nộp hoàn chỉnh.

1. Mở Google Colab, chọn **File → Upload notebook**, mở `colab/Lab21_LOCAL_UPLOAD.ipynb`.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chạy ô upload, chọn `submission/lab21_gpu_workspace.zip` từ máy.
4. Chạy setup: chọn T4, `assistant-only`, 2 epochs, đầy đủ eval, model mặc định T4.
5. Chạy pipeline **NB1 → NB2 → NB3 → NB4 → NB5**. NB1 phải chạy lại vì model T4 khác
   model CPU. README dự kiến khoảng 100–130 phút cho core.
6. Tải ZIP kết quả ở ô cuối, giữ đủ `results/` và `adapters/correct/`.

Ô 4 hiện chạy `scripts/write_gpu_report.py` để tạo `REPORT.md` trực tiếp từ kết quả
GPU; báo cáo CPU được giữ ở `REPORT-CPU.md`. Script từ chối tạo báo cáo nếu thiếu
bất kỳ run nào, ngân sách tham số/step không khớp, baseline bị đổi sau train, hoặc
điểm/verdict lệch output. Output baseline (b) được lưu đầy đủ từ NB2, dùng đối chiếu
từng mẫu với FT tại NB5. Đáp án FT sai chưa chắc là FT thua; ca thua là điểm FT thấp
hơn điểm baseline (b) trên chính mẫu đó.

Nếu gặp thông báo baseline thiếu/mismatch, dừng và kiểm tra cấu hình. NB2 không
cho ghi đè baseline khi workspace đã có adapter; thí nghiệm mới phải dùng workspace
mới. Trong cùng phiên Colab bị ngắt, sửa `STAGES` ở ô 3 để tiếp tục các notebook
chưa hoàn thành. Giữ nguyên model, mask và epoch của lần bắt đầu.

Tạo `submission/PERSONAL_REFLECTION.md` từ trải nghiệm thật của bạn nếu muốn ô 4
ghép phần phản tư vào report. Nếu chưa có, báo cáo ghi rõ còn thiếu phần này.
Bạn có thể chạy lại riêng lệnh sau sau khi bổ sung file:

```bash
python scripts/write_gpu_report.py --name "Mai Van Trung" --student-id "2A202602513" --reflection submission/PERSONAL_REFLECTION.md
python scripts/verify.py
```

Chỉnh tên và MSSV cho đúng thông tin của bạn. Script chỉ đọc file phản tư, không
tự dựng trải nghiệm hoặc cảm xúc. Nếu dataset hiện tại không có đủ hai ca FT thua,
báo cáo sẽ nêu số ca thua thật; không tự tạo ca để đạt tiêu chí rubric.

Sau run, cập nhật `REPORT.md` từ số đo GPU: model/GPU, p95/mask, baseline (a)/(b)/(c),
4 run với cùng step, trainable params khớp, LR/loss/VRAM/thời gian, target từ
`autopsy.json`, verdict/delta, ít nhất 5 ví dụ thật gồm ít nhất 2 ca FT thua.
Nếu không tìm được đủ ca thua, báo rõ thay vì bịa. Bổ sung reflection cá nhân.

Chạy `python scripts/verify.py` sau khi sửa report. Cổng phải đạt trước khi nộp.
Một verdict FAILED đã đo vẫn hợp lệ; thiếu artefact không phải verdict FAILED của
model. Không đo lại NB2 sau khi thấy FT chỉ để làm yếu baseline.

## Chạy lại CPU trên Windows

PowerShell, không cần `make` hoặc activate virtualenv:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path (Get-Location) '.cache'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
.\.venv\Scripts\python.exe scripts/verify.py --smoke
$env:HF_HUB_OFFLINE = '1' # tokenizer đã được tải vào workspace
.\.venv\Scripts\python.exe notebooks/01_data_and_mask.py
.\.venv\Scripts\python.exe scripts/audit_cpu.py
.\.venv\Scripts\python.exe scripts/verify.py
```

Lệnh cuối hiện báo thiếu artefact GPU. Smoke đạt chưa có nghĩa bài đủ để nộp.
Thư viện CPU được pin trong `requirements-cpu.lock.txt`; PyTorch CPU cài thêm để chạy
test tùy chọn, không dùng huấn luyện thay thế GPU.
