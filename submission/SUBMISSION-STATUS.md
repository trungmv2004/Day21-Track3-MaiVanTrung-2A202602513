# Trạng thái bài Lab 21 sau nhập kết quả GPU

Đã nhập `lab21_gpu_results.zip` và giữ nguyên ZIP gốc. Code trong ZIP khớp bản đã
gửi đi; corpus giữ checksum gốc. Chỉ nhập số đo, adapter chính và báo cáo; không
dùng code trong ZIP để thay nguồn dự án. Kết quả import lưu ở
`results/gpu_import_check.json`. Bản trước import được sao lưu trong `.cache/imports/`.

## Đã hoàn thành phần thực nghiệm core

- NB1: mask proof và template đạt; p95=98, max_length=1024 có giải thích.
- NB2: 50 target và 15 regression; baseline b=0,765 > a=0,000, đóng băng trước train.
- NB3: adapter correct có 32.464.896 tham số, rank 16; có loss, VRAM và log train.
- NB4: đủ correct/attn_only/wrong_lr/qlora, cùng 30 step, ngân sách attn_only lệch 0,0252%.
- NB5: đủ target/regression/format/latency và autopsy. Verdict FAILED do regression
  giảm 0,246667; giữ kết quả này. FAILED được rubric chấp nhận nếu phân tích đúng.
- Report: bổ sung phân tích cụ thể, kết luận dài đủ yêu cầu và bảy ví dụ định tính.

## Phần cần người học hoàn thiện

**Phản tư cá nhân (rubric 4.4):** đã bổ sung vào `submission/PERSONAL_REFLECTION.md`,
phần “Phản tư cá nhân” của REPORT và REFLECTION theo yêu cầu của người học.
Nội dung được AI hỗ trợ diễn đạt từ cuộc trao đổi và số đo: cách chạy Colab, đọc
tiến độ, phân biệt loss/target, đánh giá regression và sử dụng AI. Có ghi nhận vai
trò AI; không tự dựng cảm xúc, niềm tin trước bài hoặc thời gian thao tác chưa đo.
Thông tin tên/MSSV trong report được suy từ workspace; kiểm tra lại trước khi nộp.

**Ca thua (rubric 3.4):** 50 ticket có 33 thắng, 17 hòa, 0 thua; 15 câu regression
có 6 thua, 1 thắng, 8 hòa. Report có năm ví dụ ticket và hai ca FT thua regression,
ghi rõ nhóm và scorer. Nếu người chấm chỉ nhận ca thua trên ticket thì tiêu chí
này chưa đạt đầy đủ theo cách hiểu đó. Không sửa eval hoặc bịa ca để đáp ứng.

Không thực hiện NB6/B1–B5. Bản nộp dùng Option A, chỉ cần adapter chính; log và
điểm của cả ba adapter đối chứng vẫn có đầy đủ. Trọng số adapter fp32 làm ZIP lớn
hơn ước tính 5–15 MB trong rubric; không chuyển dtype sau đo chỉ để làm nhỏ bài.

## Kiểm tra và đóng gói

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts/package_submission.py
```

Script kiểm tra model/prompt/dữ liệu, step, ngân sách tham số, output/điểm/verdict,
sau đó chạy cổng verify và ghi `results/verification_gpu.json`. ZIP đầu ra là
`submission/lab21_2A202602513.zip`. Nó không chứa `.env`, cache, metadata Git,
ZIP gốc hoặc artefact CPU cũ. Nếu sửa REPORT/REFLECTION thì chạy lại lệnh để ZIP
chứa bản cuối. Cổng đạt chỉ xác nhận kỹ thuật; chất lượng phản tư và cách chấm ca
thua vẫn do người chấm đánh giá theo rubric.

Không commit, không push.
