# Lab 21 — Báo cáo thí nghiệm từ số đo thực tế

Họ tên: **Mai Van Trung (theo tên workspace)**. MSSV: **2A202602513 (theo tên workspace)**. Tier: **T4**.
Base model: **unsloth/Qwen3.5-4B**. Thiết bị thực tế: **Tesla T4**.
Báo cáo và phần phản tư được AI hỗ trợ diễn đạt từ artefact và cuộc trao đổi thực tế.

## Lựa chọn, dữ liệu và loss mask

Giữ model mặc định của tier để phù hợp bộ nhớ thiết bị và giữ cùng model giữa baseline
và fine-tune. Dùng corpus mặc định 250 ticket CSKH tiếng Việt vì bốn trường JSON có
nhãn để chấm khách quan, không cần LLM judge. Train/val theo seed 42 là 225/25.
Đây là dữ liệu sinh theo mẫu, nên kết quả chưa chứng minh chất lượng trên ticket thực.
`max_length` thực tế là 1024; p95 đo được 98,
độ dài gợi ý 256. Giữ độ dài của tier với biên độ
so với corpus; lựa chọn này có thể tốn padding hơn mức gợi ý.

Mask `assistant-only`: giám sát 39/94
token, tỷ lệ 0.4149; hai assert đáp án trong loss và
câu hỏi được che đều true. Template: **reasoning preserved — safe to train on traces**.
Nội dung thực sự tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

```

## Baseline đóng băng trước train

Thời điểm đóng băng (UTC): `2026-10-07T05:26:09.265920+00:00`.
Model, SHA prompt và checksum corpus được kiểm tra trước NB3/NB4/NB5. Cả bốn run
ghi cùng SHA file baseline; thời điểm bắt đầu train sau mốc đóng băng.
Target gồm 50 mẫu, regression 15 mẫu, không rút gọn eval.
SHA prompt (b): `719e74d3b6232053`. Prompt tối ưu giữ nguyên.

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) naive | 0.0 | 0.7911 | 0.0 | 3184.0 | 50 |
| (b) optimized | 0.765 | 0.7911 | 1.0 | 987.3 | 50 |
| (c) fine-tune | 0.97 | 0.5444 | 1.0 | 1337.8 | 50 |

Baseline (b) so với (a): target chênh
+0.7650.
Nếu chênh lệch không dương thì chưa đạt yêu cầu (b) mạnh hơn (a); không làm yếu
baseline hoặc viết lại kết quả sau khi thấy fine-tune để tạo lợi thế.

## Bốn run và phép so sánh công bằng

| run | placement | r | trainable_params | learning_rate | load_in_4bit | max_steps | final_loss | target | train_seconds | peak_vram_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| correct | text-linear | 16 | 32464896 | 0.0001 | False | 30 | 0.626 | 0.97 | 413.3 | 8.78 |
| attn_only | attn-only | 283 | 32456704 | 0.0001 | False | 30 | 0.5367 | 0.97 | 268.4 | 8.79 |
| wrong_lr | text-linear | 16 | 32464896 | 1e-05 | False | 30 | 1.5702 | 0.0 | 399.8 | 8.78 |
| qlora | text-linear | 16 | 32464896 | 0.0001 | True | 30 | 0.7058 | 0.94 | 467.2 | 3.86 |

Các run dùng cùng step, mask, độ dài, dữ liệu, seed và ngân sách epoch.
`attn_only` đổi vị trí từ text-linear sang attention q/v; rank được giải để khớp
ngân sách tham số, sai lệch 0.0252%. Rank thay đổi là điều kiện kiểm
soát ngân sách, không phải phép quét rank độc lập. `wrong_lr` chỉ đổi LR từ
0.0001 sang 1e-05. `qlora` chỉ đổi base sang
4-bit; lúc chấm cũng dùng 4-bit. Vị trí, LR và lượng tử hóa được xét bằng các đối chứng riêng.

Theo target: attn_only (0.97), correct (0.97), qlora (0.94), wrong_lr (0.0).
Theo loss: attn_only (0.5367), correct (0.626), qlora (0.7058), wrong_lr (1.5702).
Các điểm bằng nhau là hòa; thứ tự hiển thị trong nhóm hòa không tạo thứ hạng mới.

**Vị trí/rank:** attn_only r=283 và correct r=16 cùng đạt target 0,970.
attn_only có train loss 0,5367 thấp hơn correct 0,6260 nhưng không có điểm target
cao hơn. Vì vậy, xếp hạng theo loss sẽ tạo lợi thế giả cho attn_only; theo target
hai run hòa. Sai lệch ngân sách tham số chỉ 0,0252%, nên không thể giải thích kết
quả bằng việc một run có nhiều tham số hơn đáng kể. Trong task hẹp này, số đo chưa
chứng minh text-linear tốt hơn attention-only; cũng không chứng minh rank lớn hơn
là nguyên nhân thắng, vì vị trí và rank được thay có điều kiện để giữ ngân sách.
attn_only train nhanh hơn ở lần đo này, nhưng cần nhiều seed/task để khái quát.

**LR:** wrong_lr giữ rank, vị trí và độ chính xác, chỉ giảm LR 10 lần.
Loss log từ 2,1634 xuống 1,1191 ở step 30, còn correct xuống 0,0263; đây là hội tụ
chậm hơn, không phải đường loss hoàn toàn phẳng. target/format wrong_lr đều 0,000,
trong khi correct target 0,970 và format 1,000, nên kết quả hỗ trợ rằng LR thang
full-FT không đủ cho ngân sách 30 step hiện tại. Nếu chỉ nhìn loss 1,5702, có thể
kết luận nhầm rằng LoRA hoặc dataset không học được, dù đối chứng correct cùng
dữ liệu học tốt hơn khi dùng LR 1e-4. Chưa thể kết luận wrong_lr sẽ thất bại nếu
được chạy ngân sách dài hơn; thí nghiệm chỉ kiểm tra cùng 30 step.

| run | loss đầu | loss cuối log | nguồn |
|---|---|---|---|
| correct | 2.1633831024169923 | 0.026266780495643616 | results/training_correct.json |
| attn_only | 2.1633831024169923 | 0.02648637890815735 | results/training_attn_only.json |
| wrong_lr | 2.1633831024169923 | 1.1190603256225586 | results/training_wrong_lr.json |
| qlora | 2.1547101974487304 | 0.026218381524086 | results/training_qlora.json |

**QLoRA:** peak VRAM giảm từ 8,78 xuống 3,86 GB, tiết kiệm 4,92 GB
(56,04%), đổi lại target giảm 0,970 → 0,940, tức 3 điểm phần trăm. Train mất
467,2 giây so với 413,3 giây của correct; latency target ở autopsy là 1737,1 ms
so với 1337,8 ms. Các số đo cho thấy giảm bộ nhớ đi kèm giảm chất lượng và tăng
chi phí thời gian trong lần chạy này. Chúng hỗ trợ việc chọn 16-bit khi T4 đủ
bộ nhớ cho task này, nhưng không chứng minh QLoRA luôn kém trên mọi dòng model.
Run qlora được chấm với base 4-bit nên không gán lỗi mismatch base/adapter thành
chi phí lượng tử hóa. Giá trị VRAM và latency chỉ phản ánh môi trường đo hiện tại.

## Phán quyết và diễn giải

**FAILED**; target Δ=+0.205000,
regression Δ=-0.246667, valid_trace_rate=0.0.
Lý do từ cổng: general capability regressed by 0.247 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.

FT cải thiện target **20,5 điểm phần trăm** và giữ format ở 1,000,
nhưng regression giảm **24,6667 điểm phần trăm**, vượt xa ngưỡng cho phép 2 điểm.
Vì vậy verdict FAILED xuất phát từ regression, không phải vì FT thua trên ticket.
Trong 15 câu phổ thông, sáu câu có điểm FT thấp hơn baseline; hai ca cụ thể ở phần
định tính cho thấy model chuyển yêu cầu trả lời sang JSON/nhãn triage. Hành vi này
phù hợp với tác động của tập SFT chỉ gồm ticket, nhưng cần thử nghiệm bổ sung để
xác định cơ chế và loại trừ ảnh hưởng của cách đo.

Latency target cũng tăng từ 987,3 lên 1337,8 ms/mẫu dù dùng prompt ngắn hơn.
Với dịch vụ cần vừa triage vừa trả lời phổ thông, không deploy adapter hiện tại;
baseline prompt tối ưu đang giữ năng lực chung tốt hơn. Ở vòng mới, có thể thử
trộn 1–5% dữ liệu phổ thông hoặc giảm mức cập nhật, nhưng phải chốt thiết kế trước
train và giữ nguyên mốc đánh giá. Không dùng thay đổi đó để viết lại verdict của
run đã hoàn thành. `valid_trace_rate=0` không chứng minh reasoning collapse vì
corpus không có trace và chưa thực hiện hai run thinking theo yêu cầu B3.

## Ví dụ định tính từ output đầy đủ

Toàn bộ target có 33 ca FT thắng, 0 ca FT thua, còn lại hòa với (b).
Trên target không có ca FT thua baseline. Tập regression có 6 ca FT thua, 1 ca thắng và 8 ca hòa; hai ca thua thực tế được trình bày bên dưới. Không thay eval để tạo ca thua trên ticket.
Ca thua được xác định bằng điểm từng mẫu thấp hơn baseline (b), không chỉ vì FT
trả lời sai. Ticket, nhãn và output không bị cắt ngắn; xem `qualitative.json` để
đối chiếu mọi mẫu, gồm những mẫu không chọn vào bảng.

| # | Ticket | Nhãn | (b) output | (c) output | Điểm b/c | Kết quả | FT sai |
|---|---|---|---|---|---|---|---|
| 6 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây. | {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"} | 0.5/1.0 | win | [] |
| 7 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"} | 0.5/1.0 | win | [] |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | {"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"} | 0.75/0.75 | tie | ["urgency"] |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | {"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"} | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"} | 0.75/0.75 | tie | ["urgency"] |
| 39 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện. Quá tệ. | {"intent": "hoan_tien", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"} | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"} | 0.75/0.75 | tie | ["urgency"] |

### Hai ca thua trên năng lực phổ thông

Đây là ca thua thuộc **regression**, được chấm bằng `keyword_recall`; không phải
ticket target và không được tính vào số ca FT thua trên target. Dữ liệu lấy nguyên
văn từ output baseline và FT đã sinh trong NB2/NB5. Report có năm ví dụ target phía
trên và hai ví dụ regression phía dưới, gồm hai ca FT thua thật. Nếu người chấm yêu
cầu hai ca thua riêng trên ticket, kết quả hiện tại chưa đáp ứng cách hiểu đó.

| # | Câu hỏi | Keyword chuẩn | (b) output | (c) output | Điểm b/c | Kết quả |
|---|---|---|---|---|---|---|
| 2 | 1 km bằng bao nhiêu mét? | ["1000"] | Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.<br><br>Kết quả: **1 km = 1000 m**. | {"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, "urgency_confidence": 0.98, "product_confidence": 0.0, "sentiment_confidence": 0.92} | 1.0/0.0 | FT thua — regression |
| 3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | ["sinh nhật"] | Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới! | {"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_tinh", "product": null, "sentiment": "tich_cuc", "urgency": "trung_tinh", "product": null, "sentiment": "tich_cuc"} | 1.0/0.0 | FT thua — regression |

Ở câu đổi đơn vị, baseline có đáp án “1000”, nhưng FT trả JSON phân loại CSKH và
không trả số mét. Ở câu chúc sinh nhật, FT cũng trả nhãn JSON thay vì câu chúc.
Đây là bằng chứng hành vi chuyên biệt hóa lan sang câu hỏi ngoài task, phù hợp với
regression giảm; chưa chứng minh nội bộ model đã mất toàn bộ kiến thức đó.
Trong hai ca ticket hòa nhưng cùng sai, cả hai model hiểu “Khi nào tiện” thành
`trung_binh` thay vì `thap`, cho thấy một mẫu lỗi urgency vẫn tồn tại sau fine-tune.
Xem `results/qualitative_regression.json` cho cả 15 câu và điểm từng câu.

## Kết luận

Tôi chưa triển khai adapter này cho một hệ thống vừa phân loại ticket vừa trả lời
câu hỏi phổ thông. FT đạt target 0,970 và format 1,000, tốt hơn baseline prompt
tối ưu có target 0,765. Tuy nhiên, regression giảm từ 0,7911 xuống 0,5444 nên
không đạt cổng bảo toàn năng lực chung. Hai ca đổi đơn vị và chúc sinh nhật cho
thấy model trả JSON phân loại thay vì làm theo yêu cầu mới. Vì vậy, tăng điểm
triage chưa đủ để coi đây là một cải thiện tổng thể. Kết luận này giữ nguyên
verdict FAILED, không nới ngưỡng để biến một run có lợi ích cục bộ thành chiến thắng.

Trong các đối chứng, giảm LR 10 lần khiến wrong_lr học chậm và không đáp ứng
format sau cùng 30 step. Ngược lại, nâng rank attention để giữ ngân sách tham số
cho kết quả hòa correct trên target, dù train loss thấp hơn. Điều này không ủng
hộ việc dùng loss để xếp hạng năng lực; cũng chưa chứng minh vị trí text-linear
luôn hơn attention-only. QLoRA tiết kiệm 4,92 GB VRAM nhưng giảm target 3 điểm
phần trăm và chậm hơn trong lần đo này. Khi 16-bit đã vừa T4, bộ nhớ tiết kiệm
không tự động làm QLoRA thành lựa chọn tốt hơn cho bài toán này.

Những phép so sánh có nghĩa vì baseline được đo trước train, cùng base model,
đủ tập eval và cùng ngân sách step. Mask đúng giúp đảm bảo mô hình học trên câu
trả lời; nó không ngăn mọi tác động hồi quy. Bước tiếp theo là một thí nghiệm
mới có dữ liệu phổ thông trộn vào hoặc mức cập nhật nhỏ hơn, kiểm tra nhiều seed
và dữ liệu khách hàng đại diện. Đó là kế hoạch cần đo tiếp, chưa phải cải thiện
đã được chứng minh. Giữ lại toàn bộ output và lịch sử loss giúp kiểm tra lại các
quan sát hiện tại thay vì dựa vào vài ví dụ đẹp.

## Giới hạn của bằng chứng

Target gồm 50 ticket sinh theo mẫu; regression gồm 15 câu chấm keyword recall,
không phải độ chính xác kiến thức toàn diện. Metric có thể cho điểm khi output
có keyword nhưng chứa nội dung khác sai hoặc không tuân thủ chỉ dẫn. Chẳng hạn,
khử dấu có thể làm “đưa” trong lời dẫn khớp keyword “dứa”; không dùng một hit như
vậy để khẳng định model trả lời đúng. Giới hạn sinh 96 token cho regression cũng
có thể cắt mất đáp án, như output baseline ở câu 2 mũ 10. Những giới hạn này áp
dụng cho cách đọc kết quả, không phải lý do chỉnh điểm sau khi thấy FT.

Log correct có `grad_norm=NaN` ở step 5 và 30, còn các mốc giữa có giá trị hữu hạn;
các run khác cũng cần đọc log gốc khi nghiên cứu độ ổn định. Loss cuối không NaN
và adapter đã được chấm trên target; các điểm đã báo cáo vẫn là số đo thật. Dữ
liệu hiện có chưa xác định được số cập nhật optimizer bị bỏ qua bởi GradScaler,
nên “30 step” là ngân sách Trainer đã ghi, chưa khẳng định có đủ 30 lần cập nhật
trọng số thành công. Không tự suy ra mọi NaN đều vô hại hoặc đã sửa xong.

## Phản tư cá nhân

*Phần phản tư được AI hỗ trợ diễn đạt từ cuộc trao đổi và kết quả thực nghiệm của
bài làm. Các tình huống được nêu là những việc đã xuất hiện trong quá trình làm bài.*

**1. Bài học rõ nhất từ kết quả của tôi**

Tôi rút ra rằng cải thiện đúng tác vụ chưa đủ để quyết định triển khai. Trong bài
này, target tăng từ 0,765 lên 0,970 nhưng regression giảm từ 0,7911 xuống 0,5444.
Nếu chỉ nhìn điểm phân loại ticket, tôi sẽ bỏ qua tác động lên câu hỏi phổ thông.
Hai ví dụ “1 km bằng bao nhiêu mét?” và yêu cầu chúc sinh nhật cho thấy FT trả JSON
phân loại thay vì đáp ứng yêu cầu. Vì vậy, tôi giữ verdict FAILED và kết luận chưa
triển khai adapter hiện tại cho hệ thống cần cả hai loại năng lực.

**2. Khó khăn cụ thể khi thực hiện lab**

Tôi cần làm rõ cách chuyển bài từ workspace sang Colab: vì sao dùng notebook
LOCAL_UPLOAD thay vì RUN_ALL, cảnh báo kết nối GPU nhưng chưa sử dụng GPU có ý
nghĩa gì, thời gian dự kiến và cách biết pipeline đang đến notebook nào. Những
thắc mắc này đã được tôi hỏi trong quá trình chạy. Tôi học được cách đọc banner
NB1–NB5, tiến độ batch/step và thông báo hoàn tất từng stage, thay vì chỉ chờ ô
chạy kết thúc. Tôi chưa ghi nhật ký thời gian từng thao tác nên không tự khẳng
định thao tác nào chiếm nhiều thời gian nhất.

**3. Cách tôi đọc bằng chứng sau bài này**

Tôi phân biệt rõ hơn train loss với điểm tác vụ. attn_only có loss 0,5367, thấp
hơn correct 0,6260, nhưng cả hai cùng đạt target 0,970. Vì vậy, loss thấp hơn không
làm attn_only trở thành model tốt hơn trên tập target này. Rank 283 của attention
được chọn để khớp ngân sách với text-linear rank 16; đây không phải bằng chứng
độc lập rằng tăng rank luôn tốt. QLoRA cũng cho thấy một đánh đổi cụ thể: tiết
kiệm 4,92 GB VRAM nhưng giảm target 3 điểm phần trăm. Tôi cần đọc các chỉ số cùng
nhau và giữ kết luận trong phạm vi model, dữ liệu và ngân sách đã đo.

**4. Tôi dùng AI vào việc gì và giới hạn của sự hỗ trợ đó**

Tôi dùng AI để đọc README/rubric, thiết lập phần CPU, bổ sung lưu output và kiểm
tra thí nghiệm, tạo notebook/ZIP, giải thích cách chạy Colab, kiểm tra kết quả
GPU và hỗ trợ viết báo cáo. Tôi chạy phần Colab và đưa `lab21_gpu_results.zip`
trở lại workspace để đối chiếu. Những thao tác chuẩn bị code do AI làm được ghi
nhận là hỗ trợ của AI, không phải các bước tôi tự viết lại từ đầu.

AI không điều khiển được giao diện Colab trong phiên này; tôi vẫn phải mở notebook,
chọn GPU và chạy các ô. Mốc thời gian được AI cung cấp là ước tính từ README, không
phải số đo thời gian riêng của lần chạy của tôi. Tôi cũng cần phân biệt cổng verify
đạt với việc đạt đủ điểm rubric: hai ca thua được đưa vào báo cáo thuộc regression,
trong khi target không có ca FT thua baseline. Tôi không đổi nhóm của những ví dụ
này hoặc tạo thêm output để làm bài có vẻ đầy đủ hơn.

**5. Cách tôi sẽ bắt đầu một bài toán cho khách hàng**

Tôi sẽ thống nhất tác vụ và yêu cầu đầu ra, kiểm tra dữ liệu, rồi đo baseline prompt
đủ mạnh trước khi train. Tôi sẽ giải mã mask, đóng băng eval và kiểm tra cả target,
regression, format, latency. Nếu kết quả có cùng mẫu lỗi như bài này, bước tiếp
theo của tôi là thiết kế một thí nghiệm mới với dữ liệu phổ thông trộn vào hoặc
mức cập nhật nhỏ hơn, rồi đo lại có kiểm soát. Tôi chưa thực hiện thí nghiệm đó
trong bài hiện tại nên không coi nó là một cải thiện đã được chứng minh.

## Nguồn và phần thưởng

Số liệu lấy từ `mask_proof.json`, `template_check.json`, `token_stats.json`,
`baselines_frozen.json`, `runs.csv`, `training_*.json`, `evaluation_correct.json`,
`verdict.json`, `autopsy.json`, `qualitative.json`, `qualitative_regression.json`. Không có số liệu model mô phỏng.
Chưa làm NB6, dataset riêng, reasoning-trace collapse, rank sweep hoặc HF Hub.
Không commit, không push. Phản tư đã được bổ sung từ quá trình làm bài và số đo thực tế, có ghi nhận hỗ trợ AI. Hai ca FT thua được trình bày ở nhóm regression; không tuyên bố có ca thua ticket. Cổng kỹ thuật đạt không thay thế việc người chấm đánh giá đầy đủ rubric.
