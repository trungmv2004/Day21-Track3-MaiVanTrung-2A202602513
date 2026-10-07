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
