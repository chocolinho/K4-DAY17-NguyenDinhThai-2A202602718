# Bước 8 — Phân tích kết quả benchmark

Benchmark chạy ở chế độ offline trên hai bộ dữ liệu. Token được ước lượng heuristic từ độ dài nội dung; recall và response quality cũng được chấm bằng heuristic, nên các số phù hợp để so sánh tương đối trong lab.

## Kết quả

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1.033 | 10.588 | 0,00 | 0,20 | 0 | 0 |
| Standard | Advanced | 1.074 | 16.490 | 0,57 | 0,66 | 428 | 0 |
| Long-Context Stress | Baseline | 177 | 17.187 | 0,00 | 0,20 | 0 | 0 |
| Long-Context Stress | Advanced | 224 | 15.621 | 0,67 | 0,73 | 250 | 1 |

Memory growth là dung lượng hồ sơ được ghi trong lần chạy với thư mục state riêng, tránh cộng dồn dữ liệu từ lần chạy trước.

## Phân tích

**Recall qua session:** Baseline chỉ giữ lịch sử trong thread hiện tại. Câu hỏi recall được mở ở thread mới nên Baseline không còn dữ kiện trước đó, đạt 0,00 ở cả hai bộ. Advanced lưu một số thông tin ổn định vào `User.md`, đạt 0,57 ở Standard và 0,67 ở Stress. Điểm chưa tuyệt đối cho thấy trích xuất facts và xử lý câu hỏi tổng hợp vẫn còn bỏ sót dữ kiện.

**Chi phí ở hội thoại ngắn:** Trong Standard, Advanced xử lý 16.490 prompt tokens, cao hơn Baseline 10.588 khoảng 56%. Advanced cũng sinh 1.074 token so với 1.033, tăng khoảng 4%. Profile và lịch sử được đưa vào ngữ cảnh, nhưng chuỗi hội thoại chưa đủ dài để compact tạo khoản tiết kiệm bù phần chi phí này. Đây là trade-off để đổi lấy khả năng nhớ qua thread mới.

**Tác động của compact ở hội thoại dài:** Stress test kích hoạt một lần compact. Advanced xử lý 15.621 prompt tokens, thấp hơn Baseline 17.187 khoảng 9%. Kết quả cho thấy nén lịch sử đã giảm prompt load, dù mức giảm còn khiêm tốn với cấu hình và độ dài dữ liệu hiện tại. Advanced vẫn sinh nhiều token hơn (224 so với 177); compact chủ yếu tối ưu ngữ cảnh đầu vào, không đảm bảo giảm token đầu ra.

**Tăng trưởng và rủi ro của memory:** Advanced ghi thêm 428 bytes ở Standard và 250 bytes ở Stress; Baseline không tạo hồ sơ dài hạn. Dung lượng nhỏ trong benchmark này, nhưng hồ sơ có thể phình nếu facts tiếp tục được thêm mà không gộp hoặc loại bỏ dữ kiện lỗi thời. Rủi ro đáng chú ý hơn là lưu sai: câu hỏi, câu đùa, thông tin phủ định hoặc địa điểm đi công tác có thể bị nhận nhầm thành fact. Cần xử lý correction và độ tin cậy cẩn thận.

## Giới hạn của phép đo

- `estimate_tokens()` ước lượng từ ký tự, không phải tokenizer của model cụ thể.
- Recall chấm theo chuỗi khớp với thang 0 / 0,5 / 1. `Response quality` được tính từ recall và việc câu trả lời có rỗng hay không, nên chưa đánh giá độ tự nhiên hay đúng ngữ nghĩa.
- Đây là kết quả offline trên tập dữ liệu hiện tại; chưa đại diện cho chất lượng hoặc chi phí khi dùng model thật.

## Kết luận

Advanced tăng rõ khả năng recall qua session nhờ persistent memory, nhưng chịu overhead prompt ở hội thoại ngắn. Ở stress test, compact giảm prompt-token estimate khoảng 9% và được kích hoạt một lần; lợi ích còn hạn chế nhưng đi đúng hướng. Ưu tiên cải thiện tiếp theo là tăng độ chính xác khi trích xuất và cập nhật facts, đồng thời đánh giá chất lượng ngữ nghĩa độc lập với điểm recall.
