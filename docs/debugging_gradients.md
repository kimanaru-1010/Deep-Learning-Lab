# Debugging gradients

Luôn bắt đầu bằng tensor nhỏ, float64, fixed seed. In shape/value tại từng ranh giới
layer, không chỉ nhìn loss cuối mạng. Chạy known-value test trước finite differences.

| Triệu chứng | Kiểm tra tiếp theo |
|---|---|
| Gradient sai dù forward đúng | Một operation, upstream cố định; kiểm tra từng input và parameter riêng. |
| Shared variable có gradient thiếu | Graph phân nhánh `x*x + 3*x`, shared intermediate và nhiều parent. |
| Gradient nhân lên theo số lần gọi | Phân biệt leaf accumulation với intermediate scratch gradient; rebuild graph mỗi batch. |
| Sai shape sau broadcasting | Bias dùng qua nhiều batch; giảm đúng các trục từng được broadcast. |
| Gradient lớn dần qua batch | Kiểm tra `zero_grad()` trước backward; xem có tích lũy ngoài ý muốn không. |
| Learning rate quá lớn | Loss dao động/NaN; quan sát actual update và parameter norm, giảm LR để chẩn đoán. |
| Learning rate quá nhỏ | Gradient có nhưng weight thay đổi không đáng kể; xem watch before/after. |
| NaN / Inf | Tìm operation đầu tiên tạo non-finite; logits/log/exp và optimizer denominator. |
| Exploding gradient | So sánh norm qua độ dài chuỗi/layer; kiểm tra chain rule và scaling trước khi thêm clipping. |
| Vanishing gradient | Nhìn per-layer grad và saturation; thử mạng/chuỗi ngắn, kiểm tra init. |
| Khởi tạo không phù hợp | Xem variance activation theo depth; học Xavier/He thay vì sửa bằng hằng số tùy tiện. |
| Sai batch averaging | So sánh batch lặp mẫu với batch gốc; loss là mean trên trục nào, có chia hai lần không? |
| Shape mismatch | Viết bảng shape X/W/b/output; kiểm tra NCHW và final-axis features. |
| SGD đúng nhưng Adam sai | Test hai bước biết trước; state theo parameter, counter, bias correction và epsilon. |
| Eval thay đổi kết quả/weights | `eval()` phải tắt cập nhật running stats; không gọi backward/step trong evaluate. |
| ReLU/pooling gradcheck thất bại tại điểm gãy | Chọn dữ liệu không ở kink/tie, test convention riêng. |

Quy trình khi 32 mẫu không overfit:

1. Chốt seed và chính batch đó, không shuffle sang dữ liệu khác.
2. In logits và loss hữu hạn; xác nhận labels trong [0,9].
3. Xem mọi trainable parameter có `.grad`, không bị detach vì dùng `.data` trong graph.
4. Gradcheck input lẫn weight/bias của từng layer.
5. Watch một weight trước/gradient/sau step; kiểm tra LR và zero_grad.
6. Test optimizer độc lập với giá trị tính tay; giảm kiến trúc về một layer nếu cần.
7. Chỉ tăng dữ liệu khi loss trên batch này giảm rõ và accuracy tiến gần memorization.

Monitoring chỉ quan sát; không dùng công cụ này tự sửa NaN hay thay gradient bằng
finite differences. Khi cần kiểm tra optimizer state, mở NPZ với `allow_pickle=False`.
