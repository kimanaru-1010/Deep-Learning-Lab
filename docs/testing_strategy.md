# Testing strategy

Pipeline bắt buộc:

```text
Implement component → unit/shape test → known numerical test → float64 gradcheck
→ compose model → forward smoke → tiny-batch overfit → subset → full training
→ validation → final held-out test
```

## Hai nhóm test

Hạ tầng ở `tests/smoke`, `tests/data`, `tests/training` chạy độc lập các bài học.
MNIST parser dùng IDX tổng hợp trong thư mục tạm, không cần mạng. Smoke kiểm tra
mọi CLI `--help`, kể cả khi Linear/Conv2D/Transformer chưa implement.

Các test có `@pytest.mark.student` tự xfail chỉ khi ném `StudentTODO`. Cơ chế này
không bắt mọi exception. Sai số numerical, assertion fail, lỗi API hoặc lỗi utility
vẫn làm suite fail. Khi một test hoàn thành nó có thể hiện XPASS; đây là tiến độ,
không cần sửa marker. Dùng `--run-student` để kiểm tra hoàn thành bài một cách nghiêm túc.

```bash
python -m pytest -m "not student" -q
python -m pytest tests/nn/test_linear.py --run-student -q
python -m pytest tests/core/test_tensor.py --run-student -q
```

## Gradient checker

`numerical_gradient(f,x,epsilon=1e-6)` dùng central difference trên bản sao float64.
`f` phải trả scalar; đối với layer output nhiều chiều, giữ cố định upstream vector
và dùng scalar dot product. `check_gradient` trả passed, max absolute error, max
relative error và numerical array. Nó chỉ dùng trong kiểm chứng, không phải backward.

`tests/helpers.py` kiểm tra tất cả input và trainable parameter của một layer, với
upstream cố định không đều. Test LSTM kiểm tra cả hidden và cell. Attention kiểm tra
Q/K/V, causal mask, gradient các projection trong multi-head.

Tránh ReLU ở điểm 0 hoặc pooling ties khi dùng finite differences; test edge convention
riêng. BatchNorm gradcheck dùng training mode với cùng input, không lấy running stats
thay batch stats. Không để stochasticity thay đổi giữa f(x+eps) và f(x-eps).

## Definition of Done

- Layer: shape, numerical forward, gradient, edge cases đều pass; gradients hữu hạn.
- Optimizer: known-value nhiều bước, bỏ qua grad=None, state và checkpoint roundtrip đúng.
- Model: forward/loss/backward chạy, parameters có grad, update thật, overfit batch,
  train loss giảm, validation log, plots, checkpoint dùng lại được.

Nếu tiny-batch overfit fail: **STOP — DEBUG**. Không xem accuracy của một lần forward
là chứng minh training hoạt động. Setup pass không có nghĩa các bài DL đã pass.

Kiểm thử Linux được định nghĩa trong `.github/workflows/tests.yml`. Kết quả local và
giới hạn OS được ghi riêng trong `docs/verification.md`; không suy diễn CI đã chạy.
