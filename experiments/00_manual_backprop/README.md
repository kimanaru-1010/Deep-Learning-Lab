# Manual Backprop

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 00 — Manual Backprop

**Goal:** Hiểu trực tiếp chain rule với ndarray, chưa dùng autograd.

**Concepts:** Y=XW+b; dX/dW/db; ReLU; mean loss; truyền upstream thủ công.

**Files to implement:** experiments/manual.py

**Tests to run:**

```bash
python -m pytest tests/core/test_manual.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/00_manual_backprop/run.py
```

**Definition of Done:** Known-value Linear, ReLU, MSE, CE và numerical gradient input/weight/bias; MLP backward nối đúng.

Các tham số CLI: `python experiments/00_manual_backprop/run.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

