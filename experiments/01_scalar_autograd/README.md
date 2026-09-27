# Scalar Autograd

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 01 — Scalar Autograd

**Goal:** Tự xây reverse-mode autodiff trên scalar graph.

**Concepts:** Value/data/grad/parents/local backward; topological ordering; graph phân nhánh; cộng gradient.

**Files to implement:** experiments/scalar.py

**Tests to run:**

```bash
python -m pytest tests/core/test_scalar.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/01_scalar_autograd/run.py
```

**Definition of Done:** Graph ((x*w)+1)^2 với x=2,w=3 cho dx=42,dw=28; shared nodes và zero_grad đúng.

Các tham số CLI: `python experiments/01_scalar_autograd/run.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

