# Attention

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 08 — Attention

**Goal:** Tự suy ra attention và multi-head projections.

**Concepts:** Q/K/V; QK^T; scaling sqrt(d_k); softmax; allowed-position mask; weighted sum.

**Files to implement:** src/minidl/nn/attention.py, src/minidl/core/ops.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_attention.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/07_attention/run.py
```

**Definition of Done:** Known uniform attention, causal mask và Q/K/V/MHA gradients pass; fully masked row báo ValueError.

Các tham số CLI: `python experiments/07_attention/run.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

