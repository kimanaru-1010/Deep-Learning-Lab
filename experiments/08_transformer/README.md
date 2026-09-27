# Transformer

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 09 — Transformer

**Goal:** Ghép causal character language model nhỏ chạy CPU.

**Concepts:** Embedding/position; multi-head; LayerNorm; feed-forward; residual; CE; perplexity.

**Files to implement:** src/minidl/nn/transformer.py, src/minidl/nn/normalization.py, models/transformer.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_transformer.py tests/nn/test_normalization.py tests/nn/test_embedding.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/08_transformer/train.py
```

**Definition of Done:** Block gradient và causality pass; text CE giảm; validation/perplexity/samples/checkpoint được tạo.

Các tham số CLI: `python experiments/08_transformer/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

