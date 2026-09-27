# Residual Block / ResNet

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 05 — Residual Block / ResNet

**Goal:** Theo dõi gradient qua nhánh skip.

**Concepts:** y=F(x)+x; shared graph; identity skip; BatchNorm train/eval; running statistics.

**Files to implement:** models/resnet.py, src/minidl/nn/normalization.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_resnet.py tests/nn/test_normalization.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/04_resnet/train.py
```

**Definition of Done:** Zero residual là identity; gradient skip đúng; normalization tests pass; ResNet tiny training có update.

Các tham số CLI: `python experiments/04_resnet/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

