# CNN + MNIST

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 04 — CNN + MNIST

**Goal:** Hiểu spatial weight sharing và gradient qua pooling.

**Concepts:** Cross-correlation; NCHW; stride/padding; kernel gradients; pooling argmax; flatten graph.

**Files to implement:** src/minidl/nn/conv.py, src/minidl/nn/pooling.py, models/cnn.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_conv.py tests/nn/test_pooling.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/03_cnn_mnist/train.py
```

**Definition of Done:** Known-value conv, stride/padding gradcheck, pooling tie convention pass; tiny/subset CNN học được.

Các tham số CLI: `python experiments/03_cnn_mnist/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

