# MLP + MNIST

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

## 03 — MLP + MNIST

**Goal:** Tự hoàn thiện mạng phân loại đầu tiên.

**Concepts:** Module/Parameter; Linear/ReLU; stable CE; SGD; initialization; train/val split.

**Files to implement:** src/minidl/nn/linear.py, src/minidl/nn/activations.py, src/minidl/losses/, src/minidl/optim/sgd.py, models/mlp.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_linear.py tests/nn/test_activations.py tests/losses tests/optim tests/integration --run-student -q
```

**Experiment to run:**

```bash
python experiments/02_mlp_mnist/train.py
```

**Definition of Done:** Unit/gradchecks pass; overfit 32 MNIST; loss subset giảm; validation/plots/checkpoint có thật.

Các tham số CLI: `python experiments/02_mlp_mnist/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

