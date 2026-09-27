# Curriculum

Đi theo thứ tự dưới đây. Folder experiment giữ số ban đầu 00–08; Tensor Autograd
là lesson 02 trong curriculum và được thực hành bằng tests/core, nên số curriculum
sau đó lớn hơn số folder một đơn vị. Không có bài bị bỏ qua.

Mọi lệnh dùng Python của .venv (activate trước, hoặc chỉ định đường dẫn đầy đủ).
Bài tiếp theo chỉ bắt đầu sau Definition of Done của bài trước. Các TODO là nơi
bạn viết code; tests và gradient checker không chứa generic reference solution.

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

## 02 — Tensor Autograd

**Goal:** Mở rộng graph sang ndarray và broadcast.

**Concepts:** requires_grad; matmul; reduction; reshape/transpose; broadcasting; leaf accumulation.

**Files to implement:** src/minidl/core/tensor.py, src/minidl/core/ops.py, src/minidl/core/autograd.py

**Tests to run:**

```bash
python -m pytest tests/core/test_tensor.py --run-student -q
```

**Experiment to run:**

```bash
python -m pytest tests/core/test_tensor.py --run-student -q
```

**Definition of Done:** Tất cả tensor tests pass; gradients đúng shape; constant không có grad; numerical check pass.

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

## 06 — RNN

**Goal:** Hiểu hidden state, parameter sharing và BPTT.

**Concepts:** Một cell dùng lại qua thời gian; unrolling; gradient từ nhiều timestep; sequence windows.

**Files to implement:** src/minidl/nn/recurrent.py (RNNCell/RNN), src/minidl/nn/embedding.py, models/rnn.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_recurrent.py tests/nn/test_embedding.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/05_rnn/train.py
```

**Definition of Done:** Cell input/hidden/parameter gradcheck pass; sequence shape/BPTT tests pass; toy next-token CE giảm.

## 07 — LSTM

**Goal:** Tự triển khai memory qua cả cell và hidden state.

**Concepts:** Input gate; forget gate; candidate; output gate; thứ tự packed gates i,f,g,o; BPTT.

**Files to implement:** src/minidl/nn/recurrent.py (LSTMCell/LSTM), models/lstm.py

**Tests to run:**

```bash
python -m pytest tests/nn/test_recurrent.py --run-student -q
```

**Experiment to run:**

```bash
python experiments/06_lstm/train.py
```

**Definition of Done:** Cell gradients qua x/h/c và mọi weight/bias pass; shared-state unrolling đúng; toy text học được.

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

## Mốc bổ sung: Optimizers và initialization

Sau MLP/SGD, hoàn thành Momentum và Adam trong src/minidl/optim; chạy tests/optim.
Hoàn thành Xavier/He trong src/minidl/nn/initialization.py, ghi rõ fan convention,
rồi chạy tests/nn/test_initialization.py. Basic normal/uniform/zeros là hạ tầng
đã có, không thay thế bài học variance scaling.

Module traversal, Parameter storage, seed, finite differences, metrics, data I/O,
logging, plots và checkpoint đã được triển khai để bạn tập trung vào DL core.

