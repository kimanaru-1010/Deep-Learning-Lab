# RNN

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

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

Các tham số CLI: `python experiments/05_rnn/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

