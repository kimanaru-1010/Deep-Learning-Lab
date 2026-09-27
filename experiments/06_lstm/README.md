# LSTM

[Curriculum](../../docs/curriculum.md) · [Debugging](../../docs/debugging_gradients.md)

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

Các tham số CLI: `python experiments/06_lstm/train.py --help`. Bài chưa hoàn thành sẽ báo StudentTODO,
không phải chạy ra kết quả giả. Với RNN/LSTM/Transformer hãy tạo toy text bằng
`python scripts/download_text_data.py` trước.

