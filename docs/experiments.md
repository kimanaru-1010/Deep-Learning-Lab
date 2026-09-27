# Experiment guide

Mọi lệnh dùng `python` của venv sau activation. Mọi entry point hỗ trợ `--help` trước
khi các student algorithms được viết. Mỗi experiment nhận `--seed`.

| Bài | Entry point | Điều kiện |
|---|---|---|
| Manual | `experiments/00_manual_backprop/run.py` | Manual Linear forward/backward |
| Scalar | `experiments/01_scalar_autograd/run.py` | Value ops + reverse traversal |
| MLP | `experiments/02_mlp_mnist/train.py` | Tensor/Linear/ReLU/CE/SGD |
| CNN | `experiments/03_cnn_mnist/train.py` | Conv2D/MaxPool2D/reshape |
| ResNet | `experiments/04_resnet/train.py` | ResidualBlock và branch gradients |
| RNN | `experiments/05_rnn/train.py` | Embedding/RNN/BPTT, output head |
| LSTM | `experiments/06_lstm/train.py` | Cell/hidden state, gates, BPTT |
| Attention | `experiments/07_attention/run.py` | Attention forward/backward |
| Transformer | `experiments/08_transformer/train.py` | Causal LM composition và blocks |

## Debug tools

```bash
python experiments/debug/overfit_tiny_batch.py --samples 32 --steps 500
python experiments/debug/inspect_gradients.py --watch fc1.weight --watch-index 0,0 --log-every 1
python experiments/debug/optimizer_comparison.py --limit-train 1000 --epochs 3 --seed 42
```

Optimizer comparison reset seed trước mỗi model, cùng initial weights, split,
loader seed, batch size và LR; mỗi optimizer có state riêng. SGD/Momentum/Adam phải
được implement trước. Plot train loss vs step và validation accuracy vs epoch nằm
trong parent run; từng optimizer có subdirectory. Cùng LR giúp quan sát một điều
kiện kiểm soát, không có nghĩa là hyperparameter tốt nhất cho cả ba optimizer.

## Character-level next-token prediction

```bash
python scripts/download_text_data.py
python experiments/05_rnn/train.py --epochs 5 --model-dim 16 --sequence-length 8
python experiments/06_lstm/train.py --epochs 5 --model-dim 16 --sequence-length 8
python experiments/08_transformer/train.py --epochs 5 --model-dim 32 --num-heads 2 --optimizer adam --lr 0.001
```

Corpus toy được tạo offline. `CharacterVocabulary` có encode/decode; alphabet được
xem là metadata toàn corpus. Text split liên tiếp trước sequence sampling để tránh
window overlap giữa train/validation. Sampling là fixed set reproducible theo seed;
DataLoader shuffle windows mỗi epoch. Log CE và accuracy, in validation perplexity,
ghi greedy sample text theo epoch vào `samples.txt`. Greedy sampling là công cụ
quan sát, chưa có temperature hoặc top-k. `vocabulary.json` lưu alphabet.

Đầu ra model là `(N,T,V)`; loop flatten về `(N*T,V)` và labels `(N*T,)` cho CE.
RNN/LSTM reset hidden state giữa các windows; không carry state qua shuffled batches.
Transformer phải causal: thay suffix không được làm thay đổi prefix logits.

## Quản lý output

Run không chỉ định tên dùng timestamp UTC và hậu tố ngẫu nhiên. Tên trùng báo lỗi.
`scripts/clean_outputs.py RUN_NAME` chỉ liệt kê; thêm `--apply` để xóa các file của
đúng run đó. Utility từ chối path traversal/symlink, không xóa data hay source.
