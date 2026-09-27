# Deep Learning from Scratch — learning lab

Tự xây **MLP → CNN → ResNet → RNN → LSTM → Attention → Transformer** từ NumPy.
Đây là môi trường thực hành: hạ tầng dữ liệu, gradient checker, logging, plotting và
checkpoint đã có; bạn tự viết thuật toán DL. Chạy train ngay lúc đầu sẽ báo bài học
cần hoàn thành, không tạo kết quả học giả.

## Philosophy

NumPy là numerical backend duy nhất. Không dùng PyTorch, TensorFlow, Keras, JAX,
Flax, thư viện autodiff hoặc layer/optimizer dựng sẵn. `Tensor.data` là ndarray;
forward, graph, chain rule, backward và update là bài tập của bạn.
`StudentTODO` kế thừa `NotImplementedError`; hãy thay phần TODO bằng implementation
của mình. Không thay nó bằng zeros hoặc numerical gradient trong training.

## Bắt đầu trên Windows hiện tại

Python 3.12 và `.venv` đã tồn tại. Không cài lại Python/uv. Tại repository root:

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap.py --skip-install
.\.venv\Scripts\python.exe scripts/verify_env.py
.\.venv\Scripts\python.exe -m pytest tests/smoke -q
.\.venv\Scripts\python.exe experiments/00_manual_backprop/run.py --help
```

Khi clone mới hoặc thiếu dependencies, bỏ `--skip-install` để cài editable theo
`pyproject.toml`. Pip giữ các package đang thỏa mãn yêu cầu; không dùng force-reinstall.
Không cần uv để chạy lab. Mặc định không nâng pip đã có; `--upgrade-pip` là tùy chọn.

Kích hoạt tùy chọn: `.\.venv\Scripts\Activate.ps1`. Nếu chính sách PowerShell chặn
activation, cứ dùng đường dẫn Python đầy đủ ở trên, không cần đổi execution policy.

## Setup Linux / WSL2

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_env.py
python -m pytest tests/smoke -q
```

Script tìm **Python 3.12**, dự phòng **3.11**, kiểm tra phiên bản thực tế, tạo venv
chuẩn không kế thừa system site-packages và không cài interpreter hệ thống. Script
chạy từ thư mục bất kỳ; nếu `.venv` hỏng hoặc thuộc OS khác, script dừng và giữ nó.
Chạy lại môi trường đã cài: `bash scripts/setup.sh --skip-install`.

Tự làm các bước tương đương:

```bash
python3.12 -m venv .venv  # dùng python3.11 nếu 3.12 chưa có
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pip check
./.venv/bin/python scripts/verify_env.py --record
```

Nếu bạn muốn chủ động nâng pip: `./.venv/bin/python -m pip install --upgrade pip`.
Các lệnh `python ...` dưới đây chỉ dùng sau khi activate. Không cần activate thì
thay `python` bằng `./.venv/bin/python` (Linux) hoặc `.\.venv\Scripts\python.exe` (Windows).
Không chia sẻ một `.venv` giữa Windows và WSL; dùng checkout riêng trên mỗi OS.

`verify_env.py` kiểm tra interpreter, prefix, nguồn import, `pip check`, các package
bị cấm, AST runtime imports, PNG headless, gradient checker và toàn bộ test hạ tầng.
`--record` ghi phiên bản thực tế trong `outputs/environment/`; đây là snapshot,
**không phải lockfile**. `--skip-tests` chỉ là kiểm tra nhanh, không đủ nghiệm thu.

## Cấu trúc

```text
src/minidl/        core/ nn/ losses/ optim/ data/ training/ utils/
models/            MLP, CNN và skeleton ResNet/RNN/LSTM/Transformer
experiments/       bài 00–08, debug/, mnist.py, text_training.py
tests/             smoke/, data/, training/ và tests bài học
scripts/           bootstrap, verify, download, clean
docs/              curriculum, thứ tự implementation, debugging, experiments
data/raw/          MNIST gzip và toy text (không commit)
data/processed/    MNIST cache NumPy (không commit)
outputs/runs/      mỗi run một thư mục riêng (không ghi đè)
```

`experiments/mnist.py` chứa **loop training rõ ràng**, dùng chung bởi các CLI MLP,
CNN, ResNet và tiny-batch. Mở file này để xem forward → loss → zero_grad → backward
→ quan sát gradient → optimizer.step → quan sát parameter. `models/mlp.py` chỉ ghép
layer; không có thuật toán DL hoàn chỉnh bị giấu trong utility.

## Shape và precision

| Loại | Shape |
|---|---|
| Dense | `(batch, features)`; Linear weight `(in_features, out_features)` |
| Image | `(batch, channels, height, width)` — NCHW |
| Sequence features | `(batch, sequence_length, features)` |
| Token IDs | `(batch, sequence_length)`, integer ndarray |
| Cross entropy | logits `(N,C)`, labels `(N,)`, output scalar |

Unit tests và gradient checks dùng **float64**, training mặc định **float32**.
Attention mask boolean: **True = được phép attend**. RNN trả `(outputs, hidden)`;
LSTM trả `(outputs, (hidden, cell))`. Loss là mean, không cộng rồi chia batch lần hai.

## First learning workflow

1. Đọc [curriculum](docs/curriculum.md) và [thứ tự implementation](docs/implementation_order.md).
2. Viết manual Linear/ReLU/loss/MLP trong `experiments/manual.py`.
3. Viết scalar autograd, rồi Tensor ops và traversal.
4. Hoàn thành Linear/activation/loss/SGD; test forward và finite differences.
5. Overfit 32 mẫu MNIST trước, rồi subset, cuối cùng full MNIST.

```bash
python experiments/00_manual_backprop/run.py
python -m pytest tests/core/test_manual.py --run-student -q
```

Ban đầu các lệnh trên báo TODO hoặc fail đúng bài học. `pytest -q` mặc định chỉ
xfail **StudentTODO trong test được đánh dấu student**; lỗi assertion, shape,
infrastructure hay `NotImplementedError` thông thường vẫn fail. Khi học, luôn dùng
`--run-student` để không nhầm xfail với bài đã hoàn thành.

## MNIST

```bash
python scripts/download_mnist.py
python experiments/debug/overfit_tiny_batch.py --samples 32 --steps 500 --watch fc1.weight --watch-index 0,0
python experiments/02_mlp_mnist/train.py --limit-train 5000 --limit-test 1000 --epochs 10 --run-name mlp-subset
python experiments/02_mlp_mnist/train.py --epochs 10 --batch-size 64 --lr 0.01 --optimizer sgd --hidden-size 128 --seed 42 --run-name mlp-full
python experiments/02_mlp_mnist/evaluate.py --checkpoint outputs/runs/mlp-full/checkpoints/last.npz
```

Train không đọc test set. `--limit-test` chỉ được sử dụng ở evaluation; thêm flag
này vào train chỉ lưu cấu hình, không dùng test để tune. `--limit-train 5000` lấy
5000 mẫu official train **trước** validation split (mặc định 4500 train / 500 val).
Xem [MNIST experiment](docs/mnist_experiment.md).

## Quan sát model học

Mỗi run ghi `config.json`, `metrics.csv`, `loss.png`, `accuracy.png`, `grad_norm.png`,
`param_norm.png`, `lr.png`, `confusion_matrix.png`, `checkpoints/last.npz` và
`parameter_watch.jsonl` khi bật `--watch`. Tên run có sẵn sẽ báo lỗi, không ghi đè.
NaN/Inf được cảnh báo kèm tên parameter/step, không tự sửa gradient.
`--summary` in shape/count parameter; `--show` mở các ảnh đã lưu bằng desktop viewer.

```bash
python experiments/02_mlp_mnist/train.py --summary
python experiments/debug/inspect_gradients.py --samples 32 --steps 10 --watch fc1.weight --log-every 1
python experiments/debug/optimizer_comparison.py --limit-train 1000 --epochs 3 --seed 42
python scripts/download_text_data.py
python experiments/08_transformer/train.py --epochs 5 --sequence-length 16 --model-dim 32 --optimizer adam --lr 0.001
```

Các lệnh training cần TODO tương ứng hoàn thành. Dữ liệu text mặc định là văn bản
toy nguyên bản tạo offline. Có thể truyền HTTPS corpus riêng bằng `--url`; tự kiểm
tra giấy phép. Text split trước khi lấy windows; theo dõi CE, perplexity, sample text.

## Completion criteria

Layer xong khi forward/shape đúng, backward đúng, known-value/edge tests và gradient
check pass, tham gia training được. Model xong khi có gradient, update parameter,
overfit tiny batch, loss giảm, validation được theo dõi, có plots và checkpoint
roundtrip. **Không overfit nổi tiny batch thì dừng và debug**, không chạy full data.

## IDE và kiểm tra chất lượng

VS Code: **Python: Select Interpreter → .venv**. Settings dùng đường dẫn tương đối.
Linux chọn `.venv/bin/python`, Windows chọn `.venv/Scripts/python.exe`. Kiểm tra
`import sys; print(sys.executable)` trong terminal lẫn debugger. Không cài IDE/extension tự động.

```bash
python -m pytest -q
python -m pytest -m "not student" -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src/minidl
```

Mypy target 3.12 để tương thích stubs NumPy trên môi trường 3.12; mã runtime dùng cú pháp
3.11. CI template kiểm tra Ubuntu Python 3.11/3.12 khi bạn đưa repository lên GitHub.
CI không tải MNIST và không cần hoàn thành các TODO.

## Environment / Virtualenv Troubleshooting

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu Python 3.12/3.11 | Cài interpreter phù hợp ngoài lab rồi chạy setup; script không tự cài hệ thống. |
| Thiếu `venv`/`ensurepip` | Dùng bản Python có hai module này; trên Linux bổ sung gói venv phù hợp bằng công cụ hệ thống của bạn. |
| `.venv` sai version hoặc OS | Đóng tiến trình dùng venv, đổi tên thủ công sang `.venv.backup`, rồi chạy setup. Không sao chép site-packages giữa hai Python. |
| Không đủ quyền ghi | Đặt checkout ở thư mục bạn sở hữu; tránh sửa quyền toàn hệ thống. |
| Pip/network lỗi | Kiểm tra kết nối/proxy; chạy lại đúng `.venv` interpreter. Không thay bằng pip toàn cục. |
| IDE chọn nhầm Python | Chọn lại interpreter, mở terminal mới, kiểm tra `sys.executable`. |
| `ModuleNotFoundError: minidl` | Chạy `.venv` Python với `-m pip install -e ".[dev]"` tại repo root. |
| Dependency conflict | Chạy `python -m pip check`; đối chiếu `pyproject.toml` và snapshot môi trường. |
| TODO khi chạy train | Hoàn thành tệp/test được thông báo; đây không phải lỗi cài đặt. |

Không commit `.venv`, cache, executable hay snapshot đường dẫn máy cá nhân.
Xem [testing strategy](docs/testing_strategy.md), [debugging gradients](docs/debugging_gradients.md)
và [experiment guide](docs/experiments.md).
