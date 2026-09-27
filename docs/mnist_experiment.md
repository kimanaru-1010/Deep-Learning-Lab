# MNIST experiment

Nguồn dữ liệu: [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist), lưu 4 gzip
IDX gốc. Downloader kiểm checksum và dùng file `.part`; dữ liệu hỏng có sẵn sẽ báo
lỗi để bạn tự chuyển sang chỗ khác, không âm thầm ghi đè. Parser kiểm magic, dtype,
số chiều, kích thước payload, image shape, label range. Cache `.npz` chứa uint8,
normalization float32 / 255 thực hiện khi load, tùy chọn flatten hoặc NCHW.

Chạy bằng Python trong venv đã activate:

```bash
python scripts/download_mnist.py
python experiments/02_mlp_mnist/train.py --summary
```

## Level 1: một batch 32 mẫu

```bash
python experiments/debug/overfit_tiny_batch.py --samples 32 --steps 500 --lr 0.01 --watch fc1.weight --watch-index 0,0 --log-every 10
```

Lấy mẫu reproducible từ official train và lặp chính batch đó. Mỗi vòng là một step;
không có validation độc lập ở chế độ này. Confusion matrix cuối là trên tiny batch.
Mục tiêu là loss giảm rõ và model có thể gần memorize, không phải generalization.
Không hardcode threshold accuracy; bạn đánh giá cùng khả năng model và số steps.
Nếu thất bại, dừng full training và dùng [debugging guide](debugging_gradients.md).

## Level 2: subset

```bash
python experiments/02_mlp_mnist/train.py --limit-train 5000 --limit-test 1000 --epochs 10 --batch-size 64 --optimizer sgd --seed 42 --run-name subset
```

5000 mẫu được lấy từ official train trước split; với `--val-fraction 0.1`, dùng
4500 train / 500 validation. Split dùng NumPy Generator cùng seed. Batch shuffle
mới mỗi epoch, reproducible giữa các run. Last partial batch được giữ; metric trung
bình có trọng số số mẫu. Đánh giá 1000 test **chỉ sau khi chốt model**:

```bash
python experiments/02_mlp_mnist/evaluate.py --checkpoint outputs/runs/subset/checkpoints/last.npz --limit-test 1000
```

## Level 3: full MNIST

```bash
python experiments/02_mlp_mnist/train.py --epochs 10 --batch-size 64 --hidden-size 128 --lr 0.01 --optimizer sgd --seed 42 --run-name full
python experiments/02_mlp_mnist/evaluate.py --checkpoint outputs/runs/full/checkpoints/last.npz
```

Full official train có 60000 mẫu; mặc định 54000 train và 6000 validation. Official
test có 10000 mẫu. Train thay đổi weights; validation hỗ trợ chọn hyperparameters;
test dành cho đánh giá cuối. `--limit-test` trong train không tải/test model, chỉ lưu
cấu hình. Không tune dựa trên test accuracy.

MLP mặc định 784 → Linear(128) → ReLU → Linear(10). Loss nhận logits chưa softmax,
CrossEntropy phải ổn định số học. Giữ tensor graph trong mọi operation của model.

## Tệp mỗi run

CSV ghi epoch, step, train/val loss và accuracy, LR, global gradient/parameter norm.
Training metrics đo trên forward trước update; parameter norm và watch after đo sau
update. Các dòng validation có train fields rỗng. Plot loss dùng step, accuracy dùng
epoch. Confusion matrix của train script là validation; evaluate script là test.

Checkpoint lưu parameter arrays, registered buffers, epoch, step, metadata và state
optimizer số. Đây chưa phải exact-resume toàn bộ tiến trình: RNG/loader position
không lưu, và CLI chưa có resume flag. Có thể load weights để evaluation; optimizer
có thể roundtrip bằng API. Khi implement Adam, lưu counter trong `optimizer.state`.

CNN/ResNet dùng cùng loop, dữ liệu NCHW. NumPy convolution naive có thể chậm; bắt đầu
với subset nhỏ. Không cần GPU và không hứa accuracy khi core còn TODO.
