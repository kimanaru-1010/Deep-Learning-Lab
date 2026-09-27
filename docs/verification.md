# Báo cáo kiểm chứng thiết lập

Ngày: 2026-09-27. Workspace thực tế: Windows native; dùng `.venv` đã có, không
cài lại Python hoặc uv. Tất cả package dự án được cài vào venv, editable bằng
`python -m pip install -e ".[dev]"`.

| Thành phần | Kết quả |
|---|---|
| Python | 3.12.10, venv riêng, system/user site-packages bị tắt |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| tqdm | 4.70.1 |
| Package import | PASS; minidl từ src, dependencies từ venv |
| pip check | PASS |
| `scripts/verify_env.py --record` | PASS |
| `scripts/setup.sh --skip-install` | PASS qua Git Bash, cả repo root và thư mục docs |
| Smoke / infrastructure | 41 PASS (gồm 21 smoke): CLI help, PNG, data, finite differences, monitoring, history, checkpoint, evaluation |
| Lesson tests | 58 XFAIL đúng các StudentTODO; chưa có DL core implementation |
| Ruff check / format | PASS |
| Mypy | PASS cho src/minidl |
| MNIST download thật | PASS, 4 gzip checksum đúng, 60000 train / 10000 test |
| MNIST parser offline | PASS với synthetic IDX, có malformed-input cases |
| MNIST cache lần chạy lại | PASS, không tải lại file đã có |
| Toy text | Tạo offline 13100 ký tự |
| Forbidden dependencies / installed packages / runtime AST imports | NONE |

Snapshot package thực tế: `outputs/environment/installed-packages.txt`.
Báo cáo interpreter và import versions: `outputs/environment/verification.json`.
Hai file này là kết quả local được gitignore, không phải lockfile portable.

## Giới hạn đã xác nhận

- Linux/WSL2 chưa được chạy tại máy này: WSL hiện chỉ có docker-desktop, không có
  distro phát triển phù hợp. Không cài thêm OS/interpreter. `setup.sh` đã chạy thật
  bằng Git Bash trên Windows; CI template Ubuntu 3.11/3.12 đã tạo nhưng chưa chạy remote.
- Không train model DL trong bước setup vì core cố ý chưa implement. `train.py`
  đã được gọi thử và dừng bằng thông báo Linear.forward TODO có file/test cần làm.
- Checkpoint hỗ trợ weights/buffers/optimizer state và metadata, không lưu RNG hay
  vị trí DataLoader; chưa có CLI resume toàn bộ training.
- Seed được kiểm soát, nhưng không cam kết bitwise equality giữa OS/BLAS khác nhau.

## Chạy lại

Sau khi activate venv:

```bash
python scripts/verify_env.py --record
python -m pytest -q -r f
python -m ruff check .
python -m ruff format --check .
python -m mypy src/minidl
```

Muốn nghiệm thu bài học thay vì chỉ nghiệm thu setup, thêm `--run-student` vào
pytest và chọn test của component đang học. Xfail không có nghĩa bài học đã xong.
