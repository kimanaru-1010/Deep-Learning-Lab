import gzip
import struct

import numpy as np
import pytest

from minidl.data.mnist import load_mnist, read_idx


def write_idx(path, array):
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">HBB", 0, 8, array.ndim))
        stream.write(struct.pack(">" + "I" * array.ndim, *array.shape))
        stream.write(array.tobytes())


def test_parser_cache_normalization_and_subset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    images = np.full((4, 28, 28), 255, dtype=np.uint8)
    labels = np.array([0, 2, 4, 9], dtype=np.uint8)
    write_idx(raw / "train-images-idx3-ubyte.gz", images)
    write_idx(raw / "train-labels-idx1-ubyte.gz", labels)
    np.testing.assert_array_equal(read_idx(raw / "train-images-idx3-ubyte.gz"), images)
    data = load_mnist(tmp_path, limit=2)
    assert data.x.shape == (2, 784) and data.x.dtype == np.float32
    assert np.all(data.x == 1)
    np.testing.assert_array_equal(data.y, [0, 2])
    assert (tmp_path / "processed" / "mnist-train.npz").exists()
    assert load_mnist(tmp_path, flatten=False).x.shape == (4, 1, 28, 28)


@pytest.mark.parametrize(
    "payload",
    [b"", b"abcd", struct.pack(">HBBI", 0, 8, 1, 3) + b"x", struct.pack(">HBBI", 0, 8, 1, 0)],
)
def test_malformed_idx(tmp_path, payload):
    path = tmp_path / "bad.idx"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_idx(path)


def test_missing_and_bad_options(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_mnist"):
        load_mnist(tmp_path)
    with pytest.raises(ValueError):
        load_mnist(tmp_path, split="validation")
    with pytest.raises(ValueError):
        load_mnist(tmp_path, limit=0)
