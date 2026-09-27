"""MNIST gzip/IDX parsing and verified download using only stdlib + NumPy."""

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .dataset import ArrayDataset

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def read_idx(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError("Truncated IDX header")
        zero, dtype_code, dimensions = struct.unpack(">HBB", header)
        if zero != 0 or dtype_code != 8 or dimensions not in (1, 3):
            raise ValueError("Expected unsigned-byte IDX labels (1D) or images (3D)")
        size_bytes = stream.read(dimensions * 4)
        if len(size_bytes) != dimensions * 4:
            raise ValueError("Truncated IDX dimensions")
        shape = struct.unpack(">" + "I" * dimensions, size_bytes)
        if any(n == 0 for n in shape):
            raise ValueError("IDX dimensions must be positive")
        payload = stream.read()
    if len(payload) != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("IDX payload length does not match header")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()


def download_mnist(root="data", base_url=BASE_URL):
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = raw / name
        if destination.exists():
            if hashlib.md5(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Corrupt cached file: {destination}; move it aside and retry")
            continue
        temporary = destination.with_suffix(".gz.part")
        try:
            with urllib.request.urlopen(base_url + name, timeout=60) as response:
                with temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"MNIST checksum mismatch: {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {name}")


def load_mnist(root="data", split="train", flatten=True, limit=None):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    root = Path(root)
    cache = root / "processed" / f"mnist-{split}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            images, labels = stored["images"], stored["labels"]
    else:
        prefix = "train" if split == "train" else "t10k"
        try:
            images = read_idx(root / "raw" / f"{prefix}-images-idx3-ubyte.gz")
            labels = read_idx(root / "raw" / f"{prefix}-labels-idx1-ubyte.gz")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "MNIST missing. Run: python scripts/download_mnist.py"
            ) from error
        _validate(images, labels)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, images=images, labels=labels)
        cache.with_suffix(".tmp").replace(cache)
    _validate(images, labels)
    images = images[:limit].astype(np.float32) / 255.0
    images = images.reshape(len(images), -1) if flatten else images[:, None, :, :]
    return ArrayDataset(images, labels[:limit].astype(np.int64))


def _validate(images, labels):
    if images.ndim != 3 or images.shape[1:] != (28, 28) or labels.shape != (len(images),):
        raise ValueError("Invalid MNIST image/label shapes")
    if images.dtype != np.uint8 or labels.dtype != np.uint8 or np.any(labels > 9):
        raise ValueError("Invalid MNIST dtype or label range")
