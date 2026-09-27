"""Numeric NPZ + JSON metadata, never pickle. Flat optimizer state is supported."""

import json
from pathlib import Path

import numpy as np


def save_checkpoint(path, model, optimizer=None, *, epoch=0, step=0, metadata=None):
    arrays = {f"parameter/{name}": p.data for name, p in model.named_parameters()}
    arrays.update({f"buffer/{name}": array for name, array in model.named_buffers()})
    info = {"epoch": epoch, "step": step, "metadata": metadata or {}}
    if optimizer is not None:
        info["optimizer"] = type(optimizer).__name__
        info["hyperparameters"] = {
            key: value for key, value in vars(optimizer).items() if isinstance(value, (float, int))
        }
        arrays.update(
            {f"optimizer/{key}": np.asarray(value) for key, value in optimizer.state.items()}
        )
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("Checkpoint supports numeric state only")
    arrays["metadata"] = np.array(json.dumps(info))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"parameter/{name}" for name in parameters}
        actual = {name for name in archive.files if name.startswith("parameter/")}
        if expected != actual:
            raise ValueError("Checkpoint parameter names do not match model")
        if {f"buffer/{name}" for name in buffers} != {
            name for name in archive.files if name.startswith("buffer/")
        }:
            raise ValueError("Checkpoint buffer names do not match model")
        buffer_values = {name: archive[f"buffer/{name}"] for name in buffers}
        for name, value in buffer_values.items():
            if value.shape != buffers[name].shape:
                raise ValueError(f"Checkpoint buffer shape mismatch for {name}")
        info = json.loads(str(archive["metadata"]))
        values = {name: archive[f"parameter/{name}"] for name in parameters}
        for name, value in values.items():
            if value.shape != parameters[name].shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}")
        if optimizer is not None and info.get("optimizer") != type(optimizer).__name__:
            raise ValueError("Checkpoint optimizer does not match")
        state = {
            key.removeprefix("optimizer/"): archive[key].copy()
            for key in archive.files
            if key.startswith("optimizer/")
        }
    for name, value in values.items():
        parameters[name].data[...] = value
    for name, value in buffer_values.items():
        buffers[name][...] = value
    if optimizer is not None:
        optimizer.state = state
        for key, value in info["hyperparameters"].items():
            setattr(optimizer, key, value)
    return info
