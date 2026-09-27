import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import minidl
from minidl.data import ArrayDataset, DataLoader
from minidl.utils.gradcheck import check_gradient
from minidl.utils.seed import set_seed
from minidl.utils.visualization import plot_confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
CLIS = sorted(
    [
        *ROOT.glob("experiments/*/train.py"),
        *ROOT.glob("experiments/*/run.py"),
        *ROOT.glob("experiments/debug/*.py"),
        ROOT / "experiments/02_mlp_mnist/evaluate.py",
        ROOT / "scripts/download_mnist.py",
        ROOT / "scripts/download_text_data.py",
        ROOT / "scripts/verify_env.py",
        ROOT / "scripts/bootstrap.py",
        ROOT / "scripts/clean_outputs.py",
    ]
)


def test_import_numpy_seed_batching_and_gradient():
    assert minidl.__version__
    set_seed(4)
    values = random.random(), np.random.rand()
    set_seed(4)
    assert values == (random.random(), np.random.rand())
    assert len(list(DataLoader(ArrayDataset(np.zeros((3, 2)), np.zeros(3)), 2))) == 2
    assert check_gradient(lambda z: np.sum(z * z), np.ones(2), np.full(2, 2.0)).passed


def test_headless_plot_and_paths(tmp_path):
    path = plot_confusion_matrix(np.eye(2), tmp_path / "nested" / "plot.png")
    assert path.stat().st_size > 100


@pytest.mark.parametrize("path", CLIS, ids=lambda p: str(p.relative_to(ROOT)))
def test_cli_help(path):
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_unfinished_lesson_has_actionable_error():
    # This smoke check uses the explicit sentinel, not a layer whose implementation may change.
    from minidl.utils.student import StudentTODO, todo

    with pytest.raises(StudentTODO, match="--run-student"):
        todo("example.forward", "example.py", "tests/example.py")
