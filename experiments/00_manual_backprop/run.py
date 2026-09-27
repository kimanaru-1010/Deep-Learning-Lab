import argparse

import numpy as np

from experiments.common import friendly
from experiments.manual import Linear
from minidl.utils.gradcheck import check_gradient
from minidl.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Manual Linear input gradient check. Implement experiments/manual.py."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    layer = Linear(3, 2)
    x = np.array([[0.2, -0.3, 0.8], [0.6, 0.1, -0.4]])
    upstream = np.array([[0.2, 0.7], [-0.4, 0.1]])
    layer.forward(x)
    dx = layer.backward(upstream)
    report = check_gradient(lambda value: np.sum(layer.forward(value) * upstream), x, dx)
    print(report)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    friendly(main)
