import argparse

import numpy as np

from experiments.common import friendly
from minidl.core import Tensor
from minidl.nn import ScaledDotProductAttention
from minidl.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Inspect tiny Q/K/V attention forward and gradients"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = set_seed(args.seed)
    q, k, v = [Tensor(rng.normal(size=(1, 3, 2)), requires_grad=True) for _ in range(3)]
    result = ScaledDotProductAttention()(q, k, v, mask=np.tril(np.ones((3, 3), dtype=bool)))
    result.sum().backward()
    print("output:", result.data)
    print("Q/K/V gradients:", q.grad, k.grad, v.grad)


if __name__ == "__main__":
    friendly(main)
