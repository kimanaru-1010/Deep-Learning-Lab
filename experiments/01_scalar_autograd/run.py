import argparse

from experiments.common import friendly
from experiments.scalar import Value


def main():
    parser = argparse.ArgumentParser(description="Scalar graph: ((x*w)+1)^2; expected dx=42, dw=28")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    x, w = Value(2), Value(3)
    loss = (x * w + 1) ** 2
    loss.backward()
    print(f"loss={loss.data}, dx={x.grad}, dw={w.grad}")
    assert abs(x.grad - 42) < 1e-8 and abs(w.grad - 28) < 1e-8


if __name__ == "__main__":
    friendly(main)
