"""Entry point. The explicit forward/loss/backward/update loop is in experiments/mnist.py."""

from experiments.common import friendly
from experiments.mnist import main

if __name__ == "__main__":
    friendly(lambda: main("mlp"))
