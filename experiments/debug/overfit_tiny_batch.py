from experiments.common import friendly
from experiments.mnist import main

if __name__ == "__main__":
    friendly(lambda: main(tiny=True))
