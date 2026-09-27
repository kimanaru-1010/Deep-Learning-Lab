from experiments.common import friendly
from experiments.mnist import main

if __name__ == "__main__":
    # Use --watch fc1.weight --watch-index 0,0 --log-every 1.
    friendly(lambda: main(tiny=True))
