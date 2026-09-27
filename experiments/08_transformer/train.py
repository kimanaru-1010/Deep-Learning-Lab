from experiments.common import friendly
from experiments.text_training import main

if __name__ == "__main__":
    friendly(lambda: main("transformer"))
