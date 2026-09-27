import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import friendly
from experiments.mnist import arguments, run
from minidl.utils.logging import create_run


def main():
    args = arguments().parse_args()
    parent = create_run(args.output_root, args.run_name, vars(args))
    histories = {}
    for name in ("sgd", "momentum", "adam"):
        trial = copy.deepcopy(args)
        trial.optimizer, trial.run_name, trial.output_root = name, name, str(parent)
        result = run(trial)  # run resets seed, initialization, data split and loader RNG.
        if result is not None:
            histories[name] = result[0]
    for metric, axis in [("train_loss", "step"), ("val_acc", "epoch")]:
        fig, ax = plt.subplots()
        for name, history in histories.items():
            rows = [r for r in history.rows if r[metric] is not None]
            ax.plot([r[axis] for r in rows], [r[metric] for r in rows], label=name)
        ax.set(xlabel=axis, ylabel=metric)
        ax.legend()
        fig.savefig(Path(parent) / f"compare_{metric}.png")
        plt.close(fig)
    print(parent)


if __name__ == "__main__":
    friendly(main)
