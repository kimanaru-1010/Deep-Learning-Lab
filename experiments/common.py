import argparse
import sys

from minidl.optim import SGD, Adam, Momentum
from minidl.utils.student import StudentTODO


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=positive_int, default=10)
    result.add_argument("--batch-size", type=positive_int, default=64)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    result.add_argument("--run-name")
    result.add_argument("--output-root", default="outputs/runs")
    result.add_argument("--show", action="store_true", help="Open saved plots in desktop viewer")
    result.add_argument("--log-every", type=positive_int, default=10)
    result.add_argument("--watch", default="", help="Parameter name; see --summary")
    result.add_argument("--watch-index", default="0,0")
    result.add_argument("--summary", action="store_true", help="Print parameter shapes and exit")
    return result


def make_optimizer(name, parameters, lr):
    if name == "sgd":
        return SGD(parameters, lr)
    if name == "momentum":
        return Momentum(parameters, lr)
    if name == "adam":
        return Adam(parameters, lr)
    raise ValueError(f"Unknown optimizer: {name}")


def friendly(main):
    try:
        main()
    except (StudentTODO, FileNotFoundError, FileExistsError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
