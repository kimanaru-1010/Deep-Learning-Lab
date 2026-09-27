"""Explicit shared MNIST loop. Read every step here before your first training run."""

import numpy as np

from experiments.common import make_optimizer, parser, positive_int
from minidl.core import Tensor
from minidl.data import ArrayDataset, DataLoader, train_val_split
from minidl.data.mnist import load_mnist
from minidl.losses import CrossEntropyLoss
from minidl.nn import model_summary
from minidl.training.callbacks import log_watch
from minidl.training.history import History
from minidl.training.metrics import accuracy, confusion_matrix
from minidl.training.monitoring import (
    ParameterWatch,
    detect_nonfinite,
    gradient_norm,
    parameter_norm,
)
from minidl.training.trainer import evaluate
from minidl.utils.checkpoint import load_checkpoint, save_checkpoint
from minidl.utils.logging import create_run
from minidl.utils.seed import set_seed
from minidl.utils.visualization import plot_confusion_matrix, plot_history
from models.cnn import CNN
from models.mlp import MLP
from models.resnet import SmallResNet


def arguments(default_model="mlp", tiny=False, evaluation=False):
    result = parser("MNIST learning lab: complete student TODOs before training")
    result.add_argument("--model", choices=["mlp", "cnn", "resnet"], default=default_model)
    result.add_argument("--data-root", default="data")
    result.add_argument("--hidden-size", type=positive_int, default=128)
    result.add_argument("--limit-train", type=positive_int)
    result.add_argument("--limit-test", type=positive_int, help="Only used by final evaluation")
    result.add_argument("--val-fraction", type=float, default=0.1)
    if tiny:
        result.add_argument("--samples", type=positive_int, default=32)
        result.add_argument("--steps", type=positive_int, default=500)
    if evaluation:
        result.add_argument("--checkpoint", required=True)
    return result


def build_model(args):
    set_seed(args.seed)
    if args.model == "mlp":
        return MLP(hidden_size=args.hidden_size)
    return CNN() if args.model == "cnn" else SmallResNet()


def run(args, tiny=False):
    model = build_model(args)
    if args.summary:
        print(model_summary(model))
        return None
    # Probe before reading a dataset: a missing lesson produces an actionable error.
    shape = (2, 784) if args.model == "mlp" else (2, 1, 28, 28)
    model(Tensor(np.zeros(shape, dtype=np.float32)))
    dataset = load_mnist(args.data_root, flatten=args.model == "mlp", limit=args.limit_train)
    if tiny:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(dataset), min(args.samples, len(dataset)), replace=False)
        train = ArrayDataset(*dataset[ids])
        validation = None
        train_loader = DataLoader(train, len(train))
        epochs = args.steps
    else:
        train, validation = train_val_split(dataset, args.val_fraction, args.seed)
        train_loader = DataLoader(train, args.batch_size, shuffle=True, seed=args.seed)
        epochs = args.epochs
    val_loader = DataLoader(validation, args.batch_size) if validation is not None else None
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = CrossEntropyLoss()
    directory = create_run(args.output_root, args.run_name, vars(args))
    history = History(directory / "metrics.csv")
    watch = (
        ParameterWatch(args.watch, tuple(map(int, args.watch_index.split(","))), args.log_every)
        if args.watch
        else None
    )
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct_sum, seen = 0.0, 0.0, 0
        for x, y in train_loader:
            # This is the complete observable training sequence, not model.fit().
            logits = model(Tensor(x))
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = gradient_norm(model.parameters())
            detect_nonfinite(model, loss.data, step)
            before = watch.capture(model, step) if watch else None
            optimizer.step()
            detect_nonfinite(model, step=step)
            if watch:
                log_watch(directory / "parameter_watch.jsonl", watch.report(model, before))
            batch_loss, batch_acc = float(loss.data), accuracy(logits.data, y)
            loss_sum += batch_loss * len(y)
            correct_sum += batch_acc * len(y)
            seen += len(y)
            history.append(
                epoch=epoch,
                step=step,
                train_loss=batch_loss,
                train_acc=batch_acc,
                grad_norm=grad_norm,
                param_norm=parameter_norm(model.parameters()),
                lr=optimizer.lr,
            )
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step:05d} loss={batch_loss:.5f} accuracy={batch_acc:.4f} "
                    f"grad_norm={grad_norm} param_norm={parameter_norm(model.parameters()):.5f} lr={optimizer.lr}"
                )
            step += 1
        validation_metrics = evaluate(model, val_loader, criterion) if val_loader else None
        if validation_metrics:
            history.append(
                epoch=epoch,
                step=step - 1,
                val_loss=validation_metrics["loss"],
                val_acc=validation_metrics["accuracy"],
                lr=optimizer.lr,
            )
        if not tiny or epoch == epochs:
            print(
                f"Epoch {epoch}: train_loss={loss_sum / seen:.5f} train_accuracy={correct_sum / seen:.4f} "
                f"validation={None if validation_metrics is None else (validation_metrics['loss'], validation_metrics['accuracy'])}"
            )
            save_checkpoint(
                directory / "checkpoints" / "last.npz",
                model,
                optimizer,
                epoch=epoch,
                step=step,
                metadata={"config": vars(args)},
            )
    final_loader = val_loader if val_loader is not None else DataLoader(train, len(train))
    final = evaluate(model, final_loader, criterion)
    plot_confusion_matrix(
        confusion_matrix(final["targets"], final["predictions"], 10),
        directory / "confusion_matrix.png",
    )
    plot_history(history, directory, args.show)
    print(f"Run saved: {directory}")
    return history, directory


def evaluate_main():
    args = arguments(evaluation=True).parse_args()
    # Recover architecture from checkpoint metadata, not default CLI dimensions.
    import json

    with np.load(args.checkpoint, allow_pickle=False) as archive:
        config = json.loads(str(archive["metadata"]))["metadata"]["config"]
    args.model, args.hidden_size = config["model"], config["hidden_size"]
    model = build_model(args)
    load_checkpoint(args.checkpoint, model)
    dataset = load_mnist(
        args.data_root, split="test", flatten=args.model == "mlp", limit=args.limit_test
    )
    result = evaluate(model, DataLoader(dataset, args.batch_size), CrossEntropyLoss())
    directory = create_run(args.output_root, args.run_name, vars(args))
    plot_confusion_matrix(
        confusion_matrix(result["targets"], result["predictions"], 10),
        directory / "confusion_matrix.png",
        args.show,
    )
    (directory / "evaluation.json").write_text(
        json.dumps({"test_loss": result["loss"], "test_accuracy": result["accuracy"]}),
        encoding="utf-8",
    )
    print(f"test_loss={result['loss']:.5f} test_accuracy={result['accuracy']:.4f}; {directory}")


def main(default_model="mlp", tiny=False):
    run(arguments(default_model, tiny).parse_args(), tiny)
