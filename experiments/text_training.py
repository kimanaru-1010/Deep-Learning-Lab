"""Shared small character-level tasks for RNN, LSTM and Transformer lessons."""

import json
from pathlib import Path

import numpy as np

from experiments.common import make_optimizer, parser, positive_int
from minidl.data import DataLoader
from minidl.data.text import CharacterVocabulary, sequence_dataset, split_text
from minidl.losses import CrossEntropyLoss
from minidl.nn import model_summary
from minidl.training.callbacks import log_watch
from minidl.training.history import History
from minidl.training.metrics import accuracy, perplexity
from minidl.training.monitoring import (
    ParameterWatch,
    detect_nonfinite,
    gradient_norm,
    parameter_norm,
)
from minidl.training.trainer import evaluate
from minidl.utils.checkpoint import save_checkpoint
from minidl.utils.logging import create_run
from minidl.utils.seed import set_seed
from minidl.utils.visualization import plot_history
from models.lstm import LSTMLanguageModel
from models.rnn import RNNLanguageModel
from models.transformer import TinyTransformer


def sample_text(model, vocabulary, prompt, length=80, context=16):
    """Deterministic greedy sample for observing progress; no optimization here."""
    mode = model.training
    model.eval()
    tokens = vocabulary.encode(prompt).tolist()
    try:
        for _ in range(length):
            logits = model(np.array([tokens[-context:]], dtype=np.int64))
            tokens.append(int(np.argmax(logits.data[0, -1])))
    finally:
        model.train(mode)
    return vocabulary.decode(tokens)


def main(kind="transformer"):
    cli = parser(f"Small {kind} character language model")
    cli.add_argument("--text", default="data/raw/toy.txt")
    cli.add_argument("--sequence-length", type=positive_int, default=16)
    cli.add_argument("--model-dim", type=positive_int, default=32)
    cli.add_argument("--num-heads", type=positive_int, default=2)
    cli.add_argument("--sequences", type=positive_int, default=256)
    cli.add_argument("--val-fraction", type=float, default=0.1)
    args = cli.parse_args()
    set_seed(args.seed)
    text = Path(args.text).read_text(encoding="utf-8")
    training_text, validation_text = split_text(text, args.val_fraction)
    # The character alphabet is metadata. No target statistics are fitted on validation.
    vocabulary = CharacterVocabulary(text)
    train = sequence_dataset(
        vocabulary.encode(training_text), args.sequence_length, args.sequences, args.seed
    )
    val = sequence_dataset(
        vocabulary.encode(validation_text),
        args.sequence_length,
        max(16, args.sequences // 5),
        args.seed + 1,
    )
    if kind == "rnn":
        model = RNNLanguageModel(len(vocabulary), args.model_dim)
    elif kind == "lstm":
        model = LSTMLanguageModel(len(vocabulary), args.model_dim)
    else:
        model = TinyTransformer(
            len(vocabulary), args.model_dim, args.num_heads, args.sequence_length
        )
    if args.summary:
        print(model_summary(model))
        return
    model(train.x[:2])
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = CrossEntropyLoss()
    directory = create_run(args.output_root, args.run_name, {**vars(args), "model": kind})
    (directory / "vocabulary.json").write_text(json.dumps(vocabulary.characters), encoding="utf-8")
    history = History(directory / "metrics.csv")
    train_loader = DataLoader(train, args.batch_size, shuffle=True, seed=args.seed)
    watch = (
        ParameterWatch(args.watch, tuple(map(int, args.watch_index.split(","))), args.log_every)
        if args.watch
        else None
    )
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            logits = model(x)
            logits = logits.reshape(-1, len(vocabulary))
            targets = y.reshape(-1)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            norm = gradient_norm(model.parameters())
            detect_nonfinite(model, loss.data, step)
            before = watch.capture(model, step) if watch else None
            optimizer.step()
            detect_nonfinite(model, step=step)
            if watch:
                log_watch(directory / "parameter_watch.jsonl", watch.report(model, before))
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} CE={float(loss.data):.5f} grad_norm={norm} lr={optimizer.lr}"
                )
            history.append(
                epoch=epoch,
                step=step,
                train_loss=float(loss.data),
                train_acc=accuracy(logits.data, targets),
                grad_norm=norm,
                param_norm=parameter_norm(model.parameters()),
                lr=optimizer.lr,
            )
            step += 1
        result = evaluate(model, DataLoader(val, args.batch_size), criterion, sequence=True)
        history.append(
            epoch=epoch, step=step - 1, val_loss=result["loss"], val_acc=result["accuracy"]
        )
        generated = sample_text(model, vocabulary, training_text[:1], context=args.sequence_length)
        print(
            f"epoch={epoch} CE={result['loss']:.5f} perplexity={perplexity(result['loss']):.3f}\n{generated}"
        )
        with (directory / "samples.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"epoch={epoch}\n{generated}\n")
        save_checkpoint(
            directory / "checkpoints" / "last.npz",
            model,
            optimizer,
            epoch=epoch,
            step=step,
            metadata={"config": vars(args), "model": kind, "vocabulary": vocabulary.characters},
        )
    plot_history(history, directory, args.show)
