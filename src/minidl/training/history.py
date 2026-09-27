import csv
from pathlib import Path

FIELDS = [
    "epoch",
    "step",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "grad_norm",
    "param_norm",
    "lr",
]


class History:
    def __init__(self, path=None):
        self.rows = []
        self.path = Path(path) if path is not None else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def append(self, **metrics):
        if set(metrics) - set(FIELDS):
            raise ValueError("Unknown metric field")
        row = {name: metrics.get(name) for name in FIELDS}
        self.rows.append(row)
        if self.path:
            with self.path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)

    @classmethod
    def read(cls, path):
        history = cls()
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                history.rows.append({k: float(v) if v else None for k, v in row.items()})
        return history
