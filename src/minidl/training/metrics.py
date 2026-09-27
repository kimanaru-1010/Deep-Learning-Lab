import numpy as np


def accuracy(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.shape[:-1] != labels.shape or labels.size == 0:
        raise ValueError("Expected logits (...,classes) and nonempty labels (...)")
    return float(np.mean(np.argmax(logits, axis=-1) == labels))


def confusion_matrix(y_true, y_pred, num_classes):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.shape != pred.shape or true.ndim != 1 or num_classes <= 0:
        raise ValueError("Expected equal-length vectors and positive class count")
    if true.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("Labels must be integers")
    if (
        np.any(true < 0)
        or np.any(pred < 0)
        or np.any(true >= num_classes)
        or np.any(pred >= num_classes)
    ):
        raise ValueError("Class index out of range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def perplexity(loss):
    return float(np.exp(loss)) if loss < 700 else float("inf")
