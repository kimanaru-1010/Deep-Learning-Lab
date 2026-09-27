"""Central finite differences: a verification tool, never a training backend."""

from dataclasses import dataclass

import numpy as np


def numerical_gradient(function, x, epsilon=1e-6):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    work = np.array(x, dtype=np.float64, copy=True)
    if not np.isfinite(work).all():
        raise ValueError("x must be finite")
    gradient = np.empty_like(work)
    for index in np.ndindex(work.shape):
        original = work[index]
        work[index] = original + epsilon
        plus = np.asarray(function(work.copy()))
        work[index] = original - epsilon
        minus = np.asarray(function(work.copy()))
        work[index] = original
        if plus.shape != () or minus.shape != ():
            raise ValueError("function must return a scalar; use a fixed upstream dot product")
        if not np.isfinite(plus) or not np.isfinite(minus):
            raise ValueError("Non-finite function output during gradcheck")
        gradient[index] = (float(plus) - float(minus)) / (2 * epsilon)
    return gradient


def relative_error(actual, expected, floor=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("Gradient shapes differ")
    return np.abs(actual - expected) / np.maximum(floor, np.abs(actual) + np.abs(expected))


@dataclass
class GradientReport:
    passed: bool
    absolute_error: float
    relative_error: float
    numerical: np.ndarray

    def __str__(self):
        return (
            f"{'PASS' if self.passed else 'FAIL'} gradient check: "
            f"max absolute={self.absolute_error:.3e}, max relative={self.relative_error:.3e}"
        )


def check_gradient(function, x, analytical, epsilon=1e-6, atol=1e-6, rtol=1e-4):
    numeric = numerical_gradient(function, x, epsilon)
    analytical = np.asarray(analytical)
    errors = relative_error(analytical, numeric)
    passed = bool(
        np.isfinite(analytical).all() and np.allclose(analytical, numeric, atol=atol, rtol=rtol)
    )
    return GradientReport(
        passed,
        float(np.max(np.abs(analytical - numeric), initial=0)),
        float(np.max(errors, initial=0)),
        numeric,
    )
