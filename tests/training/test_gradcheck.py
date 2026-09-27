import numpy as np
import pytest

from minidl.utils.gradcheck import check_gradient, numerical_gradient, relative_error


def test_known_polynomial_and_no_input_mutation():
    x = np.array([[0.2, -0.4], [1.2, -1.0]])
    original = x.copy()
    report = check_gradient(lambda z: np.sum(z**3), x, 3 * x**2)
    assert report.passed and report.absolute_error < 1e-8
    assert "PASS" in str(report)
    np.testing.assert_array_equal(x, original)
    assert not check_gradient(lambda z: np.sum(z**3), x, np.zeros_like(x)).passed


def test_scalar_zero_and_invalid_inputs():
    assert numerical_gradient(lambda z: z**2, np.array(2.0)) == pytest.approx(4)
    assert relative_error(np.zeros(2), np.zeros(2)).max() == 0
    with pytest.raises(ValueError):
        numerical_gradient(lambda x: x, np.ones(2))
    with pytest.raises(ValueError):
        numerical_gradient(lambda x: x.sum(), np.ones(2), epsilon=0)
    with pytest.raises(ValueError):
        relative_error(np.ones(2), np.ones(3))
    assert not check_gradient(lambda z: np.sum(z), np.ones(2), [np.nan, 1]).passed
