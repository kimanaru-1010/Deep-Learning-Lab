import numpy as np
import pytest

from minidl.core import Tensor
from minidl.utils.gradcheck import check_gradient

pytestmark = pytest.mark.student


def test_arithmetic_exp_log_and_gradient():
    x = Tensor([1.0, 2.0], True)
    result = ((3 - x) / 2 + (-x)).exp().log().sum()
    assert float(result.data) == pytest.approx(-1.5)
    result.backward()
    np.testing.assert_allclose(x.grad, [-1.5, -1.5])


def test_single_op_multiple_parents_and_chain():
    x, w = Tensor(2.0, True), Tensor(3.0, True)
    result = (x * w + 1) ** 2
    assert float(result.data) == 49
    result.backward()
    assert float(x.grad) == 42 and float(w.grad) == 28


def test_branch_reused_node_and_accumulation():
    x = Tensor(2.0, True)
    loss = x * x + 3 * x
    loss.backward()
    assert float(x.grad) == 7
    (x * 2).backward()
    assert float(x.grad) == 9
    x.zero_grad()
    assert x.grad is None


def test_shared_intermediate():
    x = Tensor(2.0, True)
    a = x * x
    (a + a * 3).backward()
    assert float(x.grad) == 16


def test_constant_and_broadcast_reduction():
    x = Tensor(np.ones((2, 3)), True)
    b = Tensor(np.array([1.0, 2.0, 3.0]), True)
    constant = Tensor(np.ones((2, 3)))
    ((x + b) * constant).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
    np.testing.assert_allclose(b.grad, [2, 2, 2])
    assert constant.grad is None


def test_matmul_and_shape_ops_gradients():
    values = np.array([[0.2, 0.8], [-0.5, 0.7]])
    weight = Tensor([[2.0], [-1.0]])
    x = Tensor(values, True)
    loss = (x @ weight).transpose().reshape(-1).mean()
    loss.backward()
    report = check_gradient(
        lambda a: ((Tensor(a) @ weight).transpose().reshape(-1).mean()).data, values, x.grad
    )
    assert report.passed, report
