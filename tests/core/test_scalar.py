import pytest

from experiments.scalar import Value

pytestmark = pytest.mark.student


def test_known_scalar_chain():
    x, w = Value(2), Value(3)
    loss = (x * w + 1) ** 2
    assert loss.data == 49
    loss.backward()
    assert x.grad == 42 and w.grad == 28


def test_scalar_branch_and_shared_intermediate():
    x = Value(2)
    (x**2 + 3 * x).backward()
    assert x.grad == 7
    x.zero_grad()
    assert x.grad == 0
    a = x * x
    (a + a).backward()
    assert x.grad == 8
