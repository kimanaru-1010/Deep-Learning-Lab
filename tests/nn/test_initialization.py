import numpy as np
import pytest

from minidl.nn.initialization import he, xavier
from minidl.utils.seed import set_seed

pytestmark = pytest.mark.student


@pytest.mark.parametrize("initialize", [xavier, he])
def test_initializer_shape_dtype_finite_reproducible(initialize):
    set_seed(7)
    first = initialize((3, 4), dtype=np.float64)
    set_seed(7)
    second = initialize((3, 4), dtype=np.float64)
    assert first.shape == (3, 4) and first.dtype == np.float64
    assert np.isfinite(first).all() and np.std(first) > 0
    np.testing.assert_array_equal(first, second)
