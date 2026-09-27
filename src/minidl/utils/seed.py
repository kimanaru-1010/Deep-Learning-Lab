import random

import numpy as np


def set_seed(seed):
    """Seed Python and legacy NumPy initializers; return a local Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
