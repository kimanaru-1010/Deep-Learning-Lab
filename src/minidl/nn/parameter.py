from minidl.core.tensor import Tensor


class Parameter(Tensor):
    """Trainable storage; gradient computation is still student work."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)
