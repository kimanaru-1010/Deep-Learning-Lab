import numpy as np

from minidl.core import Tensor
from minidl.utils.gradcheck import check_gradient


def check_layer(layer, arrays, *, atol=2e-5, rtol=2e-4):
    """Check every differentiable input and parameter using a fixed output cotangent."""
    inputs = [Tensor(np.asarray(a, dtype=np.float64), requires_grad=True) for a in arrays]
    layer.zero_grad()
    result = layer(*inputs)
    outputs = result if isinstance(result, tuple) else (result,)
    upstream = [np.linspace(0.2, 0.9, output.data.size).reshape(output.shape) for output in outputs]
    loss = sum((output * weight).sum() for output, weight in zip(outputs, upstream))
    loss.backward()
    analytical_inputs = [x.grad.copy() for x in inputs]
    analytical_parameters = [(name, p, p.grad.copy()) for name, p in layer.named_parameters()]

    def scalar_output(values):
        output = layer(*[Tensor(v) for v in values])
        outputs = output if isinstance(output, tuple) else (output,)
        return sum(float(np.sum(o.data * w)) for o, w in zip(outputs, upstream))

    for index, gradient in enumerate(analytical_inputs):

        def function(value):
            copied = [a.copy() for a in arrays]
            copied[index] = value
            return scalar_output(copied)

        report = check_gradient(function, arrays[index], gradient, atol=atol, rtol=rtol)
        assert report.passed, f"input {index}: {report}"
    for name, parameter, gradient in analytical_parameters:
        original = parameter.data.copy()

        def function(value):
            parameter.data[...] = value
            return scalar_output(arrays)

        try:
            report = check_gradient(function, original, gradient, atol=atol, rtol=rtol)
            assert report.passed, f"{name}: {report}"
        finally:
            parameter.data[...] = original
