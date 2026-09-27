from .parameter import Parameter


class Module:
    """Explicit traversal of attributes, lists, tuples and dictionaries.

    Shared parameters are returned once. No automatic registration magic.
    """

    def __init__(self):
        self.training = True
        self._buffer_names = []

    def register_buffer(self, name, array):
        """Explicit non-gradient state (e.g. BatchNorm running statistics)."""
        if name in self._buffer_names or hasattr(self, name):
            raise ValueError(f"Attribute already exists: {name}")
        self._buffer_names.append(name)
        setattr(self, name, array)

    def named_buffers(self):
        seen = set()

        def walk(value, prefix):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Module):
                for name in value._buffer_names:
                    yield f"{prefix}.{name}" if prefix else name, getattr(value, name)
                for name, item in vars(value).items():
                    yield from walk(item, f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple, dict)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for name, item in items:
                    yield from walk(item, f"{prefix}.{name}")

        yield from walk(self, "")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must define forward")

    def named_parameters(self):
        seen = set()

        def walk(value, name):
            if id(value) in seen:
                return
            if isinstance(value, (Parameter, Module, list, tuple, dict)):
                seen.add(id(value))
            if isinstance(value, Parameter):
                yield name, value
            elif isinstance(value, Module):
                for key, item in vars(value).items():
                    yield from walk(item, f"{name}.{key}" if name else key)
            elif isinstance(value, (list, tuple, dict)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for key, item in items:
                    yield from walk(item, f"{name}.{key}")

        yield from walk(self, "")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode=True):
        seen = set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, Module):
                value.training = mode
                for item in vars(value).values():
                    visit(item)
            elif isinstance(value, (list, tuple, dict)):
                for item in value.values() if isinstance(value, dict) else value:
                    visit(item)

        visit(self)
        return self

    def eval(self):
        return self.train(False)


def count_parameters(model):
    return sum(p.data.size for p in model.parameters() if p.requires_grad)


def model_summary(model):
    rows = [f"{type(model).__name__}: parameter | shape | count"]
    rows.extend(f"{name} | {p.shape} | {p.data.size}" for name, p in model.named_parameters())
    rows.append(f"Total trainable parameters: {count_parameters(model)}")
    return "\n".join(rows)
