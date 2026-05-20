import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data, dtype=float)

        self._prev = set()
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = backward

        return out

    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def backward():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad

        out._backward = backward

        return out

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def backward():
            self.grad += -1 * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def backward():
            self.grad += 1 / other.data * out.grad
            other.grad += -self.data * other.data**-2 * out.grad

        out._backward = backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data / self.data)
        out._prev = {self, other}

        def backward():
            other.grad += 1 / self.data * out.grad
            self.grad += -other.data * self.data**-2 * out.grad

        out._backward = backward
        return out

    def __pow__(self, exp: int):
        out = Tensor(self.data**exp)
        out._prev = {self}

        def backward():
            self.grad += (exp) * self.data ** (exp - 1) * out.grad

        out._backward = backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0))
        out._prev = {self}

        def backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = backward
        return out

    def backward(self):
        assert (
            self.data.ndim == 0 or self.data.size == 1
        ), "backward() should only be called on a scalar"

        visited = set()
        topo = []

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=float)

        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor({self.data})"

    def __str__(self):
        return f"Tensor({self.data})"
