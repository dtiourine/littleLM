import numpy as np


def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


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
            self.grad += unbroadcast(out.grad * 1, self.data.shape)
            other.grad += unbroadcast(out.grad * 1, other.data.shape)

        out._backward = backward

        return out

    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def backward():
            self.grad += unbroadcast(out.grad * 1, self.data.shape)
            other.grad += unbroadcast(out.grad * 1, other.data.shape)

        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = backward

        return out
    
    def sum(self, axis: int | None = None):
        out = Tensor(np.sum(self.data, axis=axis))
        out._prev = {self}
        
        def backward():
            grad = np.ones_like(self.data) * out.grad
            if axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += grad
        
        out._backward = backward 
        return out 

    def mean(self, axis: int | None = None):
        out = Tensor(np.mean(self.data, axis=axis))
        out._prev = {self}
        
        def backward():
            grad = np.ones_like(self.data) * out.grad / self.data.size 
            if axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += grad
            
        out._backward = backward 
        return out 
    
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def backward():
            self.grad += unbroadcast(1 * out.grad, self.data.shape)
            other.grad += unbroadcast(-1 * out.grad, other.data.shape)

        out._backward = backward

        return out

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def backward():
            self.grad += unbroadcast(-1 * out.grad, self.data.shape)

        out._backward = backward
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def backward():
            self.grad += unbroadcast(1 / other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(
                -self.data * other.data**-2 * out.grad, other.data.shape
            )

        out._backward = backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data / self.data)
        out._prev = {self, other}

        def backward():
            other.grad += unbroadcast(1 / self.data * out.grad, other.data.shape)
            self.grad += unbroadcast(
                -other.data * self.data**-2 * out.grad, self.data.shape
            )

        out._backward = backward
        return out

    def __pow__(self, exp: int):
        out = Tensor(self.data**exp)
        out._prev = {self}

        def backward():
            self.grad += unbroadcast(
                (exp) * self.data ** (exp - 1) * out.grad, self.data.shape
            )

        out._backward = backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0))
        out._prev = {self}

        def backward():
            self.grad += unbroadcast(
                (self.data > 0).astype(float) * out.grad, self.data.shape
            )

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
