import numpy as np
from numpy.typing import NDArray

from littlelm.backend import xp


def unbroadcast(grad: NDArray, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data):
        self.data = xp.asarray(data, dtype=xp.float64)
        self.grad = xp.zeros_like(self.data, dtype=float)

        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            xp.broadcast_shapes(self.data.shape, other.data.shape)
        except ValueError:
            raise ValueError(
                f"Tensor shapes are not broadcastable: {self.data.shape} and {other.data.shape}"
            )

        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            xp.broadcast_shapes(self.data.shape, other.data.shape)
        except ValueError:
            raise ValueError(
                f"Tensor shapes are not broadcastable: {self.data.shape} and {other.data.shape}"
            )

        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            grad_self = out.grad @ other.data.swapaxes(-1, -2)
            grad_other = self.data.swapaxes(-1, -2) @ out.grad

            self.grad += unbroadcast(grad_self, self.data.shape)
            other.grad += unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(xp.maximum(self.data, 0))
        out._prev = {self}

        def _backward():
            self.grad += (self.data > 0).astype(xp.float64) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(xp.tanh(self.data))
        out._prev = {self}

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def gelu(self):
        c = xp.sqrt(2.0 / xp.pi)
        inner = c * (self + 0.044715 * (self**3))

        tanh_inner = inner.tanh()
        return 0.5 * self * (1.0 + tanh_inner)

    def softmax(self, axis=-1):
        shifted = self - self.max(axis=axis, keepdims=True)
        exp_shifted = shifted.exp()
        return exp_shifted / exp_shifted.sum(axis=axis, keepdims=True)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                grad = xp.expand_dims(grad, axis=axis)

            self.grad += xp.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = xp.expand_dims(grad, axis=axis)
            self.grad += xp.broadcast_to(grad, self.data.shape).copy() / n

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(xp.exp(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(xp.log(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.grad * (1.0 / self.data)

        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(xp.sqrt(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.grad * (0.5 / out.data)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(xp.transpose(self.data, axes=axes))
        out._prev = {self}

        def _backward():
            if axes is None:
                self.grad += xp.transpose(out.grad)
            else:
                inverse_axes = xp.argsort(axes)
                self.grad += xp.transpose(out.grad, axes=inverse_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        out = Tensor(self.data.reshape(shape))
        out._prev = {self}

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __getitem__(self, indices):
        if isinstance(indices, Tensor):
            indices = indices.data.astype(int)

        out = Tensor(self.data[indices])
        out._prev = {self}

        def _backward():
            xp.add.at(self.grad, indices, out.grad)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = xp.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data)
        out._prev = {self}

        def _backward():
            if axis is None:
                mask = (self.data == out_data).astype(float)
                mask /= mask.sum()
                self.grad += mask * out.grad
            else:
                max_keepdims = xp.max(self.data, axis=axis, keepdims=True)
                mask = (self.data == max_keepdims).astype(float)
                mask /= mask.sum(axis=axis, keepdims=True)

                grad = out.grad
                if not keepdims:
                    grad = xp.expand_dims(grad, axis=axis)
                self.grad += mask * grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(-out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast((1.0 / other.data) * out.grad, self.data.shape)
            other.grad += unbroadcast(
                -self.data / (other.data**2) * out.grad, other.data.shape
            )

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "exponent must be a scalar"
        out = Tensor(self.data**exponent)
        out._prev = {self}

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def backward(self):
        visited = set()
        topo = []

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = xp.ones_like(self.data, dtype=float)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        visited = set()
        topo = []

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for v in topo:
            v.grad = xp.zeros_like(v.data, dtype=float)

    @property
    def shape(self):
        return self.data.shape


def stack(tensors, axis=0):
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    raw_data = [t.data for t in tensors]
    out = Tensor(xp.stack(raw_data, axis=axis))

    out._prev = set(tensors)

    def _backward():
        grad_slices = xp.split(out.grad, len(tensors), axis=axis)
        grads = [xp.squeeze(g, axis=axis) for g in grad_slices]

        for t, g in zip(tensors, grads):
            t.grad += g

    out._backward = _backward
    return out
