import numpy as np
from numpy.typing import NDArray


def unbroadcast(grad: NDArray, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = {}

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        try:
            np.broadcast_shapes(self.data, other.data)
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
        other = other if isinstance(other, Tensor) else Tensor(other.data)

        try:
            np.broadcast_shapes(self.data, other.data)
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

        def _backward(self, other):
            self.grad += out.grad @ other.data.T
            other.grad = self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward(self):
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += np.broadcast_to(grad, self.data.shape()).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]

        def _backward(self):
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape()).copy() / n

        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data))
        out._prev = {self}
        
        def _backward(self):
            self.grad += out.grad * out.grad 
            
        out._backward = _backward 
        return out 
    
    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = {self}
        
        def _backward(self):
            self.grad += out.grad * (1.0 / self.data)
        
        out._backward = _backward 
        return out 
