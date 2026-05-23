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

    def sqrt(self):
        out = Tensor(np.sqrt(self.data))
        out._prev = {self}

        def _backward(self):
            self.grad += out.grad * (0.5 / out.data)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes=axes))
        out._prev = {self}

        def _backward():
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                inverse_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, axes=inverse_axes)

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
    
    

# def test_reshape():
#     x_data = np.random.randn(2, 6)
#     x = Tensor(x_data)
#     L = x.reshape(3, 4).sum()
#     L.backward()
    
#     xt = to_torch(x_data)
#     Lt = xt.reshape(3, 4).sum()
#     Lt.backward()
    
#     assert_close(x.grad, xt.grad.numpy(), "reshape x.grad")
#     print("reshape test passed")

# if __name__ == "__main__":
#     test_reshape()