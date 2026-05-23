import numpy as np
import pytest
from tensors.tensor import Tensor, unbroadcast


# ---------------------------------------------------------------------------
# Gradient checking helpers
# ---------------------------------------------------------------------------

def check_grads(*inputs_np, f, h=1e-5, atol=1e-5, rtol=1e-4):
    """
    Compare analytical gradients (via .backward()) against central-difference
    numerical gradients for every input array.
    """
    tensors = [Tensor(x) for x in inputs_np]
    out = f(*tensors)
    out.backward()
    analytical = [t.grad.copy() for t in tensors]

    numerical = []
    for i, x in enumerate(inputs_np):
        grad = np.zeros_like(x, dtype=float)
        for idx in np.ndindex(x.shape):
            perturbed_plus = [xi.copy() for xi in inputs_np]
            perturbed_plus[i][idx] = x[idx] + h
            loss_plus = f(*[Tensor(xi) for xi in perturbed_plus]).data.sum()

            perturbed_minus = [xi.copy() for xi in inputs_np]
            perturbed_minus[i][idx] = x[idx] - h
            loss_minus = f(*[Tensor(xi) for xi in perturbed_minus]).data.sum()

            grad[idx] = (loss_plus - loss_minus) / (2 * h)
        numerical.append(grad)

    for i, (ana, num) in enumerate(zip(analytical, numerical)):
        np.testing.assert_allclose(
            ana, num, atol=atol, rtol=rtol,
            err_msg=f"Gradient mismatch for input {i}"
        )


# ---------------------------------------------------------------------------
# unbroadcast
# ---------------------------------------------------------------------------

class TestUnbroadcast:
    def test_reduces_extra_leading_dims(self):
        grad = np.ones((3, 2, 4))
        result = unbroadcast(grad, (2, 4))
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result, np.full((2, 4), 3.0))

    def test_reduces_broadcast_size_one_dim(self):
        grad = np.ones((3, 1, 4)) * 2
        result = unbroadcast(grad, (3, 1, 4))
        assert result.shape == (3, 1, 4)

    def test_scalar_target_shape(self):
        grad = np.ones((2, 3))
        result = unbroadcast(grad, ())
        assert result.shape == ()
        assert result == 6.0

    def test_no_reduction_needed(self):
        grad = np.ones((2, 3))
        result = unbroadcast(grad, (2, 3))
        np.testing.assert_array_equal(result, grad)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_from_list(self):
        t = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t.data, [1.0, 2.0, 3.0])
        assert t.data.dtype == np.float64

    def test_from_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = Tensor(arr)
        np.testing.assert_array_equal(t.data, arr)

    def test_from_scalar(self):
        t = Tensor(5)
        assert t.data.shape == ()
        assert t.data == 5.0

    def test_grad_initialized_to_zeros(self):
        t = Tensor([[1, 2], [3, 4]])
        np.testing.assert_array_equal(t.grad, np.zeros((2, 2)))

    def test_prev_empty(self):
        t = Tensor([1.0])
        assert t._prev == set()

    def test_2d_shape_preserved(self):
        t = Tensor(np.ones((3, 4)))
        assert t.data.shape == (3, 4)


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

class TestAdd:
    def test_elementwise_forward(self):
        a, b = Tensor([1, 2, 3]), Tensor([4, 5, 6])
        c = a + b
        np.testing.assert_array_equal(c.data, [5, 7, 9])

    def test_scalar_operand(self):
        a = Tensor([1.0, 2.0])
        c = a + 10
        np.testing.assert_array_equal(c.data, [11.0, 12.0])

    def test_reflected_radd(self):
        a = Tensor([1.0, 2.0])
        c = 10 + a
        np.testing.assert_array_equal(c.data, [11.0, 12.0])

    def test_broadcast_shape(self):
        a = Tensor(np.ones((3, 1)))
        b = Tensor(np.ones((1, 4)))
        c = a + b
        assert c.data.shape == (3, 4)

    def test_incompatible_shapes_raise(self):
        a, b = Tensor(np.ones((2, 3))), Tensor(np.ones((4, 5)))
        with pytest.raises(ValueError):
            a + b

    def test_prev_set(self):
        a, b = Tensor([1.0]), Tensor([2.0])
        c = a + b
        assert a in c._prev and b in c._prev

    def test_grad_elementwise(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            f=lambda a, b: a + b,
        )

    def test_grad_broadcast(self):
        check_grads(
            np.ones((3, 1)),
            np.ones((1, 4)),
            f=lambda a, b: a + b,
        )

    def test_grad_scalar(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            f=lambda a: a + 5,
        )


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------

class TestSub:
    def test_forward(self):
        a, b = Tensor([5.0, 7.0]), Tensor([2.0, 3.0])
        np.testing.assert_array_equal((a - b).data, [3.0, 4.0])

    def test_scalar_operand(self):
        a = Tensor([5.0, 7.0])
        np.testing.assert_array_equal((a - 2).data, [3.0, 5.0])

    def test_reflected_rsub(self):
        a = Tensor([1.0, 2.0])
        c = 10 - a
        np.testing.assert_array_equal(c.data, [9.0, 8.0])

    def test_grad(self):
        check_grads(
            np.array([3.0, 1.0, 4.0]),
            np.array([1.0, 5.0, 9.0]),
            f=lambda a, b: a - b,
        )

    def test_grad_broadcast(self):
        check_grads(
            np.ones((2, 3)),
            np.ones((1, 3)),
            f=lambda a, b: a - b,
        )


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

class TestMul:
    def test_elementwise_forward(self):
        a, b = Tensor([2.0, 3.0]), Tensor([4.0, 5.0])
        np.testing.assert_array_equal((a * b).data, [8.0, 15.0])

    def test_scalar_operand(self):
        a = Tensor([2.0, 3.0])
        np.testing.assert_array_equal((a * 3).data, [6.0, 9.0])

    def test_reflected_rmul(self):
        a = Tensor([2.0, 3.0])
        np.testing.assert_array_equal((3 * a).data, [6.0, 9.0])

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError):
            Tensor(np.ones((2, 3))) * Tensor(np.ones((4, 5)))

    def test_grad(self):
        check_grads(
            np.array([2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 7.0]),
            f=lambda a, b: a * b,
        )

    def test_grad_broadcast(self):
        check_grads(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[2.0, 0.5]]),
            f=lambda a, b: a * b,
        )


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------

class TestDiv:
    def test_forward(self):
        a, b = Tensor([6.0, 8.0]), Tensor([2.0, 4.0])
        np.testing.assert_allclose((a / b).data, [3.0, 2.0])

    def test_scalar_divisor(self):
        a = Tensor([6.0, 9.0])
        np.testing.assert_allclose((a / 3).data, [2.0, 3.0])

    def test_reflected_rtruediv(self):
        a = Tensor([2.0, 4.0])
        np.testing.assert_allclose((8.0 / a).data, [4.0, 2.0])

    def test_grad(self):
        check_grads(
            np.array([6.0, 8.0, 10.0]),
            np.array([2.0, 4.0, 5.0]),
            f=lambda a, b: a / b,
        )


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

class TestNeg:
    def test_forward(self):
        a = Tensor([1.0, -2.0, 3.0])
        np.testing.assert_array_equal((-a).data, [-1.0, 2.0, -3.0])

    def test_grad(self):
        check_grads(
            np.array([1.0, -2.0, 3.0]),
            f=lambda a: -a,
        )


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

class TestPow:
    def test_square(self):
        a = Tensor([2.0, 3.0])
        np.testing.assert_array_equal((a ** 2).data, [4.0, 9.0])

    def test_fractional_exponent(self):
        a = Tensor([4.0, 9.0])
        np.testing.assert_allclose((a ** 0.5).data, [2.0, 3.0])

    def test_non_scalar_exponent_raises(self):
        a = Tensor([2.0])
        with pytest.raises(AssertionError):
            a ** a  # type: ignore

    def test_grad_square(self):
        check_grads(
            np.array([2.0, 3.0, 4.0]),
            f=lambda a: a ** 2,
        )

    def test_grad_cube(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            f=lambda a: a ** 3,
        )


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

class TestMatMul:
    def test_2d_forward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
        np.testing.assert_allclose((a @ b).data, expected)

    def test_vector_matmul(self):
        a = Tensor([[1.0, 2.0, 3.0]])
        b = Tensor([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose((a @ b).data, [[14.0]])

    def test_grad(self):
        check_grads(
            np.random.randn(3, 4),
            np.random.randn(4, 2),
            f=lambda a, b: a @ b,
        )

    def test_grad_square(self):
        check_grads(
            np.random.randn(3, 3),
            np.random.randn(3, 3),
            f=lambda a, b: a @ b,
        )


# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------

class TestSum:
    def test_global_sum(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.sum().data == 10.0

    def test_axis0_sum(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(a.sum(axis=0).data, [4.0, 6.0])

    def test_axis1_sum(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(a.sum(axis=1).data, [3.0, 7.0])

    def test_keepdims(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.sum(axis=1, keepdims=True)
        assert result.data.shape == (2, 1)

    def test_grad_global(self):
        check_grads(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            f=lambda a: a.sum(),
        )

    def test_grad_axis0(self):
        check_grads(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            f=lambda a: a.sum(axis=0),
        )

    def test_grad_axis1_keepdims(self):
        check_grads(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            f=lambda a: a.sum(axis=1, keepdims=True),
        )


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_global_mean(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        assert a.mean().data == 2.5

    def test_axis_mean(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(a.mean(axis=0).data, [2.0, 3.0])

    def test_keepdims(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.mean(axis=1, keepdims=True)
        assert result.data.shape == (2, 1)

    def test_grad_global(self):
        check_grads(
            np.array([1.0, 2.0, 3.0, 4.0]),
            f=lambda a: a.mean(),
        )

    def test_grad_axis0(self):
        check_grads(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            f=lambda a: a.mean(axis=0),
        )

    def test_grad_axis1_keepdims(self):
        check_grads(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            f=lambda a: a.mean(axis=1, keepdims=True),
        )


# ---------------------------------------------------------------------------
# Exp
# ---------------------------------------------------------------------------

class TestExp:
    def test_forward(self):
        a = Tensor([0.0, 1.0])
        np.testing.assert_allclose(a.exp().data, [1.0, np.e])

    def test_2d_forward(self):
        a = Tensor([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_allclose(a.exp().data, np.exp([[0, 1], [2, 3]]))

    def test_grad(self):
        check_grads(
            np.array([0.0, 0.5, 1.0, -1.0]),
            f=lambda a: a.exp(),
        )


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

class TestLog:
    def test_forward(self):
        a = Tensor([1.0, np.e])
        np.testing.assert_allclose(a.log().data, [0.0, 1.0])

    def test_grad(self):
        check_grads(
            np.array([0.5, 1.0, 2.0, 4.0]),
            f=lambda a: a.log(),
        )


# ---------------------------------------------------------------------------
# Sqrt
# ---------------------------------------------------------------------------

class TestSqrt:
    def test_forward(self):
        a = Tensor([4.0, 9.0, 16.0])
        np.testing.assert_allclose(a.sqrt().data, [2.0, 3.0, 4.0])

    def test_grad(self):
        check_grads(
            np.array([1.0, 2.0, 4.0, 9.0]),
            f=lambda a: a.sqrt(),
        )


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_2d_default(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = a.transpose()
        assert t.data.shape == (3, 2)
        np.testing.assert_array_equal(t.data, [[1, 4], [2, 5], [3, 6]])

    def test_T_property(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(a.T.data, a.transpose().data)

    def test_3d_axes(self):
        a = Tensor(np.ones((2, 3, 4)))
        t = a.transpose(axes=(2, 0, 1))
        assert t.data.shape == (4, 2, 3)

    def test_grad_2d(self):
        check_grads(
            np.random.randn(3, 4),
            f=lambda a: a.transpose(),
        )

    def test_grad_3d_axes(self):
        check_grads(
            np.random.randn(2, 3, 4),
            f=lambda a: a.transpose(axes=(2, 0, 1)),
        )

    def test_grad_T_property(self):
        check_grads(
            np.random.randn(3, 4),
            f=lambda a: a.T,
        )


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

class TestReshape:
    def test_forward_tuple(self):
        a = Tensor(np.arange(6.0))
        b = a.reshape(2, 3)
        assert b.data.shape == (2, 3)

    def test_forward_single_tuple_arg(self):
        a = Tensor(np.arange(6.0))
        b = a.reshape((2, 3))
        assert b.data.shape == (2, 3)

    def test_values_preserved(self):
        a = Tensor(np.arange(6.0))
        b = a.reshape(2, 3)
        np.testing.assert_array_equal(b.data, [[0, 1, 2], [3, 4, 5]])

    def test_grad(self):
        check_grads(
            np.arange(6.0),
            f=lambda a: a.reshape(2, 3),
        )

    def test_grad_flatten(self):
        check_grads(
            np.random.randn(2, 3, 4),
            f=lambda a: a.reshape(24),
        )


# ---------------------------------------------------------------------------
# Indexing (__getitem__)
# ---------------------------------------------------------------------------

class TestGetitem:
    def test_integer_index(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(a[0].data, [1.0, 2.0])

    def test_slice_index(self):
        a = Tensor([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(a[1:3].data, [20.0, 30.0])

    def test_tensor_index(self):
        a = Tensor([10.0, 20.0, 30.0])
        idx = Tensor([0, 2])
        np.testing.assert_array_equal(a[idx].data, [10.0, 30.0])

    def test_grad_simple(self):
        check_grads(
            np.array([1.0, 2.0, 3.0, 4.0]),
            f=lambda a: a[1:3],
        )

    def test_grad_repeated_index(self):
        a = Tensor([1.0, 2.0, 3.0])
        out = a[np.array([0, 0, 2])]
        out.backward()
        np.testing.assert_array_equal(a.grad, [2.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Max
# ---------------------------------------------------------------------------

class TestMax:
    def test_global_max(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        assert a.max().data == 5.0

    def test_axis0_max(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        np.testing.assert_array_equal(a.max(axis=0).data, [3.0, 5.0])

    def test_axis1_max(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        np.testing.assert_array_equal(a.max(axis=1).data, [5.0, 3.0])

    def test_keepdims(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        result = a.max(axis=1, keepdims=True)
        assert result.data.shape == (2, 1)

    def test_grad_global(self):
        check_grads(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            f=lambda a: a.max(),
        )

    def test_grad_axis0(self):
        check_grads(
            np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]),
            f=lambda a: a.max(axis=0),
        )

    def test_grad_axis1_keepdims(self):
        check_grads(
            np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]),
            f=lambda a: a.max(axis=1, keepdims=True),
        )


# ---------------------------------------------------------------------------
# Backward / zero_grad
# ---------------------------------------------------------------------------

class TestBackward:
    def test_scalar_backward_sets_grad_one(self):
        a = Tensor(3.0)
        b = a * a
        b.backward()
        assert a.grad == 6.0

    def test_chain_rule(self):
        a = Tensor(2.0)
        b = a * a * a
        b.backward()
        np.testing.assert_allclose(a.grad, 12.0, atol=1e-10)

    def test_multi_use_accumulates_grad(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = a + a
        b.backward()
        np.testing.assert_array_equal(a.grad, [2.0, 2.0])

    def test_grad_accumulates_across_backward_calls(self):
        a = Tensor(np.array([1.0, 2.0]))
        (a * 2).backward()
        (a * 3).backward()
        np.testing.assert_array_equal(a.grad, [5.0, 5.0])

    def test_zero_grad_resets_all_nodes(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        c = a + b
        c.backward()
        c.zero_grad()
        np.testing.assert_array_equal(a.grad, [0.0, 0.0])
        np.testing.assert_array_equal(b.grad, [0.0, 0.0])
        np.testing.assert_array_equal(c.grad, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Composite / chain rule expressions
# ---------------------------------------------------------------------------

class TestComposite:
    def test_softmax_forward(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        e = x.exp()
        s = e / e.sum()
        expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
        np.testing.assert_allclose(s.data, expected, atol=1e-10)

    def test_softmax_grad(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            f=lambda x: (x.exp() / x.exp().sum()),
        )

    def test_mse_loss_forward(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.5, 1.5, 2.5]))
        loss = ((pred - target) ** 2).mean()
        expected = np.mean((np.array([1, 2, 3]) - np.array([1.5, 1.5, 2.5])) ** 2)
        np.testing.assert_allclose(loss.data, expected)

    def test_mse_loss_grad(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 1.5, 2.5]),
            f=lambda p, t: ((p - t) ** 2).mean(),
        )

    def test_linear_layer_forward(self):
        X = Tensor(np.ones((2, 3)))
        W = Tensor(np.eye(3, 2))
        b = Tensor(np.array([0.1, 0.2]))
        out = X @ W + b
        assert out.data.shape == (2, 2)

    def test_linear_layer_grad(self):
        np.random.seed(42)
        check_grads(
            np.random.randn(4, 3),
            np.random.randn(3, 2),
            np.random.randn(2),
            f=lambda X, W, b: (X @ W + b).sum(),
        )

    def test_log_softmax_grad(self):
        check_grads(
            np.array([1.0, 2.0, 3.0]),
            f=lambda x: (x - x.max()).exp() / (x - x.max()).exp().sum(),
        )

    def test_deeply_nested_chain(self):
        check_grads(
            np.array([0.5, 1.0, 1.5]),
            f=lambda x: ((x ** 2).exp().log() * x).sum(),
        )

    def test_layer_norm_style(self):
        check_grads(
            np.random.randn(4),
            f=lambda x: (x - x.mean()) / (((x - x.mean()) ** 2).mean() + 1e-5).sqrt(),
            atol=1e-4,
        )
