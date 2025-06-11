# Standard Libray dependencies
# ...

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward


def test_CeluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.celu(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.celu(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.celu(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.celu(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_EluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.elu(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.elu(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.elu(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.elu(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_GeluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.gelu(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.gelu(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.gelu(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.gelu(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_GluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.glu(X, dim=-1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.glu(X, dim=-1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.glu(X, dim=-1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.glu(x_ref, dim=-1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_LeakyReluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.leaky_relu(X, negative_slope=0.01)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.leaky_relu(X, negative_slope=0.01).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.leaky_relu(X, negative_slope=0.01).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.leaky_relu(x_ref, negative_slope=0.01).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PreluKernelXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    weight: Tensor = torch.rand(3, requires_grad=True)
    O: Tensor = torch.nn.functional.prelu(X, weight)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert weight.hgrad[i].shape == (O.numel(),) + (weight.numel(),) * (i + 1)

    # First False
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    weight: Tensor = torch.rand(3, requires_grad=True)
    O: Tensor = torch.nn.functional.prelu(X, weight).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(weight.hgrad[0].flatten(), weight.grad.flatten())

    # Second derivative with cross terms
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    weight: Tensor = torch.rand(3, requires_grad=True)
    O: Tensor = torch.nn.functional.prelu(X, weight).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor, w_ref: Tensor) -> Tensor:
        return torch.nn.functional.prelu(x_ref, w_ref).sum()

    full_hess: Tensor = torch.autograd.functional.hessian(f, (X, weight))
    H: Tensor
    indeps: object
    shapes: object
    H00, indeps, shapes = op.fetch_hgrad([X, X], batch=False)
    H01, indeps, shapes = op.fetch_hgrad([X, weight], batch=False)
    H10, indeps, shapes = op.fetch_hgrad([weight, X], batch=False)
    H11, indeps, shapes = op.fetch_hgrad([weight, weight], batch=False)
    assert torch.allclose(H00.flatten(), full_hess[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hess[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hess[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hess[1][1].flatten(), atol=1e-4)


def test_ReluXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((5, 4), requires_grad=True)
    O: Tensor = torch.relu(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1), (
            X.hgrad[i].shape,
            (O.numel(),) + (X.numel(),) * (i + 1),
        )

    # First derivative
    X: Tensor = torch.randn((5, 4), requires_grad=True)
    O: Tensor = torch.relu(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.randn((5, 4), requires_grad=True)
    O: Tensor = torch.relu(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.relu(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_RreluWithNoiseXBackward0() -> None:
    order: int = 3
    # Shape test (deterministic mode)
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.rrelu(X, lower=0.1, upper=0.3, training=False)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.rrelu(X, lower=0.1, upper=0.3, training=False).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6)) - 0.5
    X.requires_grad_()
    O: Tensor = torch.nn.functional.rrelu(X, lower=0.1, upper=0.3, training=False).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.rrelu(
            x_ref, lower=0.1, upper=0.3, training=False
        ).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SiluXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.nn.functional.silu(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.nn.functional.silu(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.nn.functional.silu(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.silu(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
