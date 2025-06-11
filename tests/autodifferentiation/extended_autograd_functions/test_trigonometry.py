# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_SinXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sin(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sin(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sin(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sin(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_CosXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cos(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cos(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cos(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.cos(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TanXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tan(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tan(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tan(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.tan(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SinhXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sinh(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sinh(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.sinh(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sinh(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_CoshXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cosh(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cosh(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.cosh(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.cosh(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TanhXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tanh(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tanh(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.tanh(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.tanh(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
