# Standard Libray dependencies
# ...

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward


def test_SoftmaxXBackward0() -> None:
    order: int = 2
    # Shape test along last dim
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softmax(X, dim=1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softmax(X, dim=1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softmax(X, dim=1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.softmax(x_ref, dim=1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_LogSoftmaxXBackward0() -> None:
    order: int = 2
    # Shape test along last dim
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.log_softmax(X, dim=1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.log_softmax(X, dim=1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.log_softmax(X, dim=1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.log_softmax(x_ref, dim=1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SoftplusXBackward0() -> None:
    order: int = 2
    # Shape test elementwise
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softplus(X - 0.5)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softplus(X - 0.5).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X: Tensor = torch.rand((3, 4, 2), requires_grad=True)
    O: Tensor = torch.nn.functional.softplus(X - 0.5).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x_ref - 0.5).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-3)
