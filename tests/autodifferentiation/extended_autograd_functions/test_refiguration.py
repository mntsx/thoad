# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_ExpandXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((2, 1), requires_grad=True)
    O: Tensor = X.expand(2, 3)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((2, 1), requires_grad=True)
    O: Tensor = X.expand(2, 3).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 1), requires_grad=True)
    O: Tensor = X.expand(2, 3).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.expand(2, 3).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PermuteXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.permute(1, 0, 2)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.permute(1, 0, 2).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.permute(1, 0, 2).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.permute(1, 0, 2).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ViewXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.view(6, 4)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.view(6, 4).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.view(6, 4).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.view(6, 4).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_RepeatXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((2, 2), requires_grad=True)
    O: Tensor = X.repeat(2, 3)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((2, 2), requires_grad=True)
    O: Tensor = X.repeat(2, 3).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 2), requires_grad=True)
    O: Tensor = X.repeat(2, 3).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.repeat(2, 3).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ReshapeXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.reshape(4, 3)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.reshape(4, 3).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.reshape(4, 3).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.reshape(4, 3).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward0() -> None:
    order: int = 2
    # Shape test: remove all singleton dims
    X: Tensor = torch.rand((1, 3, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((1, 3, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze().sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((1, 3, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze().sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.squeeze().sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward1() -> None:
    order: int = 2

    # Shape test: squeeze specific dim
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.squeeze(1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # Shape test: squeeze specific dim
    X: Tensor = torch.rand((2, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze(1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((2, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze(1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 1, 4), requires_grad=True)
    O: Tensor = X.squeeze(1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.squeeze(1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward2() -> None:
    order: int = 2
    # No-op squeeze when no singleton dims
    X: Tensor = torch.rand((2, 3, 4, 5, 6), requires_grad=True)
    O: Tensor = X.squeeze(dim=(1, 3))
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # No-op squeeze when no singleton dims
    X: Tensor = torch.rand((2, 1, 4, 1, 6), requires_grad=True)
    O: Tensor = X.squeeze(dim=(1, 3))
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((2, 1, 4, 1, 6), requires_grad=True)
    O: Tensor = X.squeeze(dim=(1, 3)).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 1, 4, 1, 6), requires_grad=True)
    O: Tensor = X.squeeze(dim=(1, 3)).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.squeeze(dim=(1, 3)).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TXBackward0() -> None:
    order: int = 2
    # Shape test: T() is transpose for 2D
    X: Tensor = torch.rand((5, 7), requires_grad=True)
    O: Tensor = X.T
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 7), requires_grad=True)
    O: Tensor = X.T.sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 7), requires_grad=True)
    O: Tensor = X.T.sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.T.sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TransposeXBackward0() -> None:
    order: int = 2
    # Shape test: general transpose dims
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.transpose(0, 2)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.transpose(0, 2).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O: Tensor = X.transpose(0, 2).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.transpose(0, 2).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_UnsqueezeXBackward0() -> None:
    order: int = 2
    # Shape test: add singleton dim
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.unsqueeze(1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.unsqueeze(1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.unsqueeze(1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.unsqueeze(1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
