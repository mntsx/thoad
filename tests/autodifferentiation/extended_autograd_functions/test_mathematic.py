# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward


def test_AbsXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = torch.abs(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = torch.abs(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = torch.abs(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.abs(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_NegXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = -X
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = (-X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.randn((4, 5), requires_grad=True)
    O: Tensor = (-X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return (-x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MeanXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((3, 4, 5), requires_grad=True)
    O: Tensor = X.mean()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (1,) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.randn((3, 4, 5), requires_grad=True)
    O: Tensor = X.mean()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative zero
    X: Tensor = torch.randn((3, 4, 5), requires_grad=True)
    O: Tensor = X.mean()
    op: Operator = backward(O, order=2, crossings=True)
    H: Tensor = op.fetch_hgrad([X], batch=False)[0].squeeze(0)
    assert torch.allclose(H, torch.zeros_like(H))


def test_MeanXBackward1() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((2, 3, 4), requires_grad=True)
    O: Tensor = X.mean(dim=1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.randn((2, 3, 4), requires_grad=True)
    O: Tensor = X.mean(dim=1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative zero
    X: Tensor = torch.randn((2, 3, 4), requires_grad=True)
    O: Tensor = X.mean(dim=1).sum()
    op: Operator = backward(O, order=2, crossings=True)
    H: Tensor = op.fetch_hgrad([X], batch=False)[0].squeeze(0)
    assert torch.allclose(H, torch.zeros_like(H))


def test_LogXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.log(X + 0.1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.log(X + 0.1).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = torch.log(X + 0.1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.log(x_ref + 0.1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_XlogyXBackward0() -> None:
    order: int = 2
    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X: Tensor = torch.rand((3, 4), requires_grad=reqx)
        Y: Tensor = torch.rand((3, 4), requires_grad=reqy)
        O: Tensor = torch.special.xlogy(X + 0.1, Y + 0.1)
        op: Operator = backward(O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand((3, 4), requires_grad=reqx)
        Y = torch.rand((3, 4), requires_grad=reqy)
        O = torch.special.xlogy(X + 0.1, Y + 0.1).sum()
        op = backward(O, order=1)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives only if both grads active
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    Y: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = torch.special.xlogy(X + 0.1, Y + 0.1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.special.xlogy(a_ref, b_ref).sum()

    full_hess: Tensor = torch.autograd.functional.hessian(f, (X, Y))
    H: Tensor
    indeps: object
    shapes: object
    H00, indeps, shapes = op.fetch_hgrad([X, X], batch=False)
    H01, indeps, shapes = op.fetch_hgrad([X, Y], batch=False)
    H10, indeps, shapes = op.fetch_hgrad([Y, X], batch=False)
    H11, indeps, shapes = op.fetch_hgrad([Y, Y], batch=False)
    assert torch.allclose(H00.flatten(), full_hess[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hess[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hess[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hess[1][1].flatten(), atol=1e-4)


def test_SigmoidXBackward0() -> None:
    order: int = 2
    # Shape test
    X: Tensor = torch.randn((5, 4), requires_grad=True)
    O: Tensor = torch.sigmoid(X)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative
    X: Tensor = torch.randn((5, 4), requires_grad=True)
    O: Tensor = torch.sigmoid(X).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    # X: Tensor = torch.randn((5, 4), requires_grad=True)
    X: Tensor = torch.randn((2, 2), requires_grad=True)
    O: Tensor = torch.sigmoid(X).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sigmoid(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
