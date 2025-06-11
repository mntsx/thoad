# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_PowXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.pow(X, 3)
    backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.pow(X, 3).sum()
    backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.pow(X, 3).sum()
    backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.pow(x_ref, 3).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PowXBackward1() -> None:
    order: int = 3
    # Shape tests with different grad requirements
    X_shapes: list[tuple[int, int]] = [(4, 6), (1, 6), (4, 1)]
    Y_shapes: list[tuple[int, int]] = [(4, 6), (4, 6), (4, 6)]
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X: Tensor = torch.rand(xs, requires_grad=reqx)
            Y: Tensor = torch.rand(ys, requires_grad=reqy)
            O: Tensor = torch.pow(X, Y)
            op: Operator = backward(O, order=order, crossings=True)
            for i in range(order):
                if reqx:
                    assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
                if reqy:
                    assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.pow(X, Y).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives including cross terms
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.pow(X, Y).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.pow(x_ref, y_ref).sum()

    full_hessian: Tuple[Tuple[Tensor]] = torch.autograd.functional.hessian(f, (X, Y))
    # Fetch computed Hessians
    H00: Tensor
    H01: Tensor
    H10: Tensor
    H11: Tensor
    indeps: Indep
    shapes: Shape
    H00, indeps, shapes = op.fetch_hgrad([X, X], batch=False)
    H01, indeps, shapes = op.fetch_hgrad([X, Y], batch=False)
    H10, indeps, shapes = op.fetch_hgrad([Y, X], batch=False)
    H11, indeps, shapes = op.fetch_hgrad([Y, Y], batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)


def test_SqrtBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.sqrt(X)
    backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X = torch.rand((4, 6), requires_grad=True)
    O = torch.sqrt(X).sum()
    backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative vs hessian
    X = torch.rand((4, 6), requires_grad=True)
    O = torch.sqrt(X).sum()
    backward(O, order=2, crossings=True)

    def g(x_ref: Tensor) -> Tensor:
        return torch.sqrt(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(g, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ExpBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.exp(X)
    backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X = torch.rand((4, 6), requires_grad=True)
    O = torch.exp(X).sum()
    backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative vs hessian
    X = torch.rand((4, 6), requires_grad=True)
    O = torch.exp(X).sum()
    backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.exp(x_ref).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
