# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### EXPONENTIATION


def test_PowXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.pow(input=X, exponent=3)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for exponent in [1, 0, -1, 1.5, -1.5]:
        X = torch.rand(size=(4, 6), requires_grad=True, device=device)
        O = torch.pow(input=X, exponent=exponent)
        O = O.sum() ** 2
        backward(tensor=O, order=1)
        O.backward()
        assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.pow(input=X, exponent=3)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.pow(input=x_ref, exponent=3).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PowXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with different grad requirements
    X_shapes: list[Shape] = [(4, 6), (1, 6), (4, 1), (4, 6), (4, 6), (6,), (4, 6)]
    Y_shapes: list[Shape] = [(4, 6), (4, 6), (4, 6), (1, 6), (4, 1), (4, 6), (6,)]
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X = torch.rand(size=xs, requires_grad=reqx, device=device)
            Y = torch.rand(size=ys, requires_grad=reqy, device=device)
            O = torch.pow(input=X, exponent=Y)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                if reqx:
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
                if reqy:
                    assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative vs torch.autograd
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X = torch.rand(size=(3, 4), requires_grad=reqx, device=device)
            Y = torch.rand(size=(3, 4), requires_grad=reqy, device=device)
            O = torch.pow(input=X, exponent=Y)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            if reqx:
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
            if reqy:
                assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives including cross terms
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    Y = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.pow(input=X, exponent=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.pow(x_ref, y_ref).sum() ** 2

    full_hessian: Tuple[Tuple[Tensor, ...], ...] = torch.autograd.functional.hessian(
        f, (X, Y)
    )
    H00, _ = ctrl.fetch_hgrad([X, X], keep_batch=False)
    H01, _ = ctrl.fetch_hgrad([X, Y], keep_batch=False)
    H10, _ = ctrl.fetch_hgrad([Y, X], keep_batch=False)
    H11, _ = ctrl.fetch_hgrad([Y, Y], keep_batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)


def test_SqrtBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.sqrt(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.sqrt(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative vs hessian
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.sqrt(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def g(x_ref: Tensor) -> Tensor:
        return torch.sqrt(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(g, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ExpXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.exp(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.exp(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative vs hessian
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.exp(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.exp(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
