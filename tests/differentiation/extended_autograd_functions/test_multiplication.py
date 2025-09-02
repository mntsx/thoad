# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### MULTIPLICATION


def test_DivXBackward0() -> None:
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
            O = torch.div(input=(X + 0.1), other=(Y + 0.1))
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
            O = torch.div(input=(X + 0.1), other=(Y + 0.1))
            O = O.sum() ** 2
            backward(tensor=O, order=1)
            O.backward()
            if reqx:
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
            if reqy:
                assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives including cross terms
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    Y = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.div(input=X, other=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return (a_ref / b_ref).sum() ** 2

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


def test_MulXBackward0() -> None:
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
            O = torch.mul(input=X, other=Y)
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
            O = torch.mul(input=X, other=Y)
            O = O.sum() ** 2
            backward(tensor=O, order=1)
            O.backward()
            if reqx:
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
            if reqy:
                assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives including cross terms
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    Y = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X * Y
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return (a_ref * b_ref).sum() ** 2

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


def test_ProdXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test (scalar output)
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.prod()
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.prod() ** 2
    backward(tensor=O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.prod() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor) -> Tensor:
        return a_ref.prod() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ProdXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 3, 4), (2, 1, 4)]
    dimensions: list[list[int]] = [
        [0, 1, 2, -1, -2, -3],
        [0, 1, 2, -1, -2, -3],
    ]

    # Shape test
    for shape, shape_dims in zip(shapes, dimensions):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.rand(size=shape, requires_grad=True, device=device)
                O = X.prod(dim=dim, keepdim=keepdim)
                backward(tensor=O, order=order, crossings=True)
                for i in range(order):
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_dims in zip(shapes, dimensions):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.rand(size=shape, requires_grad=True, device=device)
                O = X.prod(dim=dim, keepdim=keepdim)
                O = O.sum() ** 2
                backward(tensor=O, order=order, crossings=True)
                O.backward()
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = X.prod(dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2)

    def f(a_ref: Tensor) -> Tensor:
        return a_ref.prod(dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MulXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor
    aux: Tensor
    ctrl: Controller

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    Y = torch.randn(size=(6, 4), requires_grad=True, device=device)
    aux = F.scaled_dot_product_attention(query=X, key=Y.T, value=Y.T)
    O = aux.grad_fn.next_functions[0][0].next_functions[0][0]._saved_self
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    Y = torch.randn(size=(6, 4), requires_grad=True, device=device)
    aux = F.scaled_dot_product_attention(query=X, key=Y.T, value=Y.T)
    O = aux.grad_fn.next_functions[0][0].next_functions[0][0]._saved_self
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    # X = torch.randn(size=(5, 4), requires_grad=True, device=device)
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    Y = torch.randn(size=(6, 4), requires_grad=True, device=device)
    aux = F.scaled_dot_product_attention(query=X, key=Y.T, value=Y.T)
    O = aux.grad_fn.next_functions[0][0].next_functions[0][0]._saved_self
    O = O.sum()
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        aux: Tensor
        aux = F.scaled_dot_product_attention(query=x_ref, key=y_ref.T, value=y_ref.T)
        O: Tensor = aux.grad_fn.next_functions[0][0].next_functions[0][0]._saved_self
        return O.sum()

    full_hessian: Tuple[Tuple[Tensor, ...], ...] = torch.autograd.functional.hessian(
        f, (X, Y)
    )
    H00: Tensor
    H00, _ = ctrl.fetch_hgrad([X, X], keep_batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
