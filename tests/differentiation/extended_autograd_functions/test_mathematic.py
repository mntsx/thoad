# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### MORE MATH


def test_AbsXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.abs(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.abs(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.abs(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.abs(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_LogXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log(input=(X + 0.1))
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.log(input=(x_ref + 0.1)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_Log2XBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log2(input=(X + 0.1))
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log2(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log2(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.log2(input=(x_ref + 0.1)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_Log10XBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log10(input=(X + 0.1))
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log10(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.log10(input=(X + 0.1))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.log10(input=(x_ref + 0.1)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MaxXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(4,), (4, 6), (2, 4, 6)]
    dims: list[list[Union[int, Tuple[int, ...]]]] = [
        [0, -1],
        [
            0,
            1,
            -1,
            -2,
        ],
        [0, 1, 2, -2],
    ]

    # Shape test
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.max(dim=dim, keepdim=keepdim)[0]
                backward(tensor=O, order=order, crossings=True)
                for i in range(order):
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.max(dim=dim, keepdim=keepdim)[0]
                O = O.sum() ** 2
                backward(tensor=O, order=order, crossings=True)
                O.backward()
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.max(dim=1)[0]
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.max(dim=1)[0].sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MaxXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = torch.max(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = torch.max(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3), requires_grad=True, device=device)
    O = torch.max(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.max(input=x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MaximumXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with different grad requirements
    X_shapes: list[Shape] = [(2, 3), (1, 3), (2, 1), (2, 3), (2, 3), (3,), (2, 3)]
    Y_shapes: list[Shape] = [(2, 3), (2, 3), (2, 3), (1, 3), (2, 1), (2, 3), (3,)]
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X = torch.rand(size=xs, requires_grad=reqx, device=device)
            Y = torch.rand(size=ys, requires_grad=reqy, device=device)
            O = torch.maximum(input=X, other=Y)
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
            O = torch.maximum(input=X, other=Y)
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
    O = torch.maximum(input=X, other=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.maximum(input=a_ref, other=b_ref).sum() ** 2

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


def test_MinXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(4,), (4, 6), (2, 4, 6)]
    dims: list[list[Union[int, Tuple[int, ...]]]] = [
        [0, -1],
        [
            0,
            1,
            -1,
            -2,
        ],
        [0, 1, 2, -2],
    ]

    # Shape test
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.min(dim=dim, keepdim=keepdim)[0]
                backward(tensor=O, order=order, crossings=True)
                for i in range(order):
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.min(dim=dim, keepdim=keepdim)[0]
                O = O.sum() ** 2
                backward(tensor=O, order=order, crossings=True)
                O.backward()
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.min(dim=1)[0]
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.min(dim=1)[0].sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MinXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = torch.min(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = torch.min(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3), requires_grad=True, device=device)
    O = torch.min(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.min(input=x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MinimumXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with different grad requirements
    X_shapes: list[Shape] = [(2, 3), (1, 3), (2, 1), (2, 3), (2, 3), (3,), (2, 3)]
    Y_shapes: list[Shape] = [(2, 3), (2, 3), (2, 3), (1, 3), (2, 1), (2, 3), (3,)]
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X = torch.rand(size=xs, requires_grad=reqx, device=device)
            Y = torch.rand(size=ys, requires_grad=reqy, device=device)
            O = torch.minimum(input=X, other=Y)
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
            O = torch.minimum(input=X, other=Y)
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
    O = torch.minimum(input=X, other=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.minimum(input=a_ref, other=b_ref).sum() ** 2

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


def test_MeanXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.mean()
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.mean()
    O = O.sum() ** 2
    backward(tensor=O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3), requires_grad=True, device=device)
    O = X.mean()
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.mean().sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MeanXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(4,), (4, 6), (2, 4, 6)]
    dims: list[list[Union[int, Tuple[int, ...]]]] = [
        [0, -1, (0,), (-1,)],
        [0, 1, -1, -2, (0, -1), (-1, 0), (1, -2), (-2, 1)],
        [0, 1, 2, -2],
    ]

    # Shape test
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.mean(dim=dim, keepdim=keepdim)
                backward(tensor=O, order=order, crossings=True)
                for i in range(order):
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_dims in zip(shapes, dims):
        for dim in shape_dims:
            for keepdim in (False, True):
                X = torch.randn(size=shape, requires_grad=True, device=device)
                O = X.mean(dim=dim, keepdim=keepdim)
                O = O.sum() ** 2
                backward(tensor=O, order=order, crossings=True)
                O.backward()
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative zero
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.mean(dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.mean(dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_NegXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = -X
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = -X
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = -X
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return (-x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SigmoidXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.sigmoid(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.sigmoid(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(3, 4), requires_grad=True, device=device)
    O = torch.sigmoid(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sigmoid(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_XlogyXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with different grad requirements
    X_shapes: list[Shape] = [(2, 3), (1, 3), (2, 1), (2, 3), (2, 3), (3,), (2, 3)]
    Y_shapes: list[Shape] = [(2, 3), (2, 3), (2, 3), (1, 3), (2, 1), (2, 3), (3,)]
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        for xs, ys in zip(X_shapes, Y_shapes):
            X = torch.rand(size=xs, requires_grad=reqx, device=device)
            Y = torch.rand(size=ys, requires_grad=reqy, device=device)
            O = torch.special.xlogy(input=(X + 0.1), other=(Y + 0.1))
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
            O = torch.special.xlogy(input=(X + 0.1), other=(Y + 0.1))
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            if reqx:
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
            if reqy:
                assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives only if both grads active
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    Y = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = torch.special.xlogy(input=(X + 0.1), other=(Y + 0.1))
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.special.xlogy(input=(a_ref + 0.1), other=(b_ref + 0.1)).sum() ** 2

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
