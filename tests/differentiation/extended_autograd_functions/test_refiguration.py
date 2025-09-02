# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### FIGURATION
def test_ExpandXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(1,), (4,), (2, 1, 6)]
    expansions: list[list[Shape]] = [
        [(1,), (4,), (-1,), (1, 4), (3, 4), (3, -1), (2, 3, 4)],
        [(4,), (-1,), (1, 4), (3, 4), (3, -1), (2, 3, 4)],
        [(2, 1, 6), (2, -1, 6), (2, 4, 6), (2, 4, -1), (3, 2, 4, 6), (3, 2, 4, -1)],
    ]

    # Shape test
    for shape, shape_expansions in zip(shapes, expansions):
        for expansion in shape_expansions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X.expand(size=expansion)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_expansions in zip(shapes, expansions):
        for expansion in shape_expansions:
            for keepdim in (False, True):
                X = torch.rand(size=shape, requires_grad=True, device=device)
                O = X.expand(size=expansion)
                O = O.sum() ** 2
                backward(tensor=O, order=order, crossings=True)
                O.backward()
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivatives
    X = torch.rand(size=(2, 1), requires_grad=True, device=device)
    O = X.expand(size=(2, 3))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.expand(size=(2, 3)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PermuteXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.permute(dims=(1, 0, 2))
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.permute(dims=(1, 0, 2))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.permute(dims=(1, 0, 2))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.permute(dims=(1, 0, 2)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ViewXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    for source_shape, target_shape in [
        ((2, 3, 4), (6, 4)),
        ((6, 4), (2, 3, 4)),
        ((3, 4), (4, 3)),
        ((1, 3), (3,)),
        ((3,), (1, 3)),
    ]:
        X = torch.rand(size=source_shape, requires_grad=True, device=device)
        O = X.view(size=target_shape)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.view(size=(6, 4))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.view(size=(6, 4))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.view(size=(6, 4)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_RepeatXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    for reps in [(2, 3), (1, 1), (1, 2)]:
        X = torch.rand(size=(2, 2), requires_grad=True, device=device)
        O = X.repeat(repeats=reps)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(2, 2), requires_grad=True, device=device)
    O = X.repeat(repeats=(2, 3))
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 2), requires_grad=True, device=device)
    O = X.repeat(repeats=(2, 3))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.repeat(repeats=(2, 3)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test: remove all singleton dims
    X = torch.rand(size=(1, 3, 1, 4), requires_grad=True, device=device)
    O = X.squeeze()
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(1, 3, 1, 4), requires_grad=True, device=device)
    O = X.squeeze()
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(1, 3, 1, 4), requires_grad=True, device=device)
    O = X.squeeze()
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.squeeze().sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward1() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 3, 4), (2, 1, 4)]
    dimensions: list[list[Shape]] = [
        [0, 1, 2, -1, -2, -3],
        [0, 1, 2, -1, -2, -3],
    ]

    # Shape test
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.squeeze(input=X, dim=dim)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.squeeze(input=X, dim=dim)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 1, 4), requires_grad=True, device=device)
    O = torch.squeeze(input=X, dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.squeeze(dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SqueezeXBackward2() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 3, 4, 5), (2, 1, 4, 1)]
    dimensions: list[list[Shape]] = [
        [(1, 3), (3, 1), (-3, -1), (-1, -3)],
        [(1, 3), (3, 1), (-3, -1), (-1, -3)],
    ]

    # Shape test
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dims in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.squeeze(input=X, dim=dims)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dims in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.squeeze(input=X, dim=dims)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 3, 4, 5), requires_grad=True, device=device)
    O = torch.squeeze(input=X, dim=(1, 3))
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.squeeze(input=x_ref, dim=(0, 2)).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test: T() is transpose for 2D
    X = torch.rand(size=(5, 7), requires_grad=True, device=device)
    O = X.T
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 7), requires_grad=True, device=device)
    O = X.T
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 7), requires_grad=True, device=device)
    O = X.T
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.T.sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TransposeXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    dimensions: list[Tuple[int, int]] = [
        (0, 2),
        (2, 0),
        (1, 1),
        (1, -1),
        (-1, -2),
        (-1, -3),
        (0, -3),
    ]

    # Shape test: general transpose dims
    for dim0, dim1 in dimensions:
        X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
        O = X.transpose(dim0=dim0, dim1=dim1)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for dim0, dim1 in dimensions:
        X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
        O = X.transpose(dim0=0, dim1=2)
        O = O.sum() ** 2
        backward(tensor=O, order=order, crossings=True)
        O.backward()
        assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    O = X.transpose(dim0=0, dim1=2)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.transpose(dim0=0, dim1=2).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_UnsqueezeXBackward0() -> None:
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(4,), (4, 6)]
    indexations: list[list[Tuple[Union[None, int, slice], ...]]] = [
        [
            (None, -1),
            (-1, None),
            (None, slice(4)),
            (slice(4), None),
            (None, None, -1),
            (None, -1, None),
            (-1, None, None),
        ],
        [
            (None, slice(4), slice(6)),
            (slice(4), None, slice(6)),
            (slice(4), slice(6), None),
        ],
    ]

    # Shape test
    for shape, shape_indexations in zip(shapes, indexations):
        for indexation in shape_indexations:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X[*indexation]
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_indexations in zip(shapes, indexations):
        for indexation in shape_indexations:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X[*indexation]
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4), requires_grad=True, device=device)
    O = X.unsqueeze(dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.unsqueeze(1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
