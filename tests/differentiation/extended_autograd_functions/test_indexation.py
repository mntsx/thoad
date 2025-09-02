# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### INDEXATION


def test_CloneXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.clone()
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.clone()
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = X.clone()
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.clone().sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SelectXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(4,), (4, 6), (2, 4, 6)]
    indexations: list[list[Tuple[Union[int, slice], ...]]] = [
        [(0,), (3,), (-1,), (-2,), (-4,)],
        [
            (2, slice(6)),
            (-1, slice(6)),
            (-2, slice(6)),
            (slice(4), 2),
            (slice(4), -1),
            (slice(4), -2),
            (0, 0),
            (-1, -1),
            (-2, -2),
        ],
        [(slice(2), 2, slice(6)), (slice(2), -1, slice(6)), (slice(2), -2, slice(6))],
    ]

    # Shape test
    for shape, shape_indexations in zip(shapes, indexations):
        for indexation in shape_indexations:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X[*indexation]
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_indexations in zip(shapes, indexations):
        for indexation in shape_indexations:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X[*indexation]
            O = O.sum() ** 2
            backward(tensor=O, order=1)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.select(input=X, dim=1, index=2)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.select(x_ref, 1, 2).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SliceXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(10, 10), (4, 4, 4)]
    indexations: list[list[Tuple[Union[int, slice], ...]]] = [
        [
            (slice(1, 9, 3), slice(1, 4, 4)),
            (slice(None, -1, 3), slice(None, -1, None)),
            (slice(-2, 10, 3), slice(-1, 10, None)),
            (slice(-2, None, 3), slice(-2, None, None)),
            (slice(None, 100, 4), slice(None, 100, None)),
            (slice(200, 100, 4), slice(200, 100, None)),
            (slice(200, 300, 2), slice(200, 300, None)),
            (slice(8, 2, 2), slice(8, 2, None)),
            (slice(None, None, None), slice(None, None, None)),
        ],
        [
            (slice(None, None, None), slice(None, None, None), slice(None, None, None)),
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

    # First derivative vs torch.autograd
    for shape, shape_indexations in zip(shapes, indexations):
        for indexation in shape_indexations:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X[*indexation]
            O = O * torch.rand_like(O, requires_grad=False, device=device)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4, 5), requires_grad=True, device=device)
    O = X[:].sum() ** 2
    backward(tensor=O, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref[:].sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_IndexSelectXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    IDX: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 5, 2)]
    indices: list[list[Tensor]] = [
        [
            torch.randint(low=0, high=5, size=(3,), device=device),
            torch.randint(low=0, high=5, size=(10,), device=device),
            torch.randint(low=0, high=5, size=(0,), device=device),
        ],
    ]

    # Shape test
    for shape, shape_indices in zip(shapes, indices):
        for index in shape_indices:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.index_select(input=X, dim=1, index=index)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    for shape, shape_indices in zip(shapes, indices):
        for index in shape_indices:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.index_select(input=X, dim=1, index=index)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X = torch.rand(size=(3, 5, 3), requires_grad=True, device=device)
    IDX: Tensor = torch.randint(low=0, high=5, size=(5,), device=device)
    O = torch.index_select(X, 1, IDX).sum() ** 2
    backward(tensor=O, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return torch.index_select(x_ref, 1, IDX).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
