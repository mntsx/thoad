# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### SOFT ACTIVATIONS


def test_SoftmaxXBackward0() -> None:
    ctrl: Controller
    order: int = 3
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 3, 4), (2, 1, 4)]
    dimensions: list[list[int]] = [
        [0, 1, 2, -1, -2, -3],
        [0, 1, 2, -1, -2, -3],
    ]

    # Shape test
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X.softmax(dim=dim)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = X.softmax(dim=dim)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.nn.functional.softmax(input=X, dim=1)
    O = (O**2).sum()
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return (torch.nn.functional.softmax(x_ref, dim=1) ** 2).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)

    # The 2nd order softmax test requires to do [(x.softmax(dim=n)**2).sum()] instead
    # of [(x.softmax(dim=n)**2).sum()]. The problem is that running
    # x.softmax(dim=n).sum()**2 always results in null gradients (all zero). This
    # happens because the resulting values of softmax always sum to 1 accros the
    # specified dimension


def test_SoftmaxXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 3, 4)]
    dimensions: list[list[int]] = [
        [1],
    ]

    # Shape test
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.softmax(input=X, dim=dim) ** 2
            ctrl = backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.softmax(input=X, dim=dim)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.softmax(input=X, dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.softmax(input=x_ref, dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_LogSoftmaxXBackward0() -> None:
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
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.nn.functional.log_softmax(input=X, dim=dim)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.nn.functional.log_softmax(input=X, dim=dim)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.nn.functional.log_softmax(input=X, dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.log_softmax(x_ref, dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SoftplusXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test elementwise
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.nn.functional.softplus(X - 0.5)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.nn.functional.softplus(X - 0.5)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(3, 4, 2), requires_grad=True, device=device)
    O = torch.nn.functional.softplus(X - 0.5)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x_ref - 0.5).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-3)
