# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward
from thoad.typing import Shape


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### LINEAR UNITS


def test_CeluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.celu(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.celu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.celu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.celu(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_EluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.elu(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.elu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.elu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.elu(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_GeluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.gelu(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.gelu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.gelu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.gelu(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_GluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Define test cases
    shapes: list[Shape] = [(2, 4, 6)]
    dimensions: list[list[int]] = [[0, 1, 2, -1, -2, -3]]

    # Shape test
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.nn.functional.glu(input=X, dim=dim)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.grad
    for shape, shape_dimensions in zip(shapes, dimensions):
        for dim in shape_dimensions:
            X = torch.rand(size=shape, requires_grad=True, device=device)
            O = torch.nn.functional.glu(input=X, dim=dim)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(2, 4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.glu(input=X, dim=1)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.glu(x_ref, dim=1).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_LeakyReluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.leaky_relu(input=X, negative_slope=0.01)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.leaky_relu(input=X, negative_slope=0.01)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.leaky_relu(input=X, negative_slope=0.01)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.leaky_relu(x_ref, negative_slope=0.01).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_PreluKernelXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    Y = torch.randn(size=(3,), requires_grad=True, device=device)
    O = torch.nn.functional.prelu(input=X, weight=Y)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
        assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First Derivative
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    Y = torch.randn(size=(3,), requires_grad=True, device=device)
    O = torch.nn.functional.prelu(input=X, weight=Y)
    O = O**2
    backward(tensor=O, order=order, crossings=True)

    def f(x_ref: Tensor, w_ref: Tensor) -> Tensor:
        result: Tensor = torch.nn.functional.prelu(input=x_ref, weight=w_ref)
        return result**2

    full_jacobian: Tensor = torch.autograd.functional.jacobian(f, (X, Y))
    X_grad: Tensor = full_jacobian[0]
    Y_grad: Tensor = full_jacobian[1]
    assert torch.allclose(X.hgrad[0].flatten(), X_grad.flatten(), atol=1e-4)
    assert torch.allclose(Y.hgrad[0].flatten(), Y_grad.flatten(), atol=1e-4)

    # Second derivative with cross terms
    X = torch.randn(size=(2, 3, 4), requires_grad=True, device=device)
    Y = torch.randn(size=(3,), requires_grad=True, device=device)
    O = torch.nn.functional.prelu(input=X, weight=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor, w_ref: Tensor) -> Tensor:
        return torch.nn.functional.prelu(input=x_ref, weight=w_ref).sum() ** 2

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


def test_ReluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(5, 4), requires_grad=True, device=device)
    O = torch.relu(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(5, 4), requires_grad=True, device=device)
    O = torch.relu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(5, 4), requires_grad=True, device=device)
    O = torch.relu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.relu(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_RreluWithNoiseXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test (deterministic mode)
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.rrelu(input=X, lower=0.1, upper=0.3, training=False)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.rrelu(input=X, lower=0.1, upper=0.3, training=False)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.rrelu(input=X, lower=0.1, upper=0.3, training=False)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return (
            torch.nn.functional.rrelu(x_ref, lower=0.1, upper=0.3, training=False).sum()
            ** 2
        )

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SiluXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.silu(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.silu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.randn(size=(4, 6), requires_grad=True, device=device)
    O = torch.nn.functional.silu(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.nn.functional.silu(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
