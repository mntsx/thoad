# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### TRIGONOMETRY


def test_SinXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sin(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative vs torch.autograd
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sin(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sin(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sin(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_CosXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cos(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cos(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cos(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.cos(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TanXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tan(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tan(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tan(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.tan(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SinhXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sinh(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sinh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.sinh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.sinh(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_CoshXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cosh(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cosh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.cosh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.cosh(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_TanhXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    O: Tensor

    # Shape test
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tanh(input=X)
    backward(tensor=O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))

    # First derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tanh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=order, crossings=True)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)

    # Second derivative
    X = torch.rand(size=(5, 4), requires_grad=True, device=device)
    O = torch.tanh(input=X)
    O = O.sum() ** 2
    backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.tanh(x_ref).sum() ** 2

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
