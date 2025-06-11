# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_DivXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X / Y
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O_sum: Tensor = (X / Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives including cross terms
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O_sum: Tensor = (X / Y).sum()
    op: Operator = backward(O_sum, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return (a_ref / b_ref).sum()

    full_hessian: Tensor = torch.autograd.functional.hessian(f, (X, Y))
    H: Tensor
    indeps: object
    shapes: object
    H00, indeps, shapes = op.fetch_hgrad([X, X], batch=False)
    H01, indeps, shapes = op.fetch_hgrad([X, Y], batch=False)
    H10, indeps, shapes = op.fetch_hgrad([Y, X], batch=False)
    H11, indeps, shapes = op.fetch_hgrad([Y, Y], batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)


def test_MulXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X * Y
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O_sum: Tensor = (X * Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives including cross terms
    # X: Tensor = torch.rand((4, 6), requires_grad=True)
    # Y: Tensor = torch.rand((4, 6), requires_grad=True)
    X: Tensor = torch.rand((2, 2), requires_grad=True)
    Y: Tensor = torch.rand((2, 2), requires_grad=True)
    O_sum: Tensor = (X * Y).sum()
    op: Operator = backward(O_sum, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return (a_ref * b_ref).sum()

    full_hessian: Tensor = torch.autograd.functional.hessian(f, (X, Y))
    H: Tensor
    indeps: object
    shapes: object
    H00, indeps, shapes = op.fetch_hgrad([X, X], batch=False)
    H01, indeps, shapes = op.fetch_hgrad([X, Y], batch=False)
    H10, indeps, shapes = op.fetch_hgrad([Y, X], batch=False)
    H11, indeps, shapes = op.fetch_hgrad([Y, Y], batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)


def test_ProdXBackward0() -> None:
    order: int = 2
    # Shape test (scalar output)
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.prod()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (1,) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.prod()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.prod()
    op: Operator = backward(O, order=2, crossings=True)

    def f(a_ref: Tensor) -> Tensor:
        return a_ref.prod()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_ProdXBackward1() -> None:
    order: int = 2
    # Shape test (vector output along dim)
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O: Tensor = X.prod(dim=1)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O_sum: Tensor = X.prod(dim=1).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    O_sum: Tensor = X.prod(dim=1).sum()
    op: Operator = backward(O_sum, order=2)

    def f(a_ref: Tensor) -> Tensor:
        return a_ref.prod(dim=1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
