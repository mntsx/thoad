# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_WhereXBackward0() -> None:
    order: int = 3
    # Shape test
    cond: Tensor = torch.randint(0, 2, (4, 6), dtype=torch.bool)
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.where(cond, X, Y)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    cond: Tensor = torch.randint(0, 2, (4, 6), dtype=torch.bool)
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.where(cond, X, Y).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives including cross terms
    cond: Tensor = torch.randint(0, 2, (4, 6), dtype=torch.bool)
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    Y: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.where(cond, X, Y).sum()
    op: Operator = backward(O, order=2, crossings=True)

    # Reference Hessians
    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.where(cond, x_ref, y_ref).sum()

    full_hessian: Tensor = torch.autograd.functional.hessian(f, (X, Y))

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
