# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### CONDITION


def test_WhereXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        cond: Tensor = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
        X = torch.rand(size=(3, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(3, 4), requires_grad=reqy, device=device)
        O = torch.where(condition=cond, input=X, other=Y)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative vs torch.autograd
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        cond: Tensor = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
        X = torch.rand(size=(3, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(3, 4), requires_grad=reqy, device=device)
        O = torch.where(condition=cond, input=X, other=Y)
        O = O.sum() ** 2
        backward(tensor=O, order=order, crossings=True)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives including cross terms
    cond: Tensor = torch.randint(0, 2, (4, 6), dtype=torch.bool, device=device)
    X = torch.rand(size=(4, 6), requires_grad=True, device=device)
    Y = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.where(condition=cond, input=X, other=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    # Reference Hessians
    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.where(condition=cond, input=x_ref, other=y_ref).sum() ** 2

    full_hessian: Tuple[Tuple[Tensor, ...], ...] = torch.autograd.functional.hessian(
        f, (X, Y)
    )
    # Fetch computed Hessians
    H00, _ = ctrl.fetch_hgrad([X, X], keep_batch=False)
    H01, _ = ctrl.fetch_hgrad([X, Y], keep_batch=False)
    H10, _ = ctrl.fetch_hgrad([Y, X], keep_batch=False)
    H11, _ = ctrl.fetch_hgrad([Y, Y], keep_batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)
