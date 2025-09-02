# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad import Controller, backward


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### MATRIX MULTIPLICATION


def test_AddmmXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    M1: Tensor
    M2: Tensor
    O: Tensor

    # Shape tests with different grad combinations
    for reqx, reqm1, reqm2 in [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ]:
        for beta in [1.0, 1.5]:
            X = torch.rand(size=(4, 8), requires_grad=reqx, device=device)
            M1 = torch.rand(size=(4, 6), requires_grad=reqm1, device=device)
            M2 = torch.rand(size=(6, 8), requires_grad=reqm2, device=device)
            O = torch.addmm(input=X, mat1=M1, mat2=M2, beta=beta)
            backward(tensor=O, order=order, crossings=True)
            for i in range(order):
                if reqx:
                    assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
                if reqm1:
                    assert M1.hgrad[i].shape == (O.numel(), *(M1.shape * (i + 1)))
                if reqm2:
                    assert M2.hgrad[i].shape == (O.numel(), *(M2.shape * (i + 1)))

    # First derivative vs torch.autograd
    for reqx, reqm1, reqm2 in [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ]:
        for beta in [1.0, 1.5]:
            X = torch.rand(size=(4, 8), requires_grad=reqx, device=device)
            M1 = torch.rand(size=(4, 6), requires_grad=reqm1, device=device)
            M2 = torch.rand(size=(6, 8), requires_grad=reqm2, device=device)
            O = torch.addmm(input=X, mat1=M1, mat2=M2, beta=beta)
            O = O.sum() ** 2
            backward(tensor=O, order=order, crossings=True)
            O.backward()
            if reqx:
                assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
            if reqm1:
                assert torch.allclose(M1.hgrad[0].flatten(), M1.grad.flatten())
            if reqm2:
                assert torch.allclose(M2.hgrad[0].flatten(), M2.grad.flatten())

    # Second derivatives with cross terms (all grads enabled)
    X = torch.rand(size=(4, 8), requires_grad=True, device=device)
    M1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    M2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O = torch.addmm(input=X, mat1=M1, mat2=M2)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor, m1_ref: Tensor, m2_ref: Tensor) -> Tensor:
        return torch.addmm(x_ref, m1_ref, m2_ref).sum() ** 2

    full_hessian: Tuple[Tuple[Tensor, ...], ...] = torch.autograd.functional.hessian(
        f, (X, M1, M2)
    )
    H00, _ = ctrl.fetch_hgrad([X, X], keep_batch=False)
    H01, _ = ctrl.fetch_hgrad([X, M1], keep_batch=False)
    H02, _ = ctrl.fetch_hgrad([X, M2], keep_batch=False)
    H10, _ = ctrl.fetch_hgrad([M1, X], keep_batch=False)
    H11, _ = ctrl.fetch_hgrad([M1, M1], keep_batch=False)
    H12, _ = ctrl.fetch_hgrad([M1, M2], keep_batch=False)
    H20, _ = ctrl.fetch_hgrad([M2, X], keep_batch=False)
    H21, _ = ctrl.fetch_hgrad([M2, M1], keep_batch=False)
    H22, _ = ctrl.fetch_hgrad([M2, M2], keep_batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H02.flatten(), full_hessian[0][2].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)
    assert torch.allclose(H12.flatten(), full_hessian[1][2].flatten(), atol=1e-4)
    assert torch.allclose(H20.flatten(), full_hessian[2][0].flatten(), atol=1e-4)
    assert torch.allclose(H21.flatten(), full_hessian[2][1].flatten(), atol=1e-4)
    assert torch.allclose(H22.flatten(), full_hessian[2][2].flatten(), atol=1e-4)


def test_BmmXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(2, 3, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(2, 4, 5), requires_grad=reqy, device=device)
        O = torch.bmm(input=X, mat2=Y)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative vs torch.autograd
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(2, 3, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(2, 4, 5), requires_grad=reqy, device=device)
        O = torch.bmm(input=X, mat2=Y)
        O = O.sum() ** 2
        backward(tensor=O, order=order, crossings=True)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives with cross terms
    X = torch.rand(size=(2, 3, 4), requires_grad=True, device=device)
    Y = torch.rand(size=(2, 4, 5), requires_grad=True, device=device)
    O = torch.bmm(input=X, mat2=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.bmm(x_ref, y_ref).sum() ** 2

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


def test_DotXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(10, requires_grad=reqx, device=device)
        Y = torch.rand(10, requires_grad=reqy, device=device)
        O = torch.dot(input=X, tensor=Y)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(10, requires_grad=reqx, device=device)
        Y = torch.rand(10, requires_grad=reqy, device=device)
        O = torch.dot(input=X, tensor=Y)
        O = O.sum() ** 2
        backward(tensor=O, order=order, crossings=True)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives
    X = torch.rand(10, requires_grad=True, device=device)
    Y = torch.rand(10, requires_grad=True, device=device)
    O = torch.dot(input=X, tensor=Y) ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.dot(a_ref, b_ref) ** 2

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


def test_MmXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(6, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(4, 5), requires_grad=reqy, device=device)
        O = torch.mm(input=X, mat2=Y)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(6, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(4, 5), requires_grad=reqy, device=device)
        O = torch.mm(input=X, mat2=Y)
        O = O.sum() ** 2
        backward(tensor=O, order=1)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten(), atol=1e-4)

    # Second derivatives with cross terms
    X = torch.rand(size=(6, 4), requires_grad=True, device=device)
    Y = torch.rand(size=(4, 5), requires_grad=True, device=device)
    O = torch.mm(input=X, mat2=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.mm(a_ref, b_ref).sum() ** 2

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


def test_MvXBackward0() -> None:
    ctrl: Controller
    order: int = 2
    X: Tensor
    Y: Tensor
    O: Tensor

    # Shape tests with grad combinations
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(6, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(4,), requires_grad=reqy, device=device)
        O = torch.mv(input=X, vec=Y)
        backward(tensor=O, order=order, crossings=True)
        for i in range(order):
            if reqx:
                assert X.hgrad[i].shape == (O.numel(), *(X.shape * (i + 1)))
            if reqy:
                assert Y.hgrad[i].shape == (O.numel(), *(Y.shape * (i + 1)))

    # First derivative vs torch.autograd
    for reqx, reqy in [(True, True), (True, False), (False, True)]:
        X = torch.rand(size=(6, 4), requires_grad=reqx, device=device)
        Y = torch.rand(size=(4,), requires_grad=reqy, device=device)
        O = torch.mv(input=X, vec=Y)
        O = O.sum() ** 2
        backward(tensor=O, order=1)
        O.backward()
        if reqx:
            assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten(), atol=1e-4)
        if reqy:
            assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivatives including cross terms
    # X = torch.rand(size=(6, 4), requires_grad=True, device=device)
    # Y = torch.rand(size=(4,), requires_grad=True, device=device)
    X = torch.rand(size=(3, 2), requires_grad=True, device=device)
    Y = torch.rand(size=(2,), requires_grad=True, device=device)
    O = torch.mv(input=X, vec=Y)
    O = O.sum() ** 2
    ctrl = backward(tensor=O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.mv(input=a_ref, vec=b_ref).sum() ** 2

    full_hessian = torch.autograd.functional.hessian(f, (X, Y))
    H00, _ = ctrl.fetch_hgrad([X, X], keep_batch=False)
    H01, _ = ctrl.fetch_hgrad([X, Y], keep_batch=False)
    H10, _ = ctrl.fetch_hgrad([Y, X], keep_batch=False)
    H11, _ = ctrl.fetch_hgrad([Y, Y], keep_batch=False)
    assert torch.allclose(H00.flatten(), full_hessian[0][0].flatten(), atol=1e-4)
    assert torch.allclose(H01.flatten(), full_hessian[0][1].flatten(), atol=1e-4)
    assert torch.allclose(H10.flatten(), full_hessian[1][0].flatten(), atol=1e-4)
    assert torch.allclose(H11.flatten(), full_hessian[1][1].flatten(), atol=1e-4)
