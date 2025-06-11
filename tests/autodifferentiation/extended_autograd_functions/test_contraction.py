# Standard Libray dependencies
import itertools
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward
from thoad.typing.data import Shape, Indep


def test_AddmmXBackward0() -> None:
    order: int = 3

    # # Shape tests with different grad combinations
    # for reqx, reqm1, reqm2 in [
    #     (True, True, True), (True, True, False),
    #     (True, False, True), (False, True, True)
    # ]:
    #     X: Tensor = torch.rand((4, 8), requires_grad=reqx)
    #     M1: Tensor = torch.rand((4, 6), requires_grad=reqm1)
    #     M2: Tensor = torch.rand((6, 8), requires_grad=reqm2)
    #     O: Tensor = torch.addmm(X, M1, M2)
    #     op: Operator = backward(O, order=order, crossings=True)
    #     for i in range(order):
    #         if reqx:
    #             assert X.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (X.numel(),) * (i + 1)
    #         if reqm1:
    #             assert M1.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (M1.numel(),) * (i + 1)
    #         if reqm2:
    #             assert M2.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (M2.numel(),) * (i + 1)

    # # First derivative vs torch.autograd
    # for reqx, reqm1, reqm2 in [
    #     (True, True, True), (True, True, False),
    #     (True, False, True), (False, True, True)
    # ]:
    #     X = torch.rand((4, 8), requires_grad=reqx)
    #     M1 = torch.rand((4, 6), requires_grad=reqm1)
    #     M2 = torch.rand((6, 8), requires_grad=reqm2)
    #     O = torch.addmm(X, M1, M2).sum()
    #     op = backward(O, order=1)
    #     O.backward()
    #     if reqx:
    #         assert torch.allclose(
    #             X.hgrad[0].flatten(), X.grad.flatten()
    #         )
    #     if reqm1:
    #         assert torch.allclose(
    #             M1.hgrad[0].flatten(), M1.grad.flatten()
    #         )
    #     if reqm2:
    #         assert torch.allclose(
    #             M2.hgrad[0].flatten(), M2.grad.flatten()
    #         )

    # TMP: -----------------------------------------------------
    # Shape test
    X: Tensor = torch.rand((3, 5), requires_grad=True)
    M1: Tensor = torch.rand((3, 4), requires_grad=True)
    M2: Tensor = torch.rand((4, 5), requires_grad=True)
    O: Tensor = torch.addmm(X, M1, M2)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert M1.hgrad[i].shape == (O.numel(),) + (M1.numel(),) * (i + 1)
        assert M2.hgrad[i].shape == (O.numel(),) + (M2.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 5), requires_grad=True)
    M1: Tensor = torch.rand((3, 4), requires_grad=True)
    M2: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = torch.addmm(X, M1, M2).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(M1.hgrad[0].flatten(), M1.grad.flatten())
    assert torch.allclose(M2.hgrad[0].flatten(), M2.grad.flatten())
    # ---------------------------------------------------------

    # Second derivatives with cross terms (all grads enabled)
    X = torch.rand((4, 8), requires_grad=True)
    M1 = torch.rand((4, 6), requires_grad=True)
    M2 = torch.rand((6, 8), requires_grad=True)
    O = torch.addmm(X, M1, M2).sum()
    op = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor, m1_ref: Tensor, m2_ref: Tensor) -> Tensor:
        return torch.addmm(x_ref, m1_ref, m2_ref).sum()

    full_hessian: Tensor = torch.autograd.functional.hessian(f, (X, M1, M2))
    H: Tensor
    indeps: object
    shapes: object
    input_pairs: list[Tuple[int, Tensor]] = [(0, X), (1, M1), (2, M2)]
    for (i, I), (j, J) in itertools.product(input_pairs, repeat=2):
        H, indeps, shapes = op.fetch_hgrad([I, J], batch=False)
        assert torch.allclose(H.flatten(), full_hessian[i][j].flatten(), atol=1e-4)


def test_BmmXBackward0() -> None:
    order: int = 3

    # # Shape tests with grad combinations
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X: Tensor = torch.rand((2, 3, 4), requires_grad=reqx)
    #     Y: Tensor = torch.rand((2, 4, 5), requires_grad=reqy)
    #     O: Tensor = torch.bmm(X, Y)
    #     op: Operator = backward(O, order=order, crossings=True)
    #     for i in range(order):
    #         if reqx:
    #             assert X.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (X.numel(),) * (i + 1)
    #         if reqy:
    #             assert Y.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (Y.numel(),) * (i + 1)

    # # First derivative vs torch.autograd
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X = torch.rand((2, 3, 4), requires_grad=reqx)
    #     Y = torch.rand((2, 4, 5), requires_grad=reqy)
    #     O = torch.bmm(X, Y).sum()
    #     op = backward(O, order=1)
    #     O.backward()
    #     if reqx:
    #         assert torch.allclose(
    #             X.hgrad[0].flatten(), X.grad.flatten()
    #         )
    #     if reqy:
    #         assert torch.allclose(
    #             Y.hgrad[0].flatten(), Y.grad.flatten()
    #         )

    # TMP: -----------------------------------------------------
    # Shape test
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    Y: Tensor = torch.rand((2, 4, 5), requires_grad=True)
    O: Tensor = torch.bmm(X, Y)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    Y: Tensor = torch.rand((2, 4, 5), requires_grad=True)
    O_sum: Tensor = torch.bmm(X, Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())
    # ---------------------------------------------------------

    # Second derivatives with cross terms
    X = torch.rand((2, 3, 4), requires_grad=True)
    Y = torch.rand((2, 4, 5), requires_grad=True)
    O = torch.bmm(X, Y).sum()
    op = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor, y_ref: Tensor) -> Tensor:
        return torch.bmm(x_ref, y_ref).sum()

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


def test_DotXBackward0() -> None:
    order: int = 3

    # # Shape tests with grad combinations
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X: Tensor = torch.rand(10, requires_grad=reqx)
    #     Y: Tensor = torch.rand(10, requires_grad=reqy)
    #     O: Tensor = torch.dot(X, Y)
    #     op: Operator = backward(O, order=order, crossings=True)
    #     for i in range(order):
    #         if reqx:
    #             assert X.hgrad[i].shape == (
    #                 1,
    #             ) + (X.numel(),) * (i + 1)
    #         if reqy:
    #             assert Y.hgrad[i].shape == (
    #                 1,
    #             ) + (Y.numel(),) * (i + 1)

    # # First derivative
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X = torch.rand(10, requires_grad=reqx)
    #     Y = torch.rand(10, requires_grad=reqy)
    #     O = torch.dot(X, Y)
    #     op = backward(O, order=1)
    #     O.backward()
    #     if reqx:
    #         assert torch.allclose(
    #             X.hgrad[0].flatten(), X.grad.flatten()
    #         )
    #     if reqy:
    #         assert torch.allclose(
    #             Y.hgrad[0].flatten(), Y.grad.flatten()
    #         )

    # TMP: -----------------------------------------------------
    # Shape test
    X: Tensor = torch.rand((10,), requires_grad=True)
    Y: Tensor = torch.rand((10,), requires_grad=True)
    O: Tensor = torch.dot(X, Y)
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((10,), requires_grad=True)
    Y: Tensor = torch.rand((10,), requires_grad=True)
    O_sum: Tensor = torch.dot(X, Y)
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())
    # ---------------------------------------------------------

    # Second derivatives
    X = torch.rand(10, requires_grad=True)
    Y = torch.rand(10, requires_grad=True)
    O = torch.dot(X, Y)
    op = backward(O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.dot(a_ref, b_ref)

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


def test_MmXBackward0() -> None:
    order: int = 3

    # # Shape tests with grad combinations
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X: Tensor = torch.rand((6, 4), requires_grad=reqx)
    #     Y: Tensor = torch.rand((4, 5), requires_grad=reqy)
    #     O: Tensor = torch.mm(X, Y)
    #     op: Operator = backward(O, order=order, crossings=True)
    #     for i in range(order):
    #         if reqx:
    #             assert X.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (X.numel(),) * (i + 1)
    #         if reqy:
    #             assert Y.hgrad[i].shape == (
    #                 O.numel(),
    #             ) + (Y.numel(),) * (i + 1)

    # # First derivative
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X = torch.rand((6, 4), requires_grad=reqx)
    #     Y = torch.rand((4, 5), requires_grad=reqy)
    #     O = torch.mm(X, Y).sum()
    #     op = backward(O, order=1)
    #     O.backward()
    #     if reqx:
    #         assert torch.allclose(
    #             X.hgrad[0].flatten(), X.grad.flatten()
    #         )
    #     if reqy:
    #         assert torch.allclose(
    #             Y.hgrad[0].flatten(), Y.grad.flatten()
    #         )

    # TMP: -----------------------------------------------------
    # Shape test
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O: Tensor = X @ Y
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((3, 4), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = (X @ Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())
    # ---------------------------------------------------------

    # Second derivatives with cross terms
    X = torch.rand((6, 4), requires_grad=True)
    Y = torch.rand((4, 5), requires_grad=True)
    O = torch.mm(X, Y).sum()
    op = backward(O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.mm(a_ref, b_ref).sum()

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


def test_MvXBackward0() -> None:
    order: int = 3

    X: Tensor
    Y: Tensor
    O: Tensor
    op: Operator
    O_sum: Tensor

    # # 1) Shape tests con todas las combinaciones de requires_grad
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X: Tensor = torch.rand((6, 4), requires_grad=reqx)
    #     Y: Tensor = torch.rand((4,   ), requires_grad=reqy)
    #     O: Tensor = torch.mv(X, Y)
    #     op = backward(O, order=order, crossings=True)
    #     for i in range(order):
    #         if reqx:
    #             assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
    #         if reqy:
    #             assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # # 2) Primera derivada vs .grad de PyTorch
    # for reqx, reqy in [(True, True), (True, False), (False, True)]:
    #     X = torch.rand((6, 4), requires_grad=reqx)
    #     Y = torch.rand((4,   ), requires_grad=reqy)
    #     O = torch.mv(X, Y).sum()
    #     op = backward(O, order=1)
    #     O.backward()
    #     if reqx:
    #         assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    #     if reqy:
    #         assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # 3) Prueba puntual de shapes con grad=True
    X = torch.rand((3, 4), requires_grad=True)
    Y = torch.rand((4,), requires_grad=True)
    O = X.mv(Y)
    op = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # 4) Primera derivada puntual vs torch.grad
    X = torch.rand((3, 4), requires_grad=True)
    Y = torch.rand((4,), requires_grad=True)
    O_sum = X.mv(Y).sum()
    op = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # 5) Segunda derivada completa con términos cruzados
    X = torch.rand((6, 4), requires_grad=True)
    Y = torch.rand((4,), requires_grad=True)
    O = torch.mv(X, Y).sum()
    op = backward(O, order=2, crossings=True)

    def f(a_ref: Tensor, b_ref: Tensor) -> Tensor:
        return torch.mv(a_ref, b_ref).sum()

    # Hessiano completo via torch.autograd.functional.hessian
    full_hessian: Tuple[Tuple[Tensor]] = torch.autograd.functional.hessian(f, (X, Y))

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
