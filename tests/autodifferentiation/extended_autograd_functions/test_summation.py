# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import Operator, backward


def test_AddXBackward0() -> None:
    order: int = 2
    # Shape test: elementwise add
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O: Tensor = X + Y
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = (X + Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # First derivative vs torch.autograd (changed value of alpha)
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = torch.add(input=X, other=Y, alpha=0.5).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivative: no cross term for addition (Hessian zero)
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = (X + Y).sum()
    op: Operator = backward(O_sum, order=2, crossings=True)
    H: Tensor
    _, _, _ = op.fetch_hgrad([X, Y], batch=False)
    H, _, _ = op.fetch_hgrad([X, Y], batch=False)
    H = H.squeeze(0)
    # Pure blocks zero
    assert torch.allclose(
        H[: X.numel(), : X.numel()], torch.zeros_like(H[: X.numel(), : X.numel()])
    )
    assert torch.allclose(
        H[X.numel() :, X.numel() :], torch.zeros_like(H[X.numel() :, X.numel() :])
    )
    # Cross blocks zero
    assert torch.allclose(
        H[: X.numel(), X.numel() :], torch.zeros_like(H[: X.numel(), X.numel() :])
    )
    assert torch.allclose(
        H[X.numel() :, : X.numel()], torch.zeros_like(H[X.numel() :, : X.numel()])
    )


def test_SubXBackward0() -> None:
    order: int = 2
    # Shape test: elementwise subtract
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O: Tensor = X - Y
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)
        assert Y.hgrad[i].shape == (O.numel(),) + (Y.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = (X - Y).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # First derivative vs torch.autograd (changed value of alpha)
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = torch.add(input=X, other=Y, alpha=0.5).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())
    assert torch.allclose(Y.hgrad[0].flatten(), Y.grad.flatten())

    # Second derivative: Hessian zero
    X: Tensor = torch.rand((4, 5), requires_grad=True)
    Y: Tensor = torch.rand((4, 5), requires_grad=True)
    O_sum: Tensor = (X - Y).sum()
    op: Operator = backward(O_sum, order=2, crossings=True)
    H: Tensor
    _, _, _ = op.fetch_hgrad([X, Y], batch=False)
    H, _, _ = op.fetch_hgrad([X, Y], batch=False)
    H = H.squeeze(0)
    assert torch.allclose(H, torch.zeros_like(H))


def test_SumXBackward0() -> None:
    order: int = 2
    # Shape test: sum over all elements
    X: Tensor = torch.rand((3, 4, 5), requires_grad=True)
    O: Tensor = X.sum()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (1,) + (X.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((3, 4, 5), requires_grad=True)
    O: Tensor = X.sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = X.sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SumXBackward1() -> None:
    order: int = 2
    # Shape test: sum over dim=(1, 3), keepdim=False
    X: Tensor = torch.rand((2, 3, 4, 5, 6), requires_grad=True)
    O: Tensor = X.sum(dim=(1, 3), keepdim=False).sum()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # Shape test: sum over dim=(1, 3), keepdim=True
    X: Tensor = torch.rand((2, 3, 4, 5, 6), requires_grad=True)
    O: Tensor = X.sum(dim=(1, 3), keepdim=True).sum()
    op: Operator = backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    X: Tensor = torch.rand((2, 3, 4, 5, 6), requires_grad=True)
    O: Tensor = X.sum(dim=(1, 3), keepdim=True).sum()
    op: Operator = backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # First derivative vs torch.autograd (keepdim = True)
    X: Tensor = torch.rand((2, 3, 4), requires_grad=True)
    O_sum: Tensor = X.sum(dim=1, keepdim=True).sum()
    op: Operator = backward(O_sum, order=1)
    O_sum.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((5, 4), requires_grad=True)
    O: Tensor = X.sum(dim=1).sum()
    op: Operator = backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.sum(dim=1).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)
