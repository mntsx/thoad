# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.user.interface import backward


def test_CloneXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.clone()
    backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.clone().sum()
    backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = X.clone().sum()
    backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.clone().sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_SelectXBackward0() -> None:
    order: int = 3
    # Shape test
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.select(X, 1, 2)
    backward(O, order=order, crossings=True)
    for i in range(order):
        assert X.hgrad[i].shape == (O.numel(),) + (X.numel(),) * (i + 1)

    # First derivative vs torch.grad
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.select(X, 1, 2).sum()
    backward(O, order=1)
    O.backward()
    assert torch.allclose(X.hgrad[0].flatten(), X.grad.flatten())

    # Second derivative
    X: Tensor = torch.rand((4, 6), requires_grad=True)
    O: Tensor = torch.select(X, 1, 2).sum()
    backward(O, order=2, crossings=True)

    def f(x_ref: Tensor) -> Tensor:
        return torch.select(x_ref, 1, 2).sum()

    H: Tensor = torch.autograd.functional.hessian(f, X)
    assert torch.allclose(X.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_IndexXBackward0() -> None:
    order: int = 3
    # Shape test with boolean mask
    TA: Tensor = torch.rand((4, 6), requires_grad=True)
    mask: Tensor = TA > 0
    aux: Tensor = TA[mask]
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative
    TA = torch.rand((4, 6), requires_grad=True)
    mask = TA > 0
    aux = TA[mask].sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((4, 6), requires_grad=True)
    mask = TA > 0
    aux = TA[mask].sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref[x_ref > 0].sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)

    # Additional select tests
    T0: Tensor = torch.rand((4, 4), requires_grad=True)
    T1: Tensor = torch.rand((4, 4), requires_grad=True)
    O0: Tensor = T0[0, 0]
    O1: Tensor = T1[-1, -2]
    backward(O0, order=2)
    O0.backward()
    backward(O1, order=2)
    O1.backward()
    assert T0.hgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.hgrad[0].shape == (O1.numel(), T1.numel())
    assert torch.allclose(T0.hgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T1.hgrad[0].flatten(), T1.grad.flatten())


def test_SliceXBackward0() -> None:
    order: int = 3
    # Shape test
    TA: Tensor = torch.rand((3, 4, 5), requires_grad=True)
    aux: Tensor = TA[:]
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative vs torch.autograd
    TA = torch.rand((3, 4, 5), requires_grad=True)
    aux = TA[:].sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((3, 4, 5), requires_grad=True)
    aux = TA[:].sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref[:].sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)

    # Additional slice checks
    T0: Tensor = torch.rand((10, 10), requires_grad=True)
    O0: Tensor = T0[1:9:3, 2:4:2]
    O0 = (O0 * torch.rand_like(O0, requires_grad=True)).sum()
    backward(O0, order=2)
    O0.backward()
    assert T0.hgrad[0].shape == (O0.numel(), T0.numel())
    assert torch.allclose(T0.hgrad[0].flatten(), T0.grad.flatten())

    T1: Tensor = torch.rand((10, 10), requires_grad=True)
    O1: Tensor = T1[:-1:3, :-1]
    O1 = (O1 * torch.rand_like(O1, requires_grad=True)).sum()
    backward(O1, order=2)
    O1.backward()
    assert T1.hgrad[0].shape == (O1.numel(), T1.numel())
    assert torch.allclose(T1.hgrad[0].flatten(), T1.grad.flatten())

    T2: Tensor = torch.rand((10, 10), requires_grad=True)
    O2: Tensor = T2[-2:10:3, -2:10]
    O2 = (O2 * torch.rand_like(O2, requires_grad=True)).sum()
    backward(O2, order=2)
    O2.backward()
    assert T2.hgrad[0].shape == (O2.numel(), T2.numel())
    assert torch.allclose(T2.hgrad[0].flatten(), T2.grad.flatten())

    T3: Tensor = torch.rand((10, 10), requires_grad=True)
    O3: Tensor = T3[-2::3, -2:]
    O3 = (O3 * torch.rand_like(O3, requires_grad=True)).sum()
    backward(O3, order=2)
    O3.backward()
    assert T3.hgrad[0].shape == (O3.numel(), T3.numel())
    assert torch.allclose(T3.hgrad[0].flatten(), T3.grad.flatten())

    T4: Tensor = torch.rand((10, 10), requires_grad=True)
    O4: Tensor = T4[:100:4, :100]
    O4 = (O4 * torch.rand_like(O4, requires_grad=True)).sum()
    backward(O4, order=2)
    O4.backward()
    assert T4.hgrad[0].shape == (O4.numel(), T4.numel())
    assert torch.allclose(T4.hgrad[0].flatten(), T4.grad.flatten())

    T5: Tensor = torch.rand((10, 10), requires_grad=True)
    O5: Tensor = T5[200:100:4, 200:100]
    O5 = (O5 * torch.rand_like(O5, requires_grad=True)).sum()
    backward(O5, order=2)
    O5.backward()
    assert T5.hgrad[0].shape == (O5.numel(), T5.numel())
    assert torch.allclose(T5.hgrad[0].flatten(), T5.grad.flatten())

    T6: Tensor = torch.rand((10, 10), requires_grad=True)
    O6: Tensor = T6[:, :]
    O6 = (O6 * torch.rand_like(O6, requires_grad=True)).sum()
    backward(O6, order=2)
    O6.backward()
    assert T6.hgrad[0].shape == (O6.numel(), T6.numel())
    assert torch.allclose(T6.hgrad[0].flatten(), T6.grad.flatten())

    T7: Tensor = torch.rand((10, 10, 10), requires_grad=True)
    O7: Tensor = T7[:, :, :]
    O7 = (O7 * torch.rand_like(O7, requires_grad=True)).sum()
    backward(O7, order=2)
    O7.backward()
    assert T7.hgrad[0].shape == (O7.numel(), T7.numel())
    assert torch.allclose(T7.hgrad[0].flatten(), T7.grad.flatten())


def test_IndexSelectXBackward0() -> None:
    order: int = 3
    # Shape test
    TA: Tensor = torch.rand((5, 3), requires_grad=True)
    IDX: Tensor = torch.tensor([0, 2, 4], dtype=torch.long)
    aux: Tensor = torch.index_select(TA, dim=0, index=IDX)
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative
    TA = torch.rand((5, 3), requires_grad=True)
    aux = torch.index_select(TA, 0, IDX).sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((5, 3), requires_grad=True)
    aux = torch.index_select(TA, 0, IDX).sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return torch.index_select(x_ref, 0, IDX).sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_IndexPutXBackward0() -> None:
    order: int = 3
    # Shape test
    TA: Tensor = torch.rand((5, 3), requires_grad=True)
    IDX: Tensor = torch.tensor([1, 3], dtype=torch.long)
    values: Tensor = torch.rand((2, 3))
    aux: Tensor = TA.index_put((IDX,), values)
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative
    TA = torch.rand((5, 3), requires_grad=True)
    aux = TA.index_put((IDX,), values).sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((5, 3), requires_grad=True)
    aux = TA.index_put((IDX,), values).sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.index_put((IDX,), values).sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)


def test_MaskedSelectXBackward0() -> None:
    order: int = 3
    # Shape test
    TA: Tensor = torch.rand((4, 5), requires_grad=True)
    mask: Tensor = TA >= 0.0
    aux: Tensor = torch.masked_select(TA, mask)
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative
    TA = torch.rand((4, 5), requires_grad=True)
    aux = torch.masked_select(TA, TA >= 0.0).sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((4, 5), requires_grad=True)
    aux = torch.masked_select(TA, TA >= 0.0).sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return torch.masked_select(x_ref, x_ref >= 0.0).sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)

    # Test mask all False
    TA = torch.rand((5, 5), requires_grad=True)
    aux = torch.masked_select(TA, TA > 1.0).sum()
    backward(aux, order=2)
    aux.backward()
    assert TA.hgrad[0].shape == (aux.numel(), TA.numel())
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())


def test_MaskedScatterXBackward0() -> None:
    order: int = 3
    # Shape test
    TA: Tensor = torch.rand((4, 5), requires_grad=True)
    mask: Tensor = TA < 0.5
    source: Tensor = torch.rand((mask.sum().item(),))
    aux: Tensor = TA.masked_scatter(mask, source)
    backward(aux, order=order, crossings=True)
    for i in range(order):
        assert TA.hgrad[i].shape == (aux.numel(),) + (TA.numel(),) * (i + 1)

    # First derivative
    TA = torch.rand((4, 5), requires_grad=True)
    mask = TA < 0.5
    source = torch.rand((mask.sum().item(),))
    aux = TA.masked_scatter(mask, source).sum()
    backward(aux, order=1)
    aux.backward()
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())

    # Second derivative
    TA = torch.rand((4, 5), requires_grad=True)
    mask = TA < 0.5
    source = torch.rand((mask.sum().item(),))
    aux = TA.masked_scatter(mask, source).sum()
    backward(aux, order=2)

    def f(x_ref: Tensor) -> Tensor:
        return x_ref.masked_scatter(x_ref < 0.5, source).sum()

    H: Tensor = torch.autograd.functional.hessian(f, TA)
    assert torch.allclose(TA.hgrad[1].flatten(), H.flatten(), atol=1e-4)

    # Test empty source
    TA = torch.rand((4, 5), requires_grad=True)
    mask = TA < -1.0
    source = torch.rand((0,))
    aux = TA.masked_scatter(mask, source).sum()
    backward(aux, order=2)
    aux.backward()
    assert TA.hgrad[0].shape == (aux.numel(), TA.numel())
    assert torch.allclose(TA.hgrad[0].flatten(), TA.grad.flatten())


def test_GatherXBackward0() -> None:
    sz: int = 3
    dim: int = 1

    shape: list[int] = [2, 4]
    shape.insert(dim, sz)
    T0: Tensor = torch.rand(size=tuple(shape), requires_grad=True)

    # Define a scalar function that builds random indices and gathers
    def f(x_ref: Tensor) -> Tensor:
        IDX: Tensor = torch.randint(low=0, high=x_ref.shape[dim], size=x_ref.shape)
        tmp: Tensor = torch.gather(input=x_ref, dim=dim, index=IDX)
        return tmp.sum()

    O0: Tensor = f(T0)
    O0 = O0.sum()

    backward(O0, order=2)
    O0.backward()
    T0_hessian: Tensor = torch.autograd.functional.hessian(f, T0)

    assert T0.hgrad[0].shape == (O0.numel(), T0.numel())
    assert torch.allclose(T0.hgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T0.hgrad[1].flatten(), T0_hessian.flatten())


def test_ScatterXBackward0() -> None:
    sz: int = 3
    dim: int = 1

    shape_list: list[int] = [2, 4]
    shape_list.insert(dim, sz)
    shape: Tuple[int, ...] = tuple(shape_list)
    T0: Tensor = torch.rand(size=shape, requires_grad=True)
    T1: Tensor = torch.zeros_like(T0, requires_grad=True)
    IDX: Tensor = torch.randint(low=0, high=T0.shape[dim], size=T0.shape)

    def compute_scatter_hessians(
        input_ref: Tensor, index: Tensor, source_ref: Tensor
    ) -> Tuple[Tensor, Tensor]:
        def g(x: Tensor, s: Tensor) -> Tensor:
            return torch.scatter(input=x, dim=dim, index=index, src=s).sum()

        full_hess = torch.autograd.functional.hessian(
            func=g, inputs=(input_ref, source_ref)
        )
        return full_hess[0][0], full_hess[1][1]

    T0_hessian, T1_hessian = compute_scatter_hessians(
        input_ref=T0, index=IDX, source_ref=T1
    )

    O0: Tensor = torch.scatter(input=T1, dim=dim, index=IDX, src=T0).sum()

    backward(O0, order=2)
    O0.backward()

    # Check first derivatives
    assert T0.hgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.hgrad[0].shape == (O0.numel(), T1.numel())
    assert torch.allclose(T0.hgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T1.hgrad[0].flatten(), T1.grad.flatten())
    # Check second derivatives
    assert torch.allclose(T0.hgrad[1].flatten(), T0_hessian.flatten())
    assert torch.allclose(T1.hgrad[1].flatten(), T1_hessian.flatten())


def test_TakeXBackward0() -> None:
    # Define shape and compute indices
    shape: Tuple[int, int] = (10, 10)
    numel: int = shape[0] * shape[1]
    idx_size: int = torch.randint(low=1, high=2 * numel, size=(1,)).item()

    T0: Tensor = torch.rand(size=shape, requires_grad=True)
    T1: Tensor = torch.rand(size=shape, requires_grad=True)
    T2: Tensor = torch.rand(size=shape, requires_grad=True)

    IDX0: Tensor = torch.randint(low=0, high=numel, size=(idx_size,))
    IDX1: Tensor = torch.randint(low=0, high=numel, size=(0,))
    IDX2: Tensor = torch.randint(low=0, high=1, size=(idx_size,))

    def compute_take_hessians(input_tensor: Tensor, index: Tensor) -> Tensor:
        def tfunc(x: Tensor) -> Tensor:
            return torch.take(input=x, index=index).sum()

        return torch.autograd.functional.hessian(tfunc, input_tensor)

    T0_hess: Tensor = compute_take_hessians(T0, IDX0)
    T1_hess: Tensor = compute_take_hessians(T1, IDX1)
    T2_hess: Tensor = compute_take_hessians(T2, IDX2)

    O0: Tensor = torch.take(input=T0, index=IDX0).sum()
    O1: Tensor = torch.take(input=T1, index=IDX1).sum()
    O2: Tensor = torch.take(input=T2, index=IDX2).sum()

    backward(O0, order=2)
    O0.backward()
    backward(O1, order=2)
    O1.backward()
    backward(O2, order=2)
    O2.backward()

    # Check first derivative shapes and values
    for T, O, H in [(T0, O0, T0_hess), (T1, O1, T1_hess), (T2, O2, T2_hess)]:
        assert T.hgrad[0].shape == (O.numel(), T.numel())
        assert torch.allclose(T.hgrad[0].flatten(), T.grad.flatten())
        assert torch.allclose(T.hgrad[1].flatten(), H.flatten())


def test_PutXBackward0() -> None:
    # Define shape and indices
    shape: Tuple[int, int] = (10, 10)
    numel: int = shape[0] * shape[1]
    idx_sz1: int = torch.randint(low=1, high=numel, size=(1,)).item()
    idx_sz2: int = torch.randint(low=numel, high=2 * numel, size=(1,)).item()

    # Prepare inputs and sources
    T0: Tensor = torch.rand(size=shape, requires_grad=True)
    S0: Tensor = torch.rand(size=(idx_sz1,), requires_grad=True)
    IDX0: Tensor = torch.randint(low=0, high=numel, size=(idx_sz1,))

    T1: Tensor = torch.rand(size=shape, requires_grad=True)
    S1: Tensor = torch.rand(size=(idx_sz2,), requires_grad=True)
    IDX1: Tensor = torch.randint(low=0, high=numel, size=(idx_sz2,))

    T2: Tensor = torch.rand(size=shape, requires_grad=True)
    S2: Tensor = torch.rand(size=(0,), requires_grad=True)
    IDX2: Tensor = torch.randint(low=0, high=numel, size=(0,))

    T3: Tensor = torch.rand(size=shape, requires_grad=True)
    S3: Tensor = torch.rand(size=(idx_sz1,), requires_grad=True)
    IDX3: Tensor = torch.randint(low=0, high=1, size=(idx_sz1,))

    def compute_put_hessians(
        input_ref: Tensor, index: Tensor, source_ref: Tensor
    ) -> Tuple[Tensor, Tensor]:
        def pfunc(x: Tensor, s: Tensor) -> Tensor:
            return torch.put(input=x, index=index, source=s).sum()

        full_hess = torch.autograd.functional.hessian(
            func=pfunc, inputs=(input_ref, source_ref)
        )
        return full_hess[0][0], full_hess[1][1]

    T0_h, S0_h = compute_put_hessians(T0, IDX0, S0)
    T1_h, S1_h = compute_put_hessians(T1, IDX1, S1)
    T3_h, S3_h = compute_put_hessians(T3, IDX3, S3)

    O0: Tensor = torch.put(input=T0, index=IDX0, source=S0).sum()
    O1: Tensor = torch.put(input=T1, index=IDX1, source=S1).sum()
    O2: Tensor = torch.put(input=T2, index=IDX2, source=S2).sum()
    O3: Tensor = torch.put(input=T3, index=IDX3, source=S3).sum()

    backward(O0, order=2)
    O0.backward()
    backward(O1, order=2)
    O1.backward()
    backward(O2, order=2)
    O2.backward()
    backward(O3, order=2)
    O3.backward()

    # Verify first and second derivatives
    for T, S, O, Th, Sh in [
        (T0, S0, O0, T0_h, S0_h),
        (T1, S1, O1, T1_h, S1_h),
        (T3, S3, O3, T3_h, S3_h),
    ]:
        assert T.hgrad[0].shape == (O.numel(), T.numel())
        assert S.hgrad[0].shape == (O.numel(), S.numel())
        assert torch.allclose(T.hgrad[0].flatten(), T.grad.flatten())
        assert torch.allclose(S.hgrad[0].flatten(), S.grad.flatten())
        assert torch.allclose(T.hgrad[1].flatten(), Th.flatten())
        assert torch.allclose(S.hgrad[1].flatten(), Sh.flatten())
