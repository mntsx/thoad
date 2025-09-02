# Standard Library dependencies
import math
from typing import Tuple

# Pytorch dependencies
import torch
from torch import Tensor


def prod0_derivate(tensor: torch.Tensor, order: int, eps: float = 1e-8) -> torch.Tensor:
    """
    N-th derivative of prod(x):
      shape (1, d**order), entry = prod(x)/(x[i1]...x[iN]) if all i's distinct.
    """
    # 1) flatten & safe-invert
    x_flat: Tensor = tensor.flatten() + eps  # (d,)
    d: int = x_flat.numel()
    f: Tensor = x_flat.prod()  # scalar
    inv: Tensor = x_flat.reciprocal()  # (d,)

    # cache for masks so we only build once per (d, order)
    _prod0_mask_cache: dict[tuple[int, int], Tensor] = {}

    # 2) get or build float mask of shape (d,)*order with 1 for all-distinct
    key: tuple[int, int] = (d, order)
    if key in _prod0_mask_cache:
        mask_f: Tensor = _prod0_mask_cache[key]
    else:
        idx: Tensor = torch.arange(d, device=tensor.device)
        grids: Tuple[Tensor, ...] = torch.meshgrid(*([idx] * order), indexing="ij")
        mask: Tensor = torch.ones(
            grids[0].shape, dtype=torch.bool, device=tensor.device
        )
        for i in range(order):
            for j in range(i + 1, order):
                mask &= grids[i] != grids[j]
        mask_f = mask.to(inv.dtype)
        _prod0_mask_cache[key] = mask_f

    # 3) build inv_outer = inv[...,i1]*...*inv[...,iN] via views
    inv_outer: Tensor = inv.view(*([d] + [1] * (order - 1)))
    for k in range(1, order):
        inv_outer = inv_outer * inv.view(*([1] * k + [d] + [1] * (order - k - 1)))

    # 4) apply mask, scale by f, reshape
    deriv: Tensor = f * inv_outer * mask_f  # (d,)*order
    return deriv.reshape((1,) + (d,) * order)


def prod1_derivate(
    input: Tensor, output: Tensor, dim: int, order: int, eps: float = 1e-8
) -> Tensor:
    """
    Vectorized n-th cross-derivative of prod(input, dim).
    """

    # identify batch dims (all except `dim`)
    batch_dims: list[int] = [d for d in range(input.ndim) if d != dim]
    # permute `input` so that `dim` is last
    perm: list[int] = batch_dims + [dim]

    # permuted input with epsilon for stability
    x: Tensor = (input + eps).permute(*perm).contiguous()
    shp: tuple[int, ...] = x.shape
    B: int = math.prod(shp[:-1])  # total batch size
    N: int = shp[-1]  # size along the reduced dim

    # flatten into (B, N)
    x_flat: Tensor = x.view(B, N)
    # flatten output into (B,)
    y_flat: Tensor = output.reshape(B)

    # compute 1/x
    inv: Tensor = x_flat.reciprocal()

    # build a boolean mask of shape (N,)*order that is True when all indices are distinct
    idx: Tensor = torch.arange(N, device=input.device)
    grids: Tuple[Tensor, ...] = torch.meshgrid(*([idx] * order), indexing="ij")
    mask: Tensor = torch.ones_like(grids[0], dtype=torch.bool)
    for i in range(order):
        for j in range(i + 1, order):
            mask &= grids[i] != grids[j]
    mask_f: Tensor = mask.to(inv.dtype)

    # build the outer product inv[..., i1]*...*inv[..., in]
    inv_outer: Tensor = inv.view(B, *([N] + [1] * (order - 1)))
    for ax in range(1, order):
        inv_outer = inv_outer * inv.view(B, *([1] * ax + [N] + [1] * (order - ax - 1)))

    # combine y, inv_outer, and mask to get the derivative
    der_flat: Tensor = y_flat.view(B, *([1] * order)) * inv_outer * mask_f
    # reshape back to (...batch_dims, N, …, N)
    out_shape: list[int] = [input.shape[d] for d in batch_dims] + [N] * order
    return der_flat.reshape(*out_shape)
