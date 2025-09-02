# Standard Library dependencies
import math
from typing import Tuple, Union

# Pytorch dependencies
import torch
from torch import Tensor


def softmax_derivate(input: Tensor, n: int) -> Tensor:
    r"""
    Computes the n-th derivative of the softmax function f: ℝ^(B×C) → ℝ^(B×C)
    in a fully vectorized manner. Here, input is assumed to be the softmax
    probabilities, i.e. s[b, c] = exp(x[b,c]) / ∑ₖ exp(x[b,k]).

    For each sample b and output index i the derivative is given by

      D[b, i, j₁, …, jₙ] = (∂ⁿ s_i / ∂ x_{j₁} … ∂ x_{jₙ})
                         = comb(n, c_i) * s[b,i]^(1-c_i) * (-1)^(n-c_i) * ∏ₖ s[b,jₖ],

    where for each multi–index (j₁,…, jₙ) we define
      c_i = number of times i appears in (j₁,…, jₙ).

    Because softmax is computed sample–by–sample, the full derivative of the vectorized
    function (when the input is viewed as a vector of length B×C) is block–diagonal.
    (That is, nonzero entries appear only when the batch indices match.)

    Parameters
    ----------
    input : Tensor
        Tensor of shape (B, C) containing the softmax probabilities.
    n : int
        Order of the derivative (n ≥ 0). For n == 0 the function returns a flattened
        version of the input (shape (B×C,)).

    Returns
    -------
    full_deriv : Tensor
        Tensor of shape (B×C, (B×C,)*n) containing the n-th derivative.
        (If we view the softmax as f: ℝ^(B×C) → ℝ^(B×C),
         then full_deriv[k, l₁, …, lₙ] = ∂ⁿ f_k/∂ x_{l₁} … ∂ x_{lₙ}.)

    Notes
    -----
    This implementation computes the “local” (per–sample) derivative fully vectorized.
    (The only Python loop is over the batch dimension in order to embed the local
    derivatives into the full block–diagonal tensor.) Since for any x we have ∑ᵢ
    s_i(x)=1, the effective derivatives of f(x)=∑ᵢ s_i(x) must vanish. In the n=1 case
    the formula already satisfies this; for n ≥ 2 we subtract the average over the
    output index so that the local derivative, when contracted over i, is zero.
    """
    s: Tensor = input  # s has shape (B, C)
    B, C = s.shape
    N: int = B * C  # total number of outputs (and inputs)

    # For n == 0, return the softmax output as a flattened vector.
    if n == 0:
        return s.reshape(-1)

    # ---------------------------------------------------------------------
    # (1) Compute the local n-th derivative for each sample.
    # We want, for each sample b:
    #    L[b, i, j₁, …, jₙ] = comb(n, c_i)* s[b,i]^(1-c_i)* (-1)^(n-c_i)* ∏ₖ s[b,jₖ].
    # To do so, generate all multi-indices (j₁,…, jₙ) ∈ {0,…,C-1}ⁿ.
    # ---------------------------------------------------------------------
    grid: Tuple[Tensor, ...] = torch.meshgrid(
        *[torch.arange(C, device=s.device) for _ in range(n)], indexing="ij"
    )
    mesh_idx: Tensor = torch.stack(grid, dim=0)  # shape: (n, C, C, …, C)
    partial_shape: Tuple[int, ...] = tuple(mesh_idx.shape[1:])
    # (C, C, …, C) with n copies.

    # ---------------------------------------------------------------------
    # (2) Compute the product ∏ₖ s[b, jₖ] for each sample.
    # ---------------------------------------------------------------------
    mesh_flat: Tensor = mesh_idx.view(n, -1)  # shape: (n, Cⁿ)
    s_expanded: Tensor = s.unsqueeze(1).expand(-1, n, -1)  # shape: (B, n, C)
    mesh_flat_exp: Tensor = mesh_flat.unsqueeze(0).expand(B, -1, -1)
    # shape: (B, n, Cⁿ)
    gathered: Tensor = torch.gather(s_expanded, dim=2, index=mesh_flat_exp)
    # shape: (B, n, Cⁿ)
    prod_s_flat: Tensor = gathered.prod(dim=1)  # shape: (B, Cⁿ)
    prod_s: Tensor = prod_s_flat.view((B,) + partial_shape)  # shape: (B, C, …, C)

    # ---------------------------------------------------------------------
    # (3) For each output index i and each multi-index, compute c_i =
    # = (# times i appears).
    # ---------------------------------------------------------------------
    counts: Tensor
    counts = torch.zeros((C,) + partial_shape, device=s.device, dtype=torch.long)
    counts.scatter_add_(
        dim=0, index=mesh_idx, src=torch.ones_like(mesh_idx, dtype=torch.long)
    )

    # ---------------------------------------------------------------------
    # (4) Combine the factors to compute the local derivative.
    # ---------------------------------------------------------------------
    comb_table: Tensor = torch.tensor(
        [math.comb(n, r) for r in range(n + 1)], device=s.device, dtype=s.dtype
    )
    sign_table: Tensor = torch.tensor(
        [(-1) ** (n - r) for r in range(n + 1)], device=s.device, dtype=s.dtype
    )
    counts_exp: Tensor = counts.unsqueeze(0).to(
        s.dtype
    )  # shape: (1, C, *partial_shape)
    binom_factor: Tensor = comb_table[
        counts_exp.long()
    ]  # shape: (1, C, *partial_shape)
    sign_factor: Tensor = sign_table[counts_exp.long()]  # shape: (1, C, *partial_shape)

    s_i: Tensor = s.view(B, C, *([1] * n))  # shape: (B, C, 1, …, 1)
    power_term: Tensor = s_i ** (1 - counts_exp)
    prod_s_unsq: Tensor = prod_s.unsqueeze(1)  # shape: (B, 1, *partial_shape)

    local_deriv: Tensor = (
        binom_factor * sign_factor * power_term * prod_s_unsq
    )  # shape: (B, C, *partial_shape)

    # --- Correction for n == 1 ---
    # The raw formula would yield for j == i: s_i instead of the correct s_i*(1-s_i).
    if n == 1:
        I: Tensor = torch.eye(C, device=s.device, dtype=s.dtype).unsqueeze(
            0
        )  # shape: (1, C, C)
        extra: Tensor = (1 - s).unsqueeze(2) * I + (1 - I)
        local_deriv = local_deriv * extra

    # --- Correction for n ≥ 2 ---
    # Since f(x)=∑ᵢ s_i(x) ≡ 1, its derivatives vanish.
    # Enforce that the local derivative is traceless along the output index:
    if n >= 2:
        local_deriv = local_deriv - local_deriv.sum(dim=1, keepdim=True) / C

    # ---------------------------------------------------------------------
    # (5) Embed the per–sample derivative into the full block–diagonal tensor.
    # We create a tensor of shape (B, C, B, C, …, B, C) (with 1+n pairs)
    # and place the local derivative in the diagonal blocks.
    # Then flatten each (B, C) pair into one axis.
    # ---------------------------------------------------------------------
    full_shape: Tuple[int, ...] = (B, C) + tuple(x for _ in range(n) for x in (B, C))
    full_deriv: Tensor = torch.zeros(full_shape, device=s.device, dtype=s.dtype)
    for b in range(B):
        idx: Tuple[Union[int, slice], ...]
        idx = (b, slice(None)) + tuple(x for _ in range(n) for x in (b, slice(None)))
        full_deriv[idx] = local_deriv[b]

    new_shape: Tuple[int, ...] = (B * C,) + tuple(B * C for _ in range(n))
    full_deriv = full_deriv.reshape(new_shape)

    return full_deriv


def reduced_softmax_derivate(input: torch.Tensor, order: int) -> torch.Tensor:
    r"""
    Analytic n-th derivative of softmax via one integer‑indexed einsum:
      D^n s_c = sum_k s_k ∏_{m=0..n} (δ_{k,dim_m} - s_{dim_m}),
    where dim_0 ≡ c, dim_1..dim_n ≡ j₁..jₙ.
    Supports arbitrary order ≥ 0.
    """
    eps = float(torch.finfo(torch.float32).eps)

    # 1) clamp & flatten batch
    s: Tensor = input.clamp(min=eps, max=1 - eps)
    batch_shape: Tuple[int, ...]
    C: int
    batch_shape, C = tuple(s.shape[:-1]), s.shape[-1]
    flat: Tensor = s.reshape(-1, C)  # (B, C)
    B = flat.shape[0]

    # 0th derivative is just the softmax itself
    if order == 0:
        return flat.reshape(*batch_shape, C)

    # 2) build diff[b,k,x] = δ_{k,x} - s[b,x]
    id_mat: Tensor = torch.eye(C, device=s.device, dtype=s.dtype)
    diff: Tensor = id_mat.unsqueeze(0).expand(B, C, C) - flat.unsqueeze(1)  # (B,k,x)

    # 3) set up integer subscripts for einsum
    #    dims: 0=batch, 1=k, 2=c, 3..(2+order)=j₁..jₙ
    operands: list[Tensor] = []
    subscriptss: list[list[int]] = []

    # a) the s_k term: shape (B,k) → dims [0,1]
    operands.append(flat)
    subscriptss.append([0, 1])

    # b) the (δ - s) factor for the c‑index: shape (B,k,c) → dims [0,1,2]
    operands.append(diff)
    subscriptss.append([0, 1, 2])

    # c) one more (δ - s) for each derivative axis j₁…jₙ:
    #    shape (B,k,c) but assign its last dim to 3,4,…,2+order
    for m in range(order):
        operands.append(diff)
        subscriptss.append([0, 1, 3 + m])

    # output should keep batch=0, then c=2, then j₁..jₙ = 3..(2+order)
    output_subs: list[int] = [0, 2] + [3 + m for m in range(order)]

    # 4) perform einsum: sum over k=1 dimension 1
    #    torch.einsum(x1, subs1, x2, subs2, …, out_subs)
    einsum_args: list[object] = []
    for operand, subs in zip(operands, subscriptss):
        einsum_args.extend([operand, subs])
    einsum_args.append(output_subs)

    deriv: Tensor = torch.einsum(*einsum_args)
    # deriv shape: (B, C, *[C]*order)

    # 5) reshape back to original batch dims
    return deriv.reshape(*batch_shape, C, *([C] * order))
