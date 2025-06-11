# Standard Library Dependencies
import math
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


def reduced_softmax_derivate(input: Tensor, order: int) -> Tensor:
    r"""
    Computes the n-th derivative of the softmax function f: ℝ^(B×C) → ℝ^(B×C)
    in a fully vectorized manner. Here, input is assumed to be the softmax
    probabilities
    i.e. s[b, c] = exp(x[b,c]) / ∑ₖ exp(x[b,k]).

    For each sample b and output index i the derivative is given by

      L[b, i, j₁, …, jₙ] = (∂ⁿ s_i / ∂ x_{j₁} … ∂ x_{jₙ})
                         = comb(n, c_i) * s[b,i]^(1-c_i) * (-1)^(n-c_i) * ∏ₖ s[b,jₖ],

    where for each multi–index (j₁,…, jₙ) we define
      c_i = number of times i appears in (j₁,…, jₙ).

    Since the softmax is applied sample–by–sample, the n-th derivative contains no
    cross–sample interactions. Thus the output is simply of shape:

         (B, C, C, …, C)
         (with 1+n tensor dimensions; the derivative indices are dual dimensions).

    Parameters
    ----------
    input : Tensor
        Tensor of shape (B, C) containing the softmax probabilities.
    order : int
        Order of the derivative (order ≥ 0). For order == 0 the function returns the
        input.

    Returns
    -------
    deriv : Tensor
        Tensor of shape (B, C, (C,)*n) containing the n-th derivative.
        For example, for n=1 the output has shape (B, C, C) representing the per-sample
        Jacobian.

    Notes
    -----
    For n == 1, the raw closed-form formula would yield diagonal entries equal to
    s[b,i] rather than the correct s[b,i]*(1-s[b,i]), so we apply a correction. For
    n ≥ 2 we subtract the average over the output index to ensure that for
    f(x)=∑₍i₎ s_i(x) ≡ 1 the derivative vanishes.
    """
    s: Tensor = input
    n: int = order
    # reshape input to (B, C)
    batch_shape: Tuple[int, ...] = s.shape[:-1]
    batch_size: int = math.prod(batch_shape)
    s = s.reshape(batch_size, -1)
    B, C = s.shape

    # For n == 0, just return s.
    if n == 0:
        return s

    # ---------------------------------------------------------------------
    # (1) Compute the local n-th derivative for each sample.
    # We wish to compute, for each sample b:
    #    L[b, i, j₁, …, jₙ] = comb(n, c_i)* s[b,i]^(1-c_i)* (-1)^(n-c_i)* ∏ₖ s[b,jₖ].
    # To do so, we first generate all multi-indices (j₁,…, jₙ) ∈ {0,…, C-1}ⁿ.
    # ---------------------------------------------------------------------
    grid = torch.meshgrid(
        *[torch.arange(C, device=s.device) for _ in range(n)], indexing="ij"
    )
    mesh_idx: Tensor = torch.stack(grid, dim=0)  # shape: (n, C, C, …, C)
    partial_shape: torch.Size = mesh_idx.shape[1:]  # (C, C, …, C) with n copies.

    # ---------------------------------------------------------------------
    # (2) Compute the product ∏ₖ s[b, jₖ] for each sample.
    # ---------------------------------------------------------------------
    mesh_flat: Tensor = mesh_idx.view(n, -1)  # shape: (n, C^n)
    s_expanded: Tensor = s.unsqueeze(1).expand(-1, n, -1)  # shape: (B, n, C)
    mesh_flat_exp: Tensor = mesh_flat.unsqueeze(0).expand(
        B, -1, -1
    )  # shape: (B, n, C^n)
    gathered: Tensor = torch.gather(
        s_expanded, dim=2, index=mesh_flat_exp
    )  # shape: (B, n, C^n)
    prod_s_flat: Tensor = gathered.prod(dim=1)  # shape: (B, C^n)
    prod_s: Tensor = prod_s_flat.view((B,) + partial_shape)  # shape: (B, C, …, C)

    # ---------------------------------------------------------------------
    # (3) For each output index i and for each multi-index, compute c_i = (# i appears).
    # ---------------------------------------------------------------------
    counts: Tensor = torch.zeros(
        (C,) + partial_shape, device=s.device, dtype=torch.long
    )
    counts.scatter_add_(
        dim=0, index=mesh_idx, src=torch.ones_like(mesh_idx, dtype=torch.long)
    )

    # ---------------------------------------------------------------------
    # (4) Combine factors to compute the local derivative.
    # ---------------------------------------------------------------------
    comb_table: Tensor = torch.tensor(
        [math.comb(n, r) for r in range(n + 1)], device=s.device, dtype=s.dtype
    )
    sign_table: Tensor = torch.tensor(
        [(-1) ** (n - r) for r in range(n + 1)], device=s.device, dtype=s.dtype
    )
    counts_exp: Tensor = counts.unsqueeze(0).to(s.dtype)  # (1, C, *partial_shape)
    binom_factor: Tensor = comb_table[counts_exp.long()]  # (1, C, *partial_shape)
    sign_factor: Tensor = sign_table[counts_exp.long()]  # (1, C, *partial_shape)

    s_i: Tensor = s.view(B, C, *([1] * n))  # shape: (B, C, 1, …, 1)
    power_term: Tensor = s_i ** (1 - counts_exp)
    prod_s_unsq: Tensor = prod_s.unsqueeze(1)  # shape: (B, 1, *partial_shape)

    local_deriv: Tensor = (
        binom_factor * sign_factor * power_term * prod_s_unsq
    )  # shape: (B, C, *partial_shape)

    # --- Correction for n == 1 ---
    # The raw formula would yield for j == i: s[b,i] rather than s[b,i]*(1-s[b,i]).
    if n == 1:
        I: Tensor = torch.eye(C, device=s.device, dtype=s.dtype).unsqueeze(
            0
        )  # shape: (1, C, C)
        extra: Tensor = (1 - s).unsqueeze(2) * I + (1 - I)
        local_deriv = local_deriv * extra

    # --- Correction for n ≥ 2 ---
    # Since f(x)=∑_i s_i(x) ≡ 1, its derivative must vanish.
    if n >= 2:
        local_deriv = local_deriv - local_deriv.sum(dim=1, keepdim=True) / C

    # local_deriv now has shape (B, C, (C,)*n); note that there is only one batch dim.
    # finally, unflatten batch:
    deriv_shape: Tuple[int, ...] = local_deriv.shape
    final_shape: Tuple[int, ...] = [*batch_shape, *deriv_shape[1:]]
    local_deriv = local_deriv.reshape(shape=final_shape)

    return local_deriv


class SoftmaxXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim: int = self._context["saved_dim"]
        saved_result: Tensor = self._context["saved_result"]
        projected_shape: Shape = tuple(saved_result.shape)
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        # desindep the contracted dimension
        aux: list[Union[None, int]] = list(projected_indep)
        if saved_dim in projected_indep:
            aux[projected_indep.index(saved_dim)] = None
        projected_indep = tuple(aux)
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: Tuple[int, ...] = self._grad_fn._saved_dim
        saved_result: Tensor = self._grad_fn._saved_result
        # ensure proper tensor configuration
        saved_result = saved_result.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_result"] = saved_result
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim: Tensor = self._context["saved_dim"]
        saved_result: Tensor = self._context["saved_result"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = saved_dim
        processed_context["output"] = saved_result
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dim: int = self._processed_context["dim"]
        output: Tensor = self._processed_context["output"]

        ### Carry out instrumental operations
        ndim: int = output.ndim
        dim_size: int = output.shape[dim]
        # treat batch dims
        batch_range: Tuple[int, ...] = tuple(d for d in range(ndim) if d != dim)
        batch_shape: int = [sz for i, sz in enumerate(output.shape) if i != dim]
        # permute output placing interest dimension at the end
        permutation: Tuple[int, ...] = (*batch_range, dim)
        permuted_output: Tensor = output.permute(permutation)
        # arange einsum indices
        dual_range: list[int] = [d for d in range(ndim, ndim + order)]
        internal_indices: list[int]
        internal_indices = [*permutation, *[d for d in dual_range]]
        composed_indices: list[list[int]] = list()
        pre_batch: list[int] = batch_range[:dim]
        pos_batch: list[int] = batch_range[dim:]
        for d in dual_range:
            composed_indices.append([*pre_batch, d, *pos_batch])

        ### Instantiate differential
        differential: Tensor = reduced_softmax_derivate(
            input=permuted_output,
            order=order,
        )

        ### Create einstein notation
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = composed_indices
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([[*batch_shape, *((dim_size,) * (1 + order))]])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        if out_id == 0 and all(i == 0 for i in in_id):
            order: int = len(in_id)
            (differential, einstein_notation) = self._compute_internal_0(order=order)
        else:
            (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
