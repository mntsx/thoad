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


def reduced_logsoftmax_derivate(input: Tensor, order: int) -> Tensor:
    r"""
    Computes the n-th derivative of the log-softmax function
      f: ℝ^(B×C) → ℝ^(B×C),
    *per sample*, yielding a tensor of shape (B, C, C, …, C) with n derivative axes.

    Parameters
    ----------
    input : Tensor
        Shape (B, C).  Each row is log_softmax(x[b]) for some x[b].
    n : int
        Derivative order (n ≥ 0).

    Returns
    -------
    deriv : Tensor
        Shape (B, C, C, …, C) (1 + n axes after B).  For n=0, this is just (B, C).
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
        # reshape as batch
        deriv_shape: Tuple[int, ...] = s.shape
        final_shape: Tuple[int, ...] = [*batch_shape, *deriv_shape[1:]]
        local_deriv = s.reshape(shape=final_shape)

    # Recover the underlying softmax probabilities s = exp(log s).
    s = s.exp()  # shape (B, C)

    # --- If n=1, we have derivative: δ_{i,j} - s_j
    if n == 1:
        # local_deriv[b, i, j] = delta_{i,j} - s[b, j]
        # Shape = (B, C, C)
        I: Tensor
        I = torch.eye(C, dtype=s.dtype, device=s.device).unsqueeze(0)  # (1, C, C)
        # s[b] has shape (C,). We want to broadcast it to (C, C) in j dimension.
        local_deriv = I - s.unsqueeze(1)  # => (B, C, C)

        # reshape as batch
        deriv_shape: Tuple[int, ...] = local_deriv.shape
        final_shape: Tuple[int, ...] = [*batch_shape, *deriv_shape[1:]]
        local_deriv = local_deriv.reshape(shape=final_shape)

        return local_deriv

    # --- For n >= 2, only -log(Σ e^x) contributes. Replicate that derivative across i.
    # We'll define a helper to compute the n-th derivative of g(x)= -log(sum_k e^{x_k})
    def nth_deriv_neglogsumexp(s: Tensor, order: int) -> Tensor:
        """
        Returns the n-th derivative of g(x)= -log( sum_k e^{x_k} )
            w.r.t. x_{j1}.. x_{j_n},
        as a tensor of shape (B, C^order).

        We'll do a stepwise recursion approach to generate all partial derivatives.
        """
        B_, C_ = s.shape

        # For order=1, ∂g/∂x_j = - s_j
        if order == 1:
            return -s.view(B_, C_)

        # Initialize g^(1) = -s_j
        gk = -s.view(B_, C_)

        # Recursively take derivatives up to g^(order)
        for k_ in range(2, order + 1):
            # gk has shape (B_, C_^(k_-1)) if we've done k_-1 steps
            old_shape: Tuple[int] = (B_,) + (C_,) * (k_ - 1)
            gk_tensor: Tensor = gk.view(old_shape)  # => (B_, C_, C_, ..., C_)

            # Build a meshgrid for (j1, j2, ..., j_k_)
            grid: Tuple[Tensor] = torch.meshgrid(
                *[torch.arange(C_, device=s.device) for _ in range(k_)], indexing="ij"
            )
            mesh_idx: Tensor
            mesh_idx = torch.stack(grid, dim=0)  # => shape (k_, C_, C_, ..., C_)
            partial_shape: torch.Size = mesh_idx.shape[1:]  # => (C_,)*(k_)

            # j1_idx = mesh_idx[0], j_rest = mesh_idx[1:]
            j1_idx: Tensor = mesh_idx[0]  # shape (C_,...)*(k_)
            j_rest: Tensor = mesh_idx[1:]  # shape (k_-1, (C_,...)*(k_))

            # 1) Gather from gk_tensor according to (j2.. j_k_)
            gk_flat: Tensor = gk_tensor.view(B_, -1)  # => (B_, C_^(k_-1))
            j_rest_flat: Tensor = j_rest.view((k_ - 1), -1)  # => (k_-1, C_^(k_))

            # We'll convert j_rest_flat to a single "linear index"
            #   so we can gather at once.
            # e.g. if each dimension is base C_, then linear index is:
            #   j2*C_^(k_-2) + j3*C_^(k_-3) + ...
            base_vals: list[Any] = [C_**p for p in reversed(range(k_ - 1))]
            base_powers: Tensor = torch.tensor(base_vals, device=s.device).view(
                k_ - 1, 1
            )

            linear_index: Tensor = (j_rest_flat * base_powers).sum(
                dim=0
            )  # => shape (C_^(k_))

            gathered_poly: Tensor = torch.gather(
                gk_flat, dim=1, index=linear_index.unsqueeze(0).expand(B_, -1)
            )
            # Reshape => (B_, C_,...,C_)
            gathered_poly = gathered_poly.view((B_,) + partial_shape)

            # 2) derivative wrt x_{j1}: factor = c_j1 - (k_-1)* s_{j1}
            #    where c_j1 = how many times j1 appears among j2.. j_k_.
            # We'll count how often each index a=0..C_-1 appears among j_rest.
            counts_rest: Tensor = torch.zeros(
                (C_,) + partial_shape, dtype=torch.long, device=s.device
            )
            ones_like_rest: Tensor = torch.ones_like(j_rest, dtype=torch.long)
            counts_rest.scatter_add_(dim=0, index=j_rest, src=ones_like_rest)

            j1_idx_flat: Tensor = j1_idx.view(-1)  # => shape (C_^(k_))
            counts_rest_flat: Tensor = counts_rest.view(C_, -1)  # => (C_, C_^(k_))

            # gather row = j1_idx_flat => shape (1, C_^(k_))
            c_vals: Tensor = torch.gather(counts_rest_flat, 0, j1_idx_flat.unsqueeze(0))
            c_vals = c_vals.view(partial_shape)  # => shape (C_,...,C_)

            s_j1 = torch.gather(
                s, dim=1, index=j1_idx_flat.unsqueeze(0).expand(B_, -1)
            ).view((B_,) + partial_shape)

            # factor = c_vals - (k_-1)* s_j1
            factor: Tensor = c_vals.to(s.dtype) - (k_ - 1) * s_j1

            new_gk: Tensor = gathered_poly * factor
            gk: Tensor = new_gk.view(B_, -1)  # => shape (B_, C_^k_)

        return gk  # => shape (B_, C_^order)

    # -- For n >= 2:
    # derivative
    #   w.r.t. x_{j1}... x_{j_n} is exactly the nth derivative of -logSumExp(...)
    # because the derivative of x_i vanishes at order >= 2.
    local_g: Tensor = nth_deriv_neglogsumexp(s, n)  # => shape (B, C^n)
    local_g_tensor: Tensor = local_g.view((B,) + (C,) * n)  # => (B, C, C, ..., C)

    # Replicate this same n-th derivative for each output index i
    #   => shape (B, C, C, ..., C)
    # Because for n>=2, ∂^n[x_i] = 0, so the entire derivative doesn't depend on i.
    # Just broadcast along a new "i" dimension of size C:
    local_deriv: Tensor = local_g_tensor.unsqueeze(dim=1).expand(B, C, *([C] * n))

    # Finally, unflatten batch:
    deriv_shape: Tuple[int, ...] = local_deriv.shape
    final_shape: Tuple[int, ...] = [*batch_shape, *deriv_shape[1:]]
    local_deriv = local_deriv.reshape(shape=final_shape)

    return local_deriv


class LogSoftmaxXBackward0(ContractiveFunction):

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
        differential: Tensor = reduced_logsoftmax_derivate(
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
