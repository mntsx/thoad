# Standard Library dependencies
from typing import Tuple, Union

# Pytorch dependencies
import torch
from torch import Tensor


def logsoftmax_derivate(input: Tensor, n: int) -> Tensor:
    r"""
    Computes the n-th derivative of the log-softmax function
      f : ℝ^(B×C) → ℝ^(B×C),
    in a block–diagonal, fully vectorized manner.

    The result is a (B*C, (B*C)^n) tensor storing ∂^n f_k / ∂ x_{ℓ1} … ∂ x_{ℓn}.

    Parameters
    ----------
    input : Tensor
        Shape (B, C). Each row is log_softmax(x[b]) for some x[b].
    n : int
        Derivative order (n ≥ 0).

    Returns
    -------
    full_deriv : Tensor
        Shape (B*C, (B*C)^n), storing the n-th derivative in block–diagonal form.

    Notes
    -----
    1) For n=0, we just return the flattened log_softmax output.
    2) For n=1, the known derivative is  ∂/∂x_j [ log_softmax_i(x) ] = δ_{i,j} - s_j.
    3) For n≥2, only the derivatives of -log(∑ e^{x_k}) matter (the xᵢ part vanishes).
       We compute this by a combinatorial expansion in the softmax probabilities s.
    4) We flatten the indexing tuple for embedding into out[...] so that PyTorch
       does not complain "Could not infer dtype of slice".
    """
    B, C = input.shape
    N: int = B * C

    # ------------------------------
    # Case: n=0 => just flatten log-softmax outputs
    # ------------------------------
    if n == 0:
        return input.view(-1)

    # Recover the underlying softmax probabilities s = exp(log s).
    s: Tensor = input.exp()  # shape (B, C)

    # We'll create an output container of shape (B,C) + (B,C)*n,
    # which is a single tuple (B,C, B,C, ..., B,C).
    full_shape: Tuple[int, ...] = (B, C) + (B, C) * n
    out: Tensor = torch.zeros(full_shape, dtype=input.dtype, device=input.device)

    # ------------------------------------------------------------------------
    # Helper: partial^n of g(x) = -log( sum_k exp(x_k) ).
    # Returns shape (B, C^n).  We'll embed it for the log-softmax derivative.
    # ------------------------------------------------------------------------
    def nth_deriv_neglogsumexp(s: Tensor, order: int) -> Tensor:
        """
        Returns the n-th derivative of g(x)= -log( sum_k e^{x_k} ) w.r.t. x_j1.. x_jn
        in a combinatorial, fully vectorized style.

        Output shape: (B, C^order).
        """
        B_, C_ = s.shape

        # For order=1, ∂g/∂x_j = -s_j.
        if order == 1:
            return -s.view(B_, C_)

        # For order >= 2, do a stepwise recursion:
        #   g^(1) = -s_j
        #   g^(k) = ∂/∂ x_{j1} [ g^(k-1)( j2, ..., j_k ) ]
        gk: Tensor = (-s).view(B_, C_)  # k=1

        for k_ in range(2, order + 1):
            old_shape: Tuple[int, ...] = (B_,) + (C_,) * (k_ - 1)
            gk_tensor: Tensor = gk.view(old_shape)  # => (B_, C_, C_, ..., C_)

            # Build a meshgrid for (j1, j2, ..., j_k_).
            grid: Tuple[Tensor, ...] = torch.meshgrid(
                *[torch.arange(C_, device=s.device) for _ in range(k_)], indexing="ij"
            )
            # mesh_idx => shape (k_, C_, C_, ..., C_)
            mesh_idx: Tensor = torch.stack(grid, dim=0)

            partial_shape: Tuple[int, ...] = tuple(mesh_idx.shape[1:])  # => (C_,)*(k_)

            # j1_idx => mesh_idx[0], j_rest => mesh_idx[1:]
            j1_idx: Tensor = mesh_idx[0]  # => shape (C_,)*(k_)
            j_rest: Tensor = mesh_idx[1:]  # => shape (k_-1, (C_,)*(k_))

            # 1) Gather from gk_tensor according to (j2.. j_k_).
            gk_flat: Tensor = gk_tensor.view(B_, -1)  # => (B_, C_^(k_-1))

            j_rest_flat: Tensor = j_rest.view(k_ - 1, -1)  # => (k_-1, C_^(k_))

            base_vals: list[int] = [C_**p for p in reversed(range(k_ - 1))]
            base_powers: Tensor = torch.tensor(
                base_vals, device=s.device, dtype=torch.long
            ).view(k_ - 1, 1)

            linear_index: Tensor = (j_rest_flat * base_powers).sum(
                dim=0
            )  # => (C_^(k_))

            gathered_poly: Tensor = torch.gather(
                gk_flat, dim=1, index=linear_index.unsqueeze(0).expand(B_, -1)
            )
            # reshape => (B_, (C_,)*(k_))
            gathered_poly = gathered_poly.view((B_,) + partial_shape)

            # 2) derivative wrt x_{j1}: factor = c - (k_-1)* s_{ j1 }
            counts_rest: Tensor = torch.zeros(
                (C_,) + partial_shape, dtype=torch.long, device=s.device
            )
            for a in range(C_):
                counts_rest[a] = (j_rest == a).sum(dim=0)

            counts_rest_flat: Tensor = counts_rest.view(C_, -1)
            j1_idx_flat: Tensor = j1_idx.view(-1)

            c_vals: Tensor = torch.gather(counts_rest_flat, 0, j1_idx_flat.unsqueeze(0))
            c_vals = c_vals.view(partial_shape)

            s_j1: Tensor = torch.gather(
                s, dim=1, index=j1_idx_flat.unsqueeze(0).expand(B_, -1)
            ).view((B_,) + partial_shape)

            factor: Tensor = c_vals.to(s.dtype) - (k_ - 1) * s_j1
            new_gk: Tensor = gathered_poly * factor
            gk = new_gk.view(B_, -1)  # => (B_, C_^k_)

        return gk  # shape => (B_, C_^order)

    # -----------------------------------------------------------
    # 1) If n=1: derivative is δ_{i,j} - s_j.  Embed block-diagonal.
    # -----------------------------------------------------------
    if n == 1:
        I: Tensor
        I = torch.eye(C, dtype=s.dtype, device=s.device).unsqueeze(0)  # => (1, C, C)

        # FIX: subtract s along the last dimension (j):
        local_deriv: Tensor = I - s.unsqueeze(1)  # => shape (B, C, C)

        for b in range(B):
            # We want out[b, i, b, j] = local_deriv[b, i, j]
            idx: Tuple[Union[int, slice], ...] = (b, slice(None), b, slice(None))
            out[idx] = local_deriv[b]

    # -----------------------------------------------------------
    # 2) If n>=2: derivative from -log( sum e^{x_k} ) => replicate across i
    # -----------------------------------------------------------
    else:
        local_g: Tensor = nth_deriv_neglogsumexp(s, n)  # shape => (B, C^n)
        local_g_tensor: Tensor = local_g.view((B,) + (C,) * n)  # => (B, C, C, ..., C)

        for b in range(B):
            # For each sample b and each output i, fill with the same local_g_tensor
            # because for n≥2 the x_i part’s second derivative is 0.
            idx_list: list[Union[int, slice]] = [b, slice(None)]  # (b, i)
            for _ in range(n):
                idx_list.append(b)  # (b, i, b, j1, b, j2, ...)
                idx_list.append(slice(None))
            out_idx: Tuple[Union[int, slice], ...] = tuple(idx_list)

            # local_g_tensor[b] has shape (C, ...), so we unsqueeze(0)
            #   for the i dimension
            out[out_idx] = local_g_tensor[b].unsqueeze(0)

    # Finally flatten => (B*C, (B*C)^n)
    return out.view(N, *(N,) * n)


def reduced_logsoftmax_derivate(
    input: torch.Tensor, order: int, eps: float = 1e-6
) -> torch.Tensor:
    r"""
    n‑th derivatives of log‑softmax (ℓ_i = x_i − log∑ₖe^{xₖ}) via one integer‑indexed
    einsum.
    - order=0: returns ℓ (the log‑softmax values).
    - order=1: returns Jacobian ∂ℓ_i/∂x_j = δ_{i,j} − s_j.
    - order≥2: returns ∂ⁿℓ_i/∂x_{j₁}…∂x_{jₙ}
       = − ∑ₖ sₖ ∏ₘ (δ_{k,jₘ} − s_{jₘ}),
      shaped (… , C, C, …, C) with (order+1) C‑axes.
    """
    # 1) compute & clamp softmax probabilities
    s: Tensor = input.softmax(dim=-1).clamp(min=eps, max=1 - eps)
    batch_shape: Tuple[int, ...]
    C: int
    batch_shape, C = tuple(s.shape[:-1]), s.shape[-1]
    flat: Tensor = s.reshape(-1, C)  # (B, C)
    B = flat.shape[0]

    # 0th derivative = log‑softmax values
    if order == 0:
        # ℓ = x − logsumexp(x)
        logz: Tensor = torch.logsumexp(input, dim=-1, keepdim=True)
        l: Tensor = input - logz
        return l.reshape(*batch_shape, C)

    # 2) diff[b,k,j] = δ_{k,j} − s[b,j]
    eye: Tensor = torch.eye(C, device=input.device, dtype=input.dtype)
    diff: Tensor = eye.unsqueeze(0).expand(B, C, C) - flat.unsqueeze(1)  # (B, k, j)

    # 1st derivative: Jacobian
    if order == 1:
        # ∂ℓ_i/∂x_j = δ_{i,j} − s_j  → diff[b,i,j]
        return diff.reshape(*batch_shape, C, C)

    # 3) build integer‑indexed einsum for ∑ₖ sₖ · ∏ₘ diff[..., k, jₘ]
    #
    # dims: 0 = batch, 1 = k, 2..(2+order-1) = j₁..jₙ
    operands: list[Tensor] = []
    subscriptss: list[list[int]] = []

    # a) the sₖ term: shape (B,k)
    operands.append(flat)
    subscriptss.append([0, 1])

    # b) one diff for each derivative axis j₁…jₙ
    for m in range(order):
        operands.append(diff)
        subscriptss.append([0, 1, 2 + m])

    # output keeps batch=0, then output‑index i=???
    #   for order>=2, ∂ⁿℓ_i is independent of i, so we broadcast across i
    #   j‑axes occupy dims 2..(2+order-1)
    output_subs: list[int] = [0] + [2 + m for m in range(order)]

    # 4) sum over k=1
    einsum_args: list[object] = []
    for op, subs in zip(operands, subscriptss):
        einsum_args.extend([op, subs])
    einsum_args.append(output_subs)

    base: Tensor = torch.einsum(*einsum_args)  # (B, C,…,C) but actually (B,[C]^order)
    # 5) attach minus‐sign and broadcast over the i‑axis
    deriv: Tensor = -base.unsqueeze(1).expand(B, C, *([C] * order))
    return deriv.reshape(*batch_shape, C, *([C] * order))
