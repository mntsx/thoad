# Standard Library dependencies
from typing import Union, Tuple

# Pytorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.derivate.sigmoid import sigmoid_derivate


def reduced_glu_derivate(input: Tensor, dim: int, order: int) -> Tensor:
    r"""
    Computes the local (reduced) n-th derivative of the GLU function
      f(a,b) = a ⊙ σ(b)
    where the input is assumed to be split along dimension `dim` into two halves,
    a and b, and σ is the sigmoid.

    In our reduced derivative (of shape (…, C, (2C,)*n) where C is half of the size
    along the specified dimension), the only nonzero entries are:

      (a) When all derivative directions hit the b–branch:
            D[..., i, (i+C,..., i+C)] = a[..., i] * sigmoid_derivate(σ(b)[..., i], n)

      (b) When exactly one derivative direction (say, the k-th one) hits the a–branch and
          the remaining ones hit b:
            D[..., i, ( …, i (at slot k), …, i+C (elsewhere)] =
                      sigmoid_derivate(σ(b)[..., i], n-1)

      Except for n==1 where we set
            deriv_a = σ(b)   (for the a–branch)
      rather than the n–1 (i.e. 0–th) derivative.

    Parameters
    ----------
    input : Tensor
        The full input tensor of shape (..., 2C) along dimension `dim`.
    dim : int
        The dimension along which to split the input into a and b.
    n : int
        The derivative order (n >= 1).

    Returns
    -------
    Tensor
        A tensor of shape (batch..., C, (2C,)*n) representing the local derivative.
    """
    # --- Permute so that the specified dim becomes the last dimension ---
    perm: Union[None, list[int]]
    if dim != input.ndim - 1:
        perm = [i for i in range(input.ndim) if i != dim] + [dim]
        input_perm: Tensor = input.permute(perm)
    else:
        input_perm = input
        perm = None  # no permutation performed

    if input_perm.ndim == 1:
        input_perm = input_perm.unsqueeze(0)

    # --- Split input into a and b ---
    # Here the last dimension has size 2C.
    a: Tensor
    b: Tensor
    a, b = torch.chunk(input_perm, 2, dim=-1)  # both of shape (..., C)
    sigma_b: Tensor = torch.sigmoid(b)

    # --- Compute the two “ingredients” ---
    # For the a–branch: if first derivative use σ(b), else (for orders>=2) zero.
    deriv_a: Tensor
    if order == 1:
        deriv_a = sigma_b
    else:
        deriv_a = torch.zeros_like(sigma_b, dtype=input.dtype, device=input.device)
    # For the b–branch:
    deriv_b: Tensor = a * sigmoid_derivate(sigma_b, order)
    # And for the mixed (one a, rest b) case:
    deriv_ab: Tensor = sigmoid_derivate(sigma_b, order - 1)

    # --- Build the full reduced derivative tensor ---
    # Let the “primal” (output) channels come from a (of size C) and the dual (input)
    # indices run over 2C (first half corresponding to a and second half to b).
    # Our desired output shape is:
    #    (..., C, (2C,)*n)
    # where ... is the batch part (all dimensions except the last one of input_perm).
    base_shape: Tuple[int, ...] = tuple(a.shape[:-1])  # batch dimensions
    C = a.shape[-1]
    out_shape: Tuple[int, ...] = base_shape + (C,) + (2 * C,) * order
    D: Tensor = torch.zeros(out_shape, dtype=input.dtype, device=input.device)

    # We'll work on the "primal" (batch+channel) part flattened.
    # Let N be the number of elements in the batch+channel (i.e. prod(base_shape)*C).
    D_flat: Tensor = D.view(-1, *((2 * C,) * order))
    N = D_flat.shape[0]

    # Also flatten our “ingredient” tensors.
    flat_deriv_a: Tensor = deriv_a.reshape(-1)  # shape: (N,)
    flat_deriv_b: Tensor = deriv_b.reshape(-1)  # shape: (N,)
    flat_deriv_ab: Tensor = deriv_ab.reshape(-1)  # shape: (N,)

    # For each element, we need to know its channel index (i in 0,..., C-1).
    # If we assume the flattened ordering is such that the last axis (of a) varies fastest,
    # then we can compute:
    flat_channel: Tensor = torch.arange(N, device=input.device) % C  # shape: (N,)
    idx_all: Tensor = torch.arange(N, device=input.device)

    # (A) For the case when all derivative directions hit b.
    # That is, for each element the derivative multi-index is (i+C, i+C, …, i+C).
    all_b_indices: Tuple[Tensor, ...] = tuple(
        flat_channel + C for _ in range(order)
    )  # FIXED: removed extra comma
    D_flat[idx_all, *all_b_indices] = flat_deriv_b

    # (B) For the case when exactly one derivative is taken with respect to a.
    for k in range(order):
        idx_list: list[Tensor] = []
        for j in range(order):
            if j == k:
                idx_list.append(flat_channel)
            else:
                idx_list.append(flat_channel + C)
        idx_tuple: Tuple[Tensor, ...] = tuple(idx_list)
        value: Tensor = flat_deriv_a if (order == 1 and k == 0) else flat_deriv_ab
        D_flat[idx_all, *idx_tuple] = value

    # If we performed a permutation at the start, one might consider undoing it on the batch part.
    # In our framework (mirroring the softmax derivative code) the dual axes are kept appended.
    # Thus we simply return D.
    return D
