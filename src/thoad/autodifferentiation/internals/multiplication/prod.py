# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


def zero_repeated_indices(
    tensor: Tensor, inplace: bool = True, ignore_dims: Tuple[int, ...] = ()
) -> Tensor:
    """
    Zero out all entries in `tensor` where at least two of the non-ignored
    dimension indices are the same.

    Parameters
    ----------
    tensor : Tensor
        The input tensor. It can have any shape (d0, d1, ..., d_{n-1}).
    inplace : bool
        If True, zero out entries in-place. Otherwise, return a copy.
    ignore_dims : Tuple[int, ...]
        Dimensions to ignore in the repeated-indices check. That is, these
        dimensions will not participate in comparisons for 'repeated indices'.

        - Ignored dimensions can have any size.
        - Non-ignored dimensions must all have the same size in order for
          the concept of "repeated index" across dims to make sense.

    Returns
    -------
    Tensor
        The same tensor (in-place) or a new tensor with repeated-index
        (among non-ignored dims) entries zeroed out.

    Example
    -------
    >>> # Suppose we have shape [3, 3, 4], and we want to ignore dim=2:
    >>> # Then we only check dims 0 and 1 for repeated indices.
    >>> t = torch.arange(3*3*4).view(3,3,4)
    >>> _ = zero_repeated_indices(t, inplace=True, ignore_dims=(2,))
    >>> # This zeroes out t[i, j, *] if i == j. Dim 2 is ignored completely.
    """
    n_dims: int = tensor.dim()

    # 1) Gather the non-ignored dimensions
    dims_to_check: list[int] = [d for d in range(n_dims) if d not in ignore_dims]

    # If we have fewer than 2 dimensions to check, there can be no "pair" of
    # equal indices among non-ignored dims => no zeroing needed.
    if len(dims_to_check) < 2:
        return tensor if inplace else tensor.clone()

    # 2) Check that all non-ignored dimensions have the same size
    #    (The original repeated-index logic assumes square shape for dims_to_check)
    target_size = tensor.shape[dims_to_check[0]]
    for d in dims_to_check[1:]:
        if tensor.shape[d] != target_size:
            raise ValueError(
                f"All non-ignored dimensions must have the same size, but "
                f"dim {dims_to_check[0]} has size {target_size} "
                f"while dim {d} has size {tensor.shape[d]}."
            )

    # 3) Clone if not inplace
    if not inplace:
        tensor = tensor.clone()

    # 4) Create coordinate grids only for the non-ignored dimensions,
    #    broadcasting each to the full shape of `tensor`.
    #    We do this by expanding a [1,1,...,size_d,...,1] shape to full shape.
    all_shape: torch.Size = tensor.shape
    coord_list = []
    for d in dims_to_check:
        rng: Tensor = torch.arange(all_shape[d], device=tensor.device)

        # Build a shape that places 'all_shape[d]' in the d-th position,
        # and 1 in all other positions, so we can broadcast:
        view_shape: list[int] = [1] * n_dims
        view_shape[d] = all_shape[d]

        # Expand to the full shape
        coord_d: Tensor = rng.view(view_shape).expand(all_shape)
        coord_list.append(coord_d)

    # 5) Stack these coords -> shape: (k, d0, d1, ..., d_{n-1}),
    #    where k = len(dims_to_check).
    stacked_coords: Tensor = torch.stack(coord_list, dim=0)

    # 6) Sort along dim=0 to group repeated dimension indices
    sorted_coords: Tensor
    sorted_coords, _ = torch.sort(stacked_coords, dim=0)

    # 7) Check any adjacent pairs in the sorted list
    #    if sorted_coords[i, ...] == sorted_coords[i+1, ...] for any i
    #    => repeated index => mask = True
    repeated_mask: Tensor = (sorted_coords[1:] == sorted_coords[:-1]).any(dim=0)

    # 8) Zero out wherever the repeated_mask is True
    tensor[repeated_mask] = 0

    return tensor


def prod0_derivate(tensor: torch.Tensor, order: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns the N-th internal differential of prod(x):
      - shape: (1, d**order), where d = x.numel()
      - entry [0, i1*d^(N-1) + ... + iN] =
        prod(x) / (x[i1] * ... * x[iN]) if all i's distinct, else 0.
    """
    # 1) flatten & safe-invert
    x_flat: Tensor = tensor.flatten() + eps  # shape (d,)
    d: int = x_flat.numel()
    f: Tensor = x_flat.prod()  # scalar prod(x)
    inv: Tensor = 1.0 / x_flat  # shape (d,)

    # 2) broadcast-outer inv with itself N times
    #    build a list of views like inv.view([1]*k + [d] + [1]*(N-k-1))
    views = []
    for k in range(order):
        shape: list[int] = [1] * order
        shape[k] = d
        views.append(inv.view(shape))
    # multiply them all together: result shape (d,d,...,d)
    inv_outer: Tensor = torch.ones(
        [d] * order, device=tensor.device, dtype=tensor.dtype
    )
    for v in views:
        inv_outer = inv_outer * v

    # 3) zero out repeats, multiply by prod, and reshape
    inv_outer = zero_repeated_indices(inv_outer)  # uses your fn above
    deriv_tensor: Tensor = f * inv_outer  # still shape (d,...,d)
    return deriv_tensor.view((1,) + (d,) * order)


class ProdXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        projected_shape: Shape = (1,)
        projected_indep: Indep = (None,)
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_result: Tensor = self._grad_fn._saved_result
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_result = saved_result.to(dtype=self._dtype, device=self._device)
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_result"] = saved_result
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_result: Tensor = self._context["saved_result"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input"] = saved_self
        processed_context["output"] = saved_result
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context

        input: Tuple[int, ...] = self._processed_context["input"]
        output: Tuple[int, ...] = self._processed_context["output"]

        ### Carry out instrumental operations
        input_shape: Tuple[int, ...] = tuple(input.shape)
        input_size: int = math.prod(input_shape)
        ndim: int = len(input_shape)
        # calculate differential shape and indices
        external_indices: list[int] = [0]
        internal_indices: list[int] = list(range(1 + ndim * order))
        composed_indices: list[list[int]] = list()
        for o in range(order):
            ndims: int = o * ndim
            composed_indices.append([*(range(1 + ndims, 1 + ndims + ndim))])
        quotient: Tensor = output / input

        ### Instantiate differential
        differential: Tensor
        internal_shape: list[int] = [1, *(input_shape * order)]
        if order == 1:
            quotient = quotient.view(size=tuple(internal_shape))
            differential = quotient
        else:
            quotient = quotient.view(size=(1, input_size))
            quotient = prod0_derivate(tensor=quotient, order=order)
            quotient = quotient.view(size=tuple(internal_shape))
            differential = quotient

        ### Create einstein notation
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = composed_indices
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([internal_shape])

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
