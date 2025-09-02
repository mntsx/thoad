# Standard Library Dependencies
import math
import operator
from typing import Any, Tuple, SupportsIndex

# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


def _decode_twos_complement(x: SupportsIndex) -> int:
    """Coerce to int and reinterpret common unsigned encodings as signed."""
    i: int = operator.index(x)
    if (1 << 63) <= i <= (1 << 64) - 1:
        return i - (1 << 64)  # uint64 -> signed
    if (1 << 31) <= i <= (1 << 32) - 1:
        return i - (1 << 32)  # uint32 -> signed
    return i


def adjust_slice_bounds(
    start: SupportsIndex,
    end: SupportsIndex,
    step: SupportsIndex,
    dim: SupportsIndex,
    input_shape: Tuple[int, ...],
    output_size: int,  # used to disambiguate sentinel bounds
) -> Tuple[int, int, int]:
    """
    Normalize (start, end, step) for slicing along `dim` and, if needed,
    re-derive `end` so that len(range(start, end, step)) == output_size.

    Handles negatives, out-of-range, uint encodings (uint64/uint32), and
    ATen CPU sentinels (e.g., end == -1 standing for 'None').
    """
    d: int = _decode_twos_complement(dim)
    if not (0 <= d < len(input_shape)):
        raise IndexError(f"dim {d} out of range for rank {len(input_shape)}")

    dim_size: int = input_shape[d]
    if dim_size < 0:
        raise ValueError(f"invalid dim size {dim_size}")

    s: int = _decode_twos_complement(start)
    e: int = _decode_twos_complement(end)
    st: int = _decode_twos_complement(step)
    if st == 0:
        raise ValueError("slice step cannot be zero")

    # Empty dimension → canonical empty slice with sign-correct step
    if dim_size == 0:
        st = 1 if st > 0 else -1
        return (0, 0, st)

    # CPython-style normalization
    ns: int
    ne: int
    nst: int
    ns, ne, nst = slice(s, e, st).indices(dim_size)

    # If the normalized length doesn't match what the outer shape demands,
    # recompute `end` to produce exactly `output_size` elements.
    # This fixes cases where ATen's sentinel (-1) mapped 'full slice' to (0, 9, 1).
    try:
        L = len(range(ns, ne, nst))
    except ValueError:
        # Defensive (shouldn't happen after indices())
        L = 0

    if output_size >= 0 and L != output_size:
        # Candidate end that yields exactly `output_size` items:
        # elements are ns, ns+nst, ..., ns+nst*(output_size-1); range excludes end
        ne_candidate: int = ns + nst * output_size

        # Clamp candidate to valid exclusive-end domain for the given step sign
        ne_fixed: int
        if nst > 0:
            # Valid exclusive end is in [0, dim_size]
            ne_fixed = min(max(ne_candidate, 0), dim_size)
        else:
            # For negative step, valid exclusive end is in [-1, dim_size-1]
            ne_fixed = max(min(ne_candidate, dim_size - 1), -1)

        if len(range(ns, ne_fixed, nst)) == output_size:
            ne = ne_fixed
        # If it *still* doesn't match, we leave (ns, ne, nst) as-is; a later
        # assert will catch the inconsistency.

    return (ns, ne, nst)


class SliceXBackward0(ContractiveFunction):

    schwarz: bool = True

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert self._processed_context is not None
        assert out_id == 0
        assert inp_id == 0

        start: int = self._processed_context["start"]
        end: int = self._processed_context["end"]
        step: int = self._processed_context["step"]
        dim: int = self._processed_context["dim"]
        input_shape: Shape = self._processed_context["input_shape"]

        bounded_input_shape: Shape = tuple(max(1, sz) for sz in input_shape)
        pruned_output_shape: Shape = shape[(len(shape) - len(input_shape)) :]
        bounded_output_shape: Shape = tuple(max(1, sz) for sz in pruned_output_shape)

        assert all(
            bounded_output_shape[d] == sz
            for d, sz in enumerate(bounded_input_shape)
            if d != dim
        )

        output_size: int = math.prod(bounded_output_shape)
        output_size //= math.prod(bounded_input_shape) // bounded_input_shape[dim]
        assert int(output_size) != 0 and output_size / int(output_size) == 1
        output_size = int(output_size)

        # A) Normalize bounds AND STEP, and store them back
        ns, ne, nst = adjust_slice_bounds(
            start=start,
            end=end,
            step=step,
            dim=dim,
            input_shape=input_shape,
            output_size=output_size,
        )
        self._processed_context["start"] = ns
        self._processed_context["end"] = ne
        self._processed_context["step"] = nst

        projected_shape: Shape = tuple(
            output_size if d == dim else sz for d, sz in enumerate(bounded_input_shape)
        )
        projected_indep: Indep = indep

        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )

        self._shape0 = projected_shape
        projected_indep = tuple(d if d != dim else None for d in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        saved_end: int = getattr(self._grad_fn, "_saved_end")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        saved_start: int = getattr(self._grad_fn, "_saved_start")
        saved_step: int = getattr(self._grad_fn, "_saved_step")

        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_end"] = saved_end
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        context["saved_start"] = saved_start
        context["saved_step"] = saved_step
        self._context = context

        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        saved_dim: int = self._context["saved_dim"]
        saved_end: int = self._context["saved_end"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        saved_start: int = self._context["saved_start"]
        saved_step: int = self._context["saved_step"]

        dim: int = recalibrate_dim(dim=saved_dim, shape=saved_self_sym_sizes)

        processed_context: dict[str, Any] = dict()
        processed_context["start"] = saved_start
        processed_context["end"] = saved_end
        processed_context["step"] = saved_step
        processed_context["dim"] = dim
        processed_context["input_shape"] = saved_self_sym_sizes
        self._processed_context = processed_context
        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        start: int = self._processed_context["start"]
        end: int = self._processed_context["end"]
        step: int = self._processed_context["step"]
        dim: int = self._processed_context["dim"]
        input_shape: Shape = self._processed_context["input_shape"]

        ndim: int = len(input_shape)
        N_in: int = input_shape[dim]

        external_indices: list[int] = list(range(ndim))
        internal_indices: list[int] = [dim, ndim]
        internal_independencies: list[bool] = [False, True]
        composed_indices: list[int] = [
            ndim if d == dim else d for d in external_indices
        ]

        # Always produce a 2-D derivative (no scalar path)
        L: int = max(0, len(range(start, end, step)))
        internal_shape: list[int] = [L, max(0, N_in)]
        derivative: Tensor
        if L == 0:
            derivative = torch.zeros(
                (0, max(0, N_in)), device=self._device, dtype=self._dtype
            )
        else:
            idx: Tensor = torch.arange(start, end, step, device=self._device)
            derivative = F.one_hot(idx, num_classes=max(0, N_in)).to(self._dtype)

        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(internal_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = []
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()
        return ID_data
