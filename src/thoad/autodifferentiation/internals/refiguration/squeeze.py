# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import math
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class SqueezeXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        assert math.prod(shape) == math.prod(saved_self_sym_sizes)
        assert shape == tuple(sz for sz in saved_self_sym_sizes if sz != 1)
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self_sym_sizes = self._context["saved_self_sym_sizes"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = saved_self_sym_sizes
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        squeezed_indices: list[int]
        squeezed_indices = [i for i, sz in enumerate(input_shape) if sz == 1]
        external_indices: list[int]
        external_indices = [i for i in composed_indices if i not in squeezed_indices]
        # create dummy tensor and calculate shape for internal differential
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        internal_shape: Tuple[int, ...] = (1,) * len(squeezed_indices)

        ### Instantiate differential
        differential: Tensor = t1.view(size=internal_shape)

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = squeezed_indices
        einstein_composed: list[list[int]] = [composed_indices]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list(internal_shape)])

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
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)


class SqueezeXBackward1(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim: int = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        assert math.prod(shape) == math.prod(saved_self_sym_sizes)
        squeezed_self_sym_sizes: list[int] = list(saved_self_sym_sizes)
        if squeezed_self_sym_sizes[saved_dim] == 1:
            squeezed_self_sym_sizes.pop(saved_dim)
        assert shape == tuple(squeezed_self_sym_sizes)
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, assert already covers disalignment in dims
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: Tuple[int] = self._grad_fn._saved_dim
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim: Tuple[int] = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        reduce_dim: bool = saved_self_sym_sizes[saved_dim] == 1
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = saved_dim
        processed_context["input_shape"] = saved_self_sym_sizes
        processed_context["reduce_dim"] = reduce_dim
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dim: int = self._processed_context["dim"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]
        reduce_dim: bool = self._processed_context["reduce_dim"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        dims: Tuple[int, ...] = [dim] if reduce_dim else list()
        external_indices: list[int] = [d for d in composed_indices if d not in dims]
        # create dummy tensor and calculate shape for internal differential
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor
        internal_indices: Tuple[int, ...]
        internal_shape: Tuple[int, ...]
        if reduce_dim:
            internal_indices = [dim]
            internal_shape = (1,)
            differential = t1.view(size=internal_shape)
        else:
            internal_indices = list()
            internal_shape = list()
            differential = t1.sum()

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = [composed_indices]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list(internal_shape)])

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
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)


class SqueezeXBackward2(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim: Tuple[int, ...] = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        assert math.prod(shape) == math.prod(saved_self_sym_sizes)
        squeezed_self_sym_sizes: list[int] = list(saved_self_sym_sizes)
        for dim in sorted(saved_dim)[::-1]:
            if squeezed_self_sym_sizes[dim] == 1:
                squeezed_self_sym_sizes.pop(dim)
        assert shape == tuple(squeezed_self_sym_sizes)
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, assert already covers disalignment in dims
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = self._grad_fn._saved_dim
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim: Tuple[int, ...] = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dims: Tuple[int, ...]
        dims = tuple([d for d in saved_dim if saved_self_sym_sizes[d] == 1])
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = dims
        processed_context["input_shape"] = saved_self_sym_sizes
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dims: Tuple[int, ...] = self._processed_context["dims"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        squeezed_indices: list[int] = list(dims)
        external_indices: list[int]
        external_indices = [i for i in composed_indices if i not in squeezed_indices]
        # create dummy tensor and calculate shape for internal differential
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        internal_shape: Tuple[int, ...] = (1,) * len(squeezed_indices)

        ### Instantiate differential
        differential: Tensor = t1.view(size=internal_shape)

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = squeezed_indices
        einstein_composed: list[list[int]] = [composed_indices]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list(internal_shape)])

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
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
