# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_shape
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class SqueezeXBackward0(ContractiveFunction):

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
        input_shape: Shape = self._processed_context["input_shape"]
        assert math.prod(shape) == math.prod(input_shape)
        assert shape == tuple(sz for sz in input_shape if sz != 1)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self_sym_sizes = self._context["saved_self_sym_sizes"]
        # process context
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        input_shape: Shape = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        squeezed_indices: list[int]
        squeezed_indices = [i for i, sz in enumerate(input_shape) if sz == 1]
        external_indices: list[int]
        external_indices = [i for i in composed_indices if i not in squeezed_indices]
        # create dummy tensor and calculate shape for internal derivative
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        internal_shape: Tuple[int, ...] = (1,) * len(squeezed_indices)

        ### Instantiate derivative
        derivative: Tensor = t1.view(size=internal_shape)

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(squeezed_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(True for _ in internal_shape))
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


class SqueezeXBackward1(ContractiveFunction):

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
        dim: int = self._processed_context["dim"]
        input_shape: Shape = self._processed_context["input_shape"]
        assert math.prod(shape) == math.prod(input_shape)
        squeezed_self_sym_sizes: list[int] = list(input_shape)
        if squeezed_self_sym_sizes[dim] == 1:
            squeezed_self_sym_sizes.pop(dim)
        assert shape == tuple(squeezed_self_sym_sizes)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, assert already covers disalignment in dims
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_dim: int = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dim: int = recalibrate_dim(dim=saved_dim, shape=saved_self_sym_sizes)
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        reduce_dim: bool = saved_self_sym_sizes[dim] == 1
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = dim
        processed_context["input_shape"] = input_shape
        processed_context["reduce_dim"] = reduce_dim
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dim: int = self._processed_context["dim"]
        input_shape: Shape = self._processed_context["input_shape"]
        reduce_dim: bool = self._processed_context["reduce_dim"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        dims: list[int] = [dim] if reduce_dim else list()
        external_indices: list[int] = [d for d in composed_indices if d not in dims]
        # create dummy tensor and calculate shape for internal derivative
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor
        internal_indices: Tuple[int, ...]
        internal_shape: Tuple[int, ...]
        if reduce_dim:
            internal_indices = (dim,)
            internal_shape = (1,)
            derivative = t1.view(size=internal_shape)
        else:
            internal_indices = tuple()
            internal_shape = tuple()
            derivative = t1.sum()

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(internal_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(True for _ in internal_shape))
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


class SqueezeXBackward2(ContractiveFunction):

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
        dims: Tuple[int, ...] = self._processed_context["dims"]
        input_shape: Shape = self._processed_context["input_shape"]
        assert math.prod(shape) == math.prod(input_shape)
        squeezed_self_sym_sizes: list[int] = list(input_shape)
        for dim in sorted(dims)[::-1]:
            if squeezed_self_sym_sizes[dim] == 1:
                squeezed_self_sym_sizes.pop(dim)
        assert shape == tuple(squeezed_self_sym_sizes)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, assert already covers disalignment in dims
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_dim: Tuple[int, ...] = self._context["saved_dim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dims: Tuple[int, ...] = tuple(
            recalibrate_dim(dim=dim, shape=saved_self_sym_sizes) for dim in saved_dim
        )
        dims = tuple([d for d in dims if saved_self_sym_sizes[d] == 1])
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = dims
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dims: Tuple[int, ...] = self._processed_context["dims"]
        input_shape: Shape = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        composed_indices: list[int] = list(range(ndim))
        squeezed_indices: list[int] = list(dims)
        external_indices: list[int]
        external_indices = [i for i in composed_indices if i not in squeezed_indices]
        # create dummy tensor and calculate shape for internal derivative
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        internal_shape: Tuple[int, ...] = (1,) * len(squeezed_indices)

        ### Instantiate derivative
        derivative: Tensor = t1.view(size=internal_shape)

        ### Create einstein notation
        # note. torch.einsum allows to remove indices associated to dims of size 1
        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(squeezed_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(True for _ in internal_shape))
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
