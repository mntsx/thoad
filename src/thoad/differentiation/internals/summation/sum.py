# Standard Library Dependencies
import math
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_shape
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class SumXBackward0(ContractiveFunction):

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
        # initialize shape and indep projections
        projected_shape: Shape = tuple()
        projected_indep: Indep = tuple()
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
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
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
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
        input_shape: Tuple[int, ...]
        input_shape = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate derivative shape and indices
        internal_shape: Tuple[int, ...] = input_shape
        external_indices: list[int] = list()
        internal_indices: list[int] = list(range(ndim))
        composed_indices: list[int] = list(range(ndim))

        ### Instantiate derivative
        derivative: Tensor = torch.ones(
            size=internal_shape, dtype=self._dtype, device=self._device
        )

        ### Create einstein notation
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


class SumXBackward1(ContractiveFunction):

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
        keepdim: Tuple[int, ...] = self._processed_context["keepdim"]
        input_shape: Shape = self._processed_context["input_shape"]
        # reduce input shape according to saved dims
        expected_output_shape: list[int] = list(input_shape)
        for dim in sorted(dims)[::-1]:
            if keepdim:
                expected_output_shape[dim] = 1
            else:
                expected_output_shape.pop(dim)
        projected_shape: Tuple[int, ...] = tuple(expected_output_shape)
        projected_indep: Tuple[Union[None, int], ...] = indep
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # adjust projected indep
        if keepdim:
            aux: list[Union[None, int]] = list(projected_indep)
            for dim in sorted(dims)[::-1]:
                if dim in projected_indep:
                    aux[indep.index(dim)] = None
            projected_indep = tuple(projected_indep)
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: Tuple[int, ...] = getattr(self._grad_fn, "_saved_dim")
        saved_keepdim: bool = getattr(self._grad_fn, "_saved_keepdim")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_keepdim"] = saved_keepdim
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
        saved_keepdim: Tuple[int, ...] = self._context["saved_keepdim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dims: Tuple[int, ...] = tuple(
            recalibrate_dim(dim=dim, shape=saved_self_sym_sizes) for dim in saved_dim
        )
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = dims
        processed_context["keepdim"] = saved_keepdim
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dims: Tuple[int, ...] = self._processed_context["dims"]
        keepdim: Tuple[int, ...] = self._processed_context["keepdim"]
        input_shape: Shape = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        reduced_size: int = math.prod(
            sz for d, sz in enumerate(input_shape) if d in dims
        )
        # obtain reduced sizes
        internal_shape: Tuple[int, ...] = tuple(input_shape[d] for d in dims)
        # treat batch dims
        batch_range: Tuple[int, ...]
        batch_range = tuple([d for d, _ in enumerate(input_shape) if d not in dims])
        # create einstein indices
        external_indices: list[int] = list(batch_range)
        internal_indices: list[int] = list()
        composed_indices: list[int] = list(batch_range)
        for dim in dims:
            internal_indices.append(dim)
            composed_indices.insert(dim, dim)
            if keepdim:
                external_indices.insert(dim, dim)

        ### Instantiate derivative
        derivative: Tensor = torch.ones(
            size=internal_shape, dtype=self._dtype, device=self._device
        )

        ### Create einstein notation
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
