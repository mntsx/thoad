# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class SumXBackward0(ContractiveFunction):

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
        saved_self_sym_sizes: Tensor = self._grad_fn._saved_self_sym_sizes
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
        saved_self_sym_sizes: Tensor = self._context["saved_self_sym_sizes"]
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
        input_shape: Tuple[int, ...]
        input_shape = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate differential shape and indices
        internal_shape: Tuple[int, ...] = (1, *input_shape)
        external_indices: list[int] = [0]
        internal_indices: list[int] = list(range(1 + ndim))
        composed_indices: list[int] = list(range(1, 1 + ndim))

        ### Instantiate differential
        differential: Tensor = torch.ones(
            size=internal_shape, dtype=self._dtype, device=self._device
        )

        ### Create einstein notation
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = [composed_indices]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([(1, *input_shape)])

        # assert False, (differential.shape, einstein_notation)

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


class SumXBackward1(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim: Tuple[int, ...] = self._context["saved_dim"]
        saved_keepdim: Tuple[int, ...] = self._context["saved_keepdim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # reduce input shape according to saved dims
        squeezed_self_sym_sizes: list[int] = list(saved_self_sym_sizes)
        adjusted_indep: list[Union[None, int]] = list()
        for dim in sorted(saved_dim)[::-1]:
            if saved_keepdim:
                squeezed_self_sym_sizes[dim] = 1
                if dim in indep:
                    adjusted_indep[indep.index(dim)] = None
            else:
                squeezed_self_sym_sizes.pop(dim)
        projected_shape: Tuple[int, ...] = tuple(squeezed_self_sym_sizes)
        projected_indep: Tuple[Union[None, int]] = indep
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=tuple(saved_self_sym_sizes),
            )
        # adjust projected indep
        if saved_keepdim:
            aux: list[Union[None, int]] = list(projected_indep)
            for dim in sorted(saved_dim)[::-1]:
                if dim in projected_indep:
                    aux[indep.index(dim)] = None
            projected_indep = tuple(aux)
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: Tuple[int, ...] = self._grad_fn._saved_dim
        saved_keepdim: bool = self._grad_fn._saved_keepdim
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_keepdim"] = saved_keepdim
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim: Tuple[int, ...] = self._context["saved_dim"]
        saved_keepdim: Tuple[int, ...] = self._context["saved_keepdim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = saved_dim
        processed_context["keepdim"] = saved_keepdim
        processed_context["input_shape"] = saved_self_sym_sizes
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dims: Tuple[int, ...] = self._processed_context["dims"]
        keepdim: Tuple[int, ...] = self._processed_context["keepdim"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        # obtain reduced sizes
        internal_shape: list[int] = tuple(input_shape[d] for d in dims)
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

        ### Instantiate differential
        differential: Tensor = torch.ones(
            size=internal_shape,
            dtype=self._dtype,
            device=self._device,
        )

        ### Create einstein notation
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
