# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class TXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        assert len(shape) >= 2
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no useful context info for projection
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        #   no context info
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        # ...
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        # ...
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        # ...
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        # ...

        ### Carry out instrumental operations
        ndim: int = len(self._shape)
        dim0: int = ndim - 1
        dim1: int = ndim - 1
        # create composed einsum indices
        composed_range: list[int] = list(range(ndim))
        composed_range[dim0] = dim1
        composed_range[dim1] = dim0
        # construct dummy tensor to avoid None tensor errors
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list()
        einstein_composed: list[list[int]] = [composed_range]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list()])

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


class TransposeXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim0: int = self._context["saved_dim0"]
        saved_dim1: int = self._context["saved_dim1"]
        assert len(shape) >= (max(saved_dim0, saved_dim1) + 1)
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no useful context info for projection
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim0: int = self._grad_fn._saved_dim0
        saved_dim1: int = self._grad_fn._saved_dim1
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim0"] = saved_dim0
        context["saved_dim1"] = saved_dim1
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim0: int = self._context["saved_dim0"]
        saved_dim1: int = self._context["saved_dim1"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim0"] = saved_dim0
        processed_context["dim1"] = saved_dim1
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dim0: int = self._processed_context["dim0"]
        dim1: int = self._processed_context["dim1"]

        ### Carry out instrumental operations
        ndim: int = len(self._shape)
        # create composed einsum indices
        composed_range: list[int] = list(range(ndim))
        composed_range[dim0] = dim1
        composed_range[dim1] = dim0
        # construct dummy tensor to avoid None tensor errors
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list()
        einstein_composed: list[list[int]] = [composed_range]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list()])

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
