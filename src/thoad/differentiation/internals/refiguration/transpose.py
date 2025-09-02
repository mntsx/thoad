# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class TXBackward0(ContractiveFunction):

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
        assert len(shape) >= 2
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no useful context info for projection
        # save as class attributes
        self._shape0 = projected_shape
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
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
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

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        # ...

        ### Carry out instrumental operations
        ndim: int = len(self._shape0)
        dim0: int = ndim - 1
        dim1: int = ndim - 1
        # create composed einsum indices
        composed_range: list[int] = list(range(ndim))
        composed_range[dim0] = dim1
        composed_range[dim1] = dim0
        # construct dummy tensor to avoid None tensor errors
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple()
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_range),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(), tuple()))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data


class TransposeXBackward0(ContractiveFunction):

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
        dim0: int = self._processed_context["dim0"]
        dim1: int = self._processed_context["dim1"]
        dim0: int = recalibrate_dim(dim=dim0, shape=shape)
        dim1: int = recalibrate_dim(dim=dim1, shape=shape)
        self._processed_context["dim0"] = dim0
        self._processed_context["dim1"] = dim1
        assert len(shape) >= (max(dim0, dim1) + 1)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no useful context info for projection
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim0: int = getattr(self._grad_fn, "_saved_dim0")
        saved_dim1: int = getattr(self._grad_fn, "_saved_dim1")
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim0"] = saved_dim0
        context["saved_dim1"] = saved_dim1
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
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

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dim0: int = self._processed_context["dim0"]
        dim1: int = self._processed_context["dim1"]

        ### Carry out instrumental operations
        ndim: int = len(self._shape0)
        # create composed einsum indices
        composed_range: list[int] = list(range(ndim))
        composed_range[dim0] = dim1
        composed_range[dim1] = dim0
        # construct dummy tensor to avoid None tensor errors
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple()
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_range),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(), tuple()))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data
