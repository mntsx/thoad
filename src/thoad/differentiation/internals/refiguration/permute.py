# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class PermuteXBackward0(ContractiveFunction):

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
        assert len(shape) == len(dims)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   no modification of shape -> unnecesary
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dims: Tuple[int, ...] = getattr(self._grad_fn, "_saved_dims")
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dims"] = saved_dims
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_dims: Tuple[int, ...] = self._context["saved_dims"]
        # process context
        dims: Tuple[int, ...] = tuple(
            recalibrate_dim(dim=dim, shape=saved_dims) for dim in saved_dims
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = dims
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dims: Tuple[int, ...] = self._processed_context["dims"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1.sum()

        ### Create einstein notation
        ndim: int = len(dims)
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple()
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(dims),)
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
