# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class PermuteXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dims: Tuple[int, ...] = self._context["saved_dims"]
        assert len(shape) == len(saved_dims)
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   no modification of shape -> unnecesary
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dims: Tuple[int, ...] = self._grad_fn._saved_dims
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dims"] = saved_dims
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dims: Tuple[int, ...] = self._context["saved_dims"]
        # process context
        dims: Tuple[int, ...] = saved_dims
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dims"] = dims
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dims: Tuple[int, ...] = self._processed_context["dims"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1.sum()

        ### Create einstein notation
        ndim: int = len(dims)
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list()
        einstein_composed: list[list[int]] = [dims]
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
