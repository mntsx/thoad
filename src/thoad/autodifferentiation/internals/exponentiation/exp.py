# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class ExpXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        raise NotImplementedError()
        assert self._context is not None
        # ...
        projected_shape: Shape = shape
        projected_indep: Indep = indep
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
        raise NotImplementedError()
        # extract info
        # ...
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        # ...
        self._context = context

        return None

    def _process_context(self) -> None:
        raise NotImplementedError()
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

        ### Instantiate differential
        differential: Tensor
        differential = None

        ### Create einstein notation
        ndim: int = None
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim))]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

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
                raise NotImplementedError()
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
