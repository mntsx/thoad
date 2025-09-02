# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.typing import Shape, Indep, Notation, IDData


class NegXBackward0(ContractiveFunction):

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
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        # ...
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
        internal_shape: Tuple[int, ...] = (1,) * ndim

        ### Instantiate derivative
        derivative: Tensor = torch.full(
            fill_value=(-1.0),
            size=internal_shape,
            dtype=self._dtype,
            device=self._device,
        )

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(True for _ in self._shape0))
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
