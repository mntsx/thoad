# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class AddXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_alpha: float = self._grad_fn._saved_alpha
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_alpha"] = saved_alpha
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_alpha: float = self._context["saved_alpha"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["alpha"] = saved_alpha
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        # ...

        ### Carry out instrumental operations
        ndim: int = len(self._shape)
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list()
        einstein_composed: list[list[int]] = [list(range(ndim))]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list()])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> None:
        assert self._processed_context is not None
        ### Gather context
        alpha: float = self._processed_context["alpha"]

        ### Carry out instrumental operations
        ndim: int = len(self._shape)
        # calculate internal shape
        internal_broadcastable_shape: Tuple[int, ...] = (1,) * ndim
        # create dummy tensor
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate tensor differently dependding on the value of alpha
        differential: Tensor
        einstein_external: list[int]
        einstein_internal: list[int]
        einstein_composed: list[list[int]]
        composed_shape: list[int]

        if alpha == 1.0:
            # instantiate differential
            differential = t1.sum()
            # create einstein notation
            einstein_external = list(range(ndim))
            einstein_internal = list()
            einstein_composed = [list(range(ndim))]
            composed_shape = list()
        else:
            # instantiate differential
            differential = torch.full(
                fill_value=alpha,
                size=internal_broadcastable_shape,
                dtype=self._dtype,
                device=self._device,
            )
            # create einstein notation
            einstein_external = list(range(ndim))
            einstein_internal = list(range(ndim))
            einstein_composed = [list(range(ndim))]
            composed_shape = list(self._shape)

        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([composed_shape])

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
            case (0, (1,)):
                (differential, einstein_notation) = self._compute_internal_0_1()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
