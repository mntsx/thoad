# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData
from thoad.autodifferentiation.internals.mathematic.sigmoid import sigmoid_derivate


class SiluXBackward0(ContractiveFunction):

    def check_shape(
        self,
        shape: Shape,
        indep: Indep,
    ) -> Tuple[Shape, Indep]:
        # extract input shape
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
        projected_indep: Indep = indep
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # save input
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict["str", Any] = dict()
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        # process context
        sigmoid_input: Tensor = torch.sigmoid(input=saved_self)
        # save processed context
        processed_context: dict["str", Any] = dict()
        processed_context["input"] = saved_self
        processed_context["sigmoid_input"] = sigmoid_input
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        ### Gather context
        input: Tensor = self._processed_context["input"]
        sigmoid_input: Tensor = self._processed_context["sigmoid_input"]

        ## Instantiate internal differential derivative via product rule
        differential: Tensor = input * sigmoid_derivate(
            tensor=sigmoid_input,
            order=(order),
        )
        differential += order * sigmoid_derivate(
            tensor=sigmoid_input, order=(order - 1)
        )

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim)) for _ in range(order)]
        einstein_notation: Notation = []
        einstein_notation.append(
            [
                einstein_external,
                einstein_internal,
            ]
        )
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
        if out_id == 0 and all(i == 0 for i in in_id):
            order: int = len(in_id)
            (differential, einstein_notation) = self._compute_internal_0(order=order)
        else:
            (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
