# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class LeakyReluXBackward0(ContractiveFunction):

    def check_shape(
        self,
        shape: Shape,
        indep: Indep,
    ) -> Tuple[Shape, Indep]:
        # extract saved input
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
        # adjust indep if broadcasted
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
        # extract info
        saved_self: Tensor = self._grad_fn._saved_self
        saved_neg: float = self._grad_fn._saved_negative_slope
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_neg"] = saved_neg
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_neg: float = self._context["saved_neg"]
        # process context
        condition: Tensor = saved_self > 0
        slope_tensor: Tensor = torch.tensor(
            [saved_neg],
            dtype=self._dtype,
            device=self._device,
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["condition"] = condition
        processed_context["slope"] = slope_tensor
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        ### Gather context
        condition: Tensor = self._processed_context["condition"]
        slope_tensor: Tensor = self._processed_context["slope"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = torch.where(
            condition=condition,
            input=t1,
            other=slope_tensor,
        )

        ### Create einstein notation
        einstein_external: list[int] = list(range(condition.ndim))
        einstein_internal: list[int] = list(range(condition.ndim))
        einstein_composed: list[list[int]] = [list(range(condition.ndim))]
        einstein_notation: Notation = []
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
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
