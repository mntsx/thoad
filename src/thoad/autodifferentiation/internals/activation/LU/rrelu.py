# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class RreluWithNoiseXBackward0(ContractiveFunction):

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
        # extract info
        saved_training: bool = self._grad_fn._saved_training
        saved_lower: float = self._grad_fn._saved_lower
        saved_upper: float = self._grad_fn._saved_upper
        saved_noise: Tensor = self._grad_fn._saved_noise
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_noise = saved_noise.to(dtype=self._dtype, device=self._device)
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_training"] = saved_training
        context["saved_lower"] = saved_lower
        context["saved_upper"] = saved_upper
        context["saved_noise"] = saved_noise
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_training: bool = self._context["saved_training"]
        saved_lower: float = self._context["saved_lower"]
        saved_upper: float = self._context["saved_upper"]
        saved_noise: Tensor = self._context["saved_noise"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        condition: Tensor = saved_self > 0
        # save processed context
        processed_context: dict["str", Any] = dict()
        processed_context["training"] = saved_training
        processed_context["lower"] = saved_lower
        processed_context["upper"] = saved_upper
        processed_context["noise"] = saved_noise
        processed_context["input"] = saved_self
        processed_context["condition"] = condition
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        ### Gather context
        training: bool = self._processed_context["training"]
        lower: float = self._processed_context["lower"]
        upper: float = self._processed_context["upper"]
        noise: Tensor = self._processed_context["noise"]
        input: Tensor = self._processed_context["input"]
        condition: Tensor = self._processed_context["condition"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        slope: float = (lower + upper) / 2.0
        ts: Tensor = torch.tensor([slope], dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor
        if training:
            differential: Tensor = torch.where(condition, t1, noise)
        else:
            differential: Tensor = torch.where(condition, t1, ts)

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim))]
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
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
