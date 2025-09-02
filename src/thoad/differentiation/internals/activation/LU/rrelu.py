# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape, Indep, Notation, IDData


class RreluWithNoiseXBackward0(ContractiveFunction):

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
        # extract input shape
        input: Tensor = self._processed_context["input"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(input.shape)
        projected_indep: Indep = indep
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_training: bool = getattr(self._grad_fn, "_saved_training")
        saved_lower: float = getattr(self._grad_fn, "_saved_lower")
        saved_upper: float = getattr(self._grad_fn, "_saved_upper")
        saved_noise: Tensor = getattr(self._grad_fn, "_saved_noise")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        saved_noise = saved_noise.to(dtype=self._dtype, device=self._device)
        getattr(saved_noise, "_fix_weakref")()
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_training"] = saved_training
        context["saved_lower"] = saved_lower
        context["saved_upper"] = saved_upper
        context["saved_noise"] = saved_noise
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        # load context
        saved_training: bool = self._context["saved_training"]
        saved_lower: float = self._context["saved_lower"]
        saved_upper: float = self._context["saved_upper"]
        saved_noise: Tensor = self._context["saved_noise"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        noise: Tensor = denull_tensor(
            tensor=saved_noise, dtype=self._dtype, device=self._device
        )
        condition: Tensor = saved_self > 0
        # save processed context
        processed_context: dict["str", Any] = dict()
        processed_context["training"] = saved_training
        processed_context["lower"] = saved_lower
        processed_context["upper"] = saved_upper
        processed_context["input"] = input
        processed_context["noise"] = noise
        processed_context["condition"] = condition
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        training: bool = self._processed_context["training"]
        lower: float = self._processed_context["lower"]
        upper: float = self._processed_context["upper"]
        input: Tensor = self._processed_context["input"]
        noise: Tensor = self._processed_context["noise"]
        condition: Tensor = self._processed_context["condition"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        slope: float = (lower + upper) / 2.0
        ts: Tensor = torch.tensor([slope], dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor
        if training:
            derivative: Tensor = torch.where(condition, t1, noise)
        else:
            derivative: Tensor = torch.where(condition, t1, ts)

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = []
        einstein_notation.append(
            (
                einstein_external,
                einstein_internal,
            )
        )
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
