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


class LeakyReluXBackward0(ContractiveFunction):

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
        # extract saved condition
        condition: Tensor = self._processed_context["condition"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(condition.shape)
        # adjust indep if broadcasted
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
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_negative_slope: float = getattr(self._grad_fn, "_saved_negative_slope")
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_negative_slope"] = saved_negative_slope
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_neg: float = self._context["saved_negative_slope"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        condition: Tensor = input > 0
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

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        condition: Tensor = self._processed_context["condition"]
        slope_tensor: Tensor = self._processed_context["slope"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=t1,
            other=slope_tensor,
        )

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(condition.ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(condition.ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(condition.ndim)),)
        einstein_notation: Notation = []
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
