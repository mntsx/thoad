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


class ReluXBackward0(ContractiveFunction):

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
        condition: Tensor = self._processed_context["condition"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(condition.shape)
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_result: Tensor = getattr(self._grad_fn, "_saved_result")
        # ensure proper tensor configuration
        saved_result = saved_result.to(dtype=self._dtype, device=self._device)
        getattr(saved_result, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_result"] = saved_result
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_result: Tensor = self._context["saved_result"]
        # process context
        output: Tensor = denull_tensor(
            tensor=saved_result, dtype=self._dtype, device=self._device
        )
        condition: Tensor = output > 0
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["condition"] = condition
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        condition: Tensor = self._processed_context["condition"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.where(condition=condition, input=t1, other=t0)

        ### Create einstein notation
        ndim: int = condition.ndim
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
