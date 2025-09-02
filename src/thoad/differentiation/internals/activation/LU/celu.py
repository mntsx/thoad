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


class CeluXBackward0(ContractiveFunction):

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
        input: Tensor = self._processed_context["input"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(input.shape)
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
        saved_alpha: float = float(getattr(self._grad_fn, "_saved_alpha"))
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_alpha"] = saved_alpha
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_alpha: float = self._context["saved_alpha"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        condition: Tensor = input > 0
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["alpha"] = saved_alpha
        processed_context["condition"] = condition
        processed_context["input"] = input
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        alpha: float = self._processed_context["alpha"]
        condition: Tensor = self._processed_context["condition"]
        input: Tensor = self._processed_context["input"]

        ### Carry out instrumental operations
        # create dummy tensors
        t0: Tensor = torch.zeros((1,), dtype=self._dtype, device=self._device)
        t1: Tensor = torch.ones((1,), dtype=self._dtype, device=self._device)
        ta: Tensor = torch.full(
            fill_value=alpha,
            size=(1,),
            dtype=self._dtype,
            device=self._device,
        )
        # compute exponential term
        exp_term: Tensor = torch.exp(input / ta)

        ### Instantiate derivative
        derivative: Tensor
        if order == 1:
            derivative = torch.where(
                condition=condition,
                input=t1,
                other=exp_term,
            )
        else:
            factor: Tensor = (1 / ta) ** (order - 1)
            derivative = torch.where(
                condition=condition,
                input=t0,
                other=(factor * exp_term),
            )

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = tuple(tuple(range(ndim)) for _ in range(order))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(True for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        ID_data: IDData
        if out_id == 0 and all(i == 0 for i in inp_id):
            order: int = len(inp_id)
            ID_data = self._compute_internal_0(order=order)
        else:
            ID_data = (None, None)
        return ID_data
