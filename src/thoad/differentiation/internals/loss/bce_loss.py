# Standard Library Dependencies
import math
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor, denull_shape
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.differentiation.internals.utils.derivate import log_derivate
from thoad.typing import Shape, Indep, Notation, IDData


class BinaryCrossEntropyXBackward0(ContractiveFunction):

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
        assert inp_id in (0, 1)
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        reduce: bool = self._processed_context["reduce"]
        projected_shape: Shape
        projected_indep: Indep
        if reduce:
            # initialize shape and indep projections
            projected_shape = tuple()
            projected_indep = tuple()
        else:
            # initialize shape and indep projections
            projected_shape = broadcasted_shape
            projected_indep = indep
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
        saved_reduction: int = getattr(self._grad_fn, "_saved_reduction")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_target: Tensor = getattr(self._grad_fn, "_saved_target")
        saved_weight: Union[None, Tensor] = getattr(self._grad_fn, "_saved_weight")
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        saved_target = saved_target.to(dtype=self._dtype, device=self._device)
        getattr(saved_target, "_fix_weakref")()
        if saved_weight is not None:
            saved_weight = saved_weight.to(dtype=self._dtype, device=self._device)
            getattr(saved_weight, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_reduction"] = saved_reduction
        context["saved_self"] = saved_self
        context["saved_target"] = saved_target
        context["saved_weight"] = saved_weight
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_reduction: int = self._context["saved_reduction"]
        saved_self: Tensor = self._context["saved_self"]
        saved_target: Tensor = self._context["saved_target"]
        saved_weight: Union[None, Tensor] = getattr(self._grad_fn, "_saved_weight")
        # process context
        reduce: bool = saved_reduction in (1, 2)
        N: int = max(saved_self.numel(), saved_target.numel())
        numerator: Tensor
        if saved_weight is None:
            numerator = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        else:
            numerator = saved_weight
        constant: Tensor = numerator / ((-N) if saved_reduction == 1 else (-1))
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        target: Tensor = denull_tensor(
            tensor=saved_target, dtype=self._dtype, device=self._device
        )
        tensors_shapes: list[Shape]
        tensors_shapes = [T.shape for T in (input, target) if T is not None]
        broadcasted_shape: Shape = infer_broadcast(shapes=tensors_shapes)
        mod_input: Tensor = 1 - input
        mod_target: Tensor = target - 1
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["reduce"] = reduce
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["constant"] = constant
        processed_context["input"] = input
        processed_context["target"] = target
        processed_context["mod_input"] = mod_input
        processed_context["mod_target"] = mod_target
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order_0: int, order_1: int) -> IDData:
        assert self._processed_context is not None
        ### Checks
        assert order_1 in (0, 1)

        ### Read context
        reduce: bool = self._processed_context["reduce"]
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        constant: Tensor = self._processed_context["constant"]
        input: Tensor = self._processed_context["input"]
        target: Tensor = self._processed_context["target"]
        mod_input: Tensor = self._processed_context["mod_input"]
        mod_target: Tensor = self._processed_context["mod_target"]

        ### Carry out instrumental operations
        # declare info about internal derivative shape & values
        internal_shape: list[int] = list(broadcasted_shape)
        internal_independencies: list[bool] = [True for _ in broadcasted_shape]
        # create auxiliar tensors
        term1: Tensor = log_derivate(tensor=input, order=order_0)
        term2: Tensor = log_derivate(tensor=mod_input, order=order_0)
        term2 *= (-1) ** order_0
        if order_1 == 0:
            term1 *= target
            term2 *= mod_target

        ### Instantiate derivative
        derivative: Tensor = term1.sub_(term2)
        derivative *= constant

        ### Create einstein notation
        ndim: int = len(broadcasted_shape)
        einstein_external: Tuple[int, ...] = tuple() if reduce else tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = tuple(tuple(range(ndim)) for _ in range(order_0 + order_1))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        order_0: int = inp_id.count(0)
        order_1: int = inp_id.count(1)
        if order_1 in (0, 1):
            ID_data = self._compute_internal_0(
                order_0=order_0,
                order_1=order_1,
            )
        else:
            pass

        return ID_data
