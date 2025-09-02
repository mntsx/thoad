# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor, denull_shape
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape, Indep, Notation, IDData


class SmoothL1LossXBackward0(ContractiveFunction):

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
        saved_beta: float = getattr(self._grad_fn, "_saved_beta")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_target: Tensor = getattr(self._grad_fn, "_saved_target")
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        saved_target = saved_target.to(dtype=self._dtype, device=self._device)
        getattr(saved_target, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_reduction"] = saved_reduction
        context["saved_beta"] = saved_beta
        context["saved_self"] = saved_self
        context["saved_target"] = saved_target
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_beta: float = self._context["saved_beta"]
        saved_reduction: int = self._context["saved_reduction"]
        saved_self: Tensor = self._context["saved_self"]
        saved_target: Tensor = self._context["saved_target"]
        # process context
        reduce: bool = saved_reduction in (1, 2)
        N: int = max(saved_self.numel(), saved_target.numel())
        constant1: float = 1 / (saved_beta * (N if saved_reduction == 1 else 1))
        constant2: float = 1 / (N if saved_reduction == 1 else 1)
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        target: Tensor = denull_tensor(
            tensor=saved_target, dtype=self._dtype, device=self._device
        )
        tensors_shapes: list[Shape]
        tensors_shapes = [T.shape for T in (input, target) if T is not None]
        broadcasted_shape: Shape = infer_broadcast(shapes=tensors_shapes)
        difference: Tensor = input - target
        condition: Tensor = abs(difference) < saved_beta
        sign: Tensor = torch.where(
            condition=(difference > 0),
            input=torch.ones(size=(1,), dtype=self._dtype, device=self._device),
            other=torch.full(
                fill_value=(-1.0), size=(1,), dtype=self._dtype, device=self._device
            ),
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["constant1"] = constant1
        processed_context["constant2"] = constant2
        processed_context["condition"] = condition
        processed_context["difference"] = difference
        processed_context["input"] = input
        processed_context["target"] = target
        processed_context["sign"] = sign
        processed_context["reduce"] = reduce
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        constant1: float = self._processed_context["constant1"]
        constant2: float = self._processed_context["constant2"]
        condition: Tensor = self._processed_context["condition"]
        difference: Tensor = self._processed_context["difference"]
        sign: Tensor = self._processed_context["sign"]
        reduce: bool = self._processed_context["reduce"]

        ### Carry out instrumental operations
        # declare info about internal derivative shape & values
        internal_shape: list[int] = list(broadcasted_shape)
        internal_independencies: list[bool] = [True for _ in broadcasted_shape]
        # create auxiliar tensors
        tensor1: Tensor = constant1 * difference
        tensor2: Tensor = constant2 * sign

        ### Instantiate derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=tensor1,
            other=tensor2,
        )

        ### Create einstein notation
        ndim: int = len(broadcasted_shape)
        einstein_external: Tuple[int, ...] = tuple() if reduce else tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        constant1: float = self._processed_context["constant1"]
        constant2: float = self._processed_context["constant2"]
        condition: Tensor = self._processed_context["condition"]
        difference: Tensor = self._processed_context["difference"]
        sign: Tensor = self._processed_context["sign"]
        reduce: bool = self._processed_context["reduce"]

        ### Carry out instrumental operations
        # declare info about internal derivative shape & values
        internal_shape: list[int] = list(broadcasted_shape)
        internal_independencies: list[bool] = [True for _ in broadcasted_shape]
        # create auxiliar tensors
        tensor1: Tensor = (-constant1) * difference
        tensor2: Tensor = (-constant2) * sign

        ### Instantiate derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=tensor1,
            other=tensor2,
        )

        ### Create einstein notation
        ndim: int = len(broadcasted_shape)
        einstein_external: Tuple[int, ...] = tuple() if reduce else tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def _compute_double_internal(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        constant1: float = self._processed_context["constant1"]
        condition: Tensor = self._processed_context["condition"]
        reduce: bool = self._processed_context["reduce"]

        ### Carry out instrumental operations
        # declare info about internal derivative shape & values
        internal_shape: list[int] = list(broadcasted_shape)
        internal_independencies: list[bool] = [True for _ in broadcasted_shape]
        # create auxiliar tensors
        tensor1: Tensor = torch.full(
            fill_value=constant1, size=(1,), dtype=self._dtype, device=self._device
        )
        tensor2: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=tensor1,
            other=tensor2,
        )

        ### Create einstein notation
        ndim: int = len(broadcasted_shape)
        einstein_external: Tuple[int, ...] = tuple() if reduce else tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)), tuple(range(ndim)))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def _compute_internal_0_00(self) -> IDData:
        return self._compute_double_internal()

    def _compute_internal_0_11(self) -> IDData:
        return self._compute_double_internal()

    def _compute_crossed_internal(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        broadcasted_shape: Shape = self._processed_context["broadcasted_shape"]
        constant1: float = self._processed_context["constant1"]
        condition: Tensor = self._processed_context["condition"]
        reduce: bool = self._processed_context["reduce"]

        ### Carry out instrumental operations
        # declare info about internal derivative shape & values
        internal_shape: list[int] = list(broadcasted_shape)
        internal_independencies: list[bool] = [True for _ in broadcasted_shape]
        # create auxiliar tensors
        tensor1: Tensor = torch.full(
            fill_value=(-constant1), size=(1,), dtype=self._dtype, device=self._device
        )
        tensor2: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=tensor1,
            other=tensor2,
        )

        ### Create einstein notation
        ndim: int = len(broadcasted_shape)
        einstein_external: Tuple[int, ...] = tuple() if reduce else tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)), tuple(range(ndim)))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)
