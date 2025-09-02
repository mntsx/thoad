# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.broadcasting import (
    unbroadcast_IDData,
    determnine_repeats,
)
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape, Indep, Notation, IDData


class PreluKernelXBackward0(ContractiveFunction):

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
        raw_input: Tensor = self._processed_context["raw_input"]
        raw_weight: Tensor = self._processed_context["raw_weight"]
        # initialize shape and indep projections
        projected_shape: Shape = broadcasted_shape
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # adjust indeps to broadcasting requirements
        if isinstance(raw_input, Tensor) and isinstance(raw_weight, Tensor):
            input_repeats: Tuple[Union[None, int], ...] = determnine_repeats(
                shape=projected_shape, raw_shape=raw_input.shape
            )
            weight_repeats: Tuple[Union[None, int], ...] = determnine_repeats(
                shape=projected_shape, raw_shape=raw_weight.shape
            )
            aux: list[Union[None, int]] = list(projected_indep)
            for i, d in enumerate(projected_indep):
                if inp_id == 0 and input_repeats[i] != 1:
                    aux[i] = None
                if inp_id == 1 and weight_repeats[i] != 1:
                    aux[i] = None
            projected_indep = tuple(aux)
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract input and weight tensors
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_weight: Tensor = getattr(self._grad_fn, "_saved_weight")
        # ensure proper dtype and device
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        if saved_weight is not None:
            saved_weight = saved_weight.to(dtype=self._dtype, device=self._device)
            getattr(saved_weight, "_fix_weakref")()
        # save raw context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_weight"] = saved_weight
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_weight: Tensor = self._context["saved_weight"]
        # process context
        raw_input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        raw_weight: Tensor = denull_tensor(
            tensor=saved_weight, dtype=self._dtype, device=self._device
        )
        tensors_shapes: list[Shape]
        tensors_shapes = [T.shape for T in (raw_input, raw_weight) if T is not None]
        broadcasted_shape: Shape = infer_broadcast(shapes=tensors_shapes)
        input: Union[None, Tensor] = None
        weight: Union[None, Tensor] = None
        if isinstance(raw_input, Tensor):
            input = raw_input.broadcast_to(size=broadcasted_shape)
        if isinstance(raw_weight, Tensor):
            weight = raw_weight.broadcast_to(size=broadcasted_shape)
        condition: Tensor = input >= 0
        min_bounded_input: Tensor = torch.clamp(input=input, min=0)
        # ---
        squeezed_weight: Tensor = saved_weight.squeeze()
        extended_shape: Tuple[int, ...]
        extended_shape = tuple(
            [s if d == 1 else 1 for d, s in enumerate(saved_self.shape)]
        )
        extended_weight: Tensor = squeezed_weight.view(extended_shape)
        expanded_weight: Tensor = extended_weight.expand(saved_self.shape)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["condition"] = condition
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["input"] = input
        processed_context["raw_input"] = raw_input
        processed_context["min_bounded_input"] = min_bounded_input
        processed_context["weight"] = weight
        processed_context["raw_weight"] = raw_weight
        self._processed_context = processed_context
        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        condition: Tensor = self._processed_context["condition"]
        weight: Tensor = self._processed_context["weight"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate internal derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=t1,
            other=weight,
        )

        ### Create einstein notation
        ndim: int = len(self._shape0)
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

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        condition: Tensor = self._processed_context["condition"]
        input: Tensor = self._processed_context["input"]

        ### Carry out instrumental operations
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate internal derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=t0,
            other=input,
        )

        ### Create einstein notation
        ndim: int = len(self._shape0)
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

    def _compute_crossed_internal(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        condition: Tensor = self._processed_context["condition"]

        ### Carry out instrumental operations
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate internal derivative
        derivative: Tensor = torch.where(
            condition=condition,
            input=t0,
            other=t1,
        )

        ### Create einstein notation
        ndim: int = len(self._shape0)
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)), tuple(range(ndim)))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(True for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def _compute_internal_0_01(self) -> IDData:
        return self._compute_crossed_internal()

    def _compute_internal_0_10(self) -> IDData:
        return self._compute_crossed_internal()

    def _unbroadcast(self, ID_data: IDData, inp_id: Tuple[int, ...]) -> IDData:
        assert self._processed_context is not None
        ### Read context
        input: Tensor = self._processed_context["input"]
        raw_input: Tensor = self._processed_context["raw_input"]
        weight: Tensor = self._processed_context["weight"]
        raw_weight: Tensor = self._processed_context["raw_weight"]

        if isinstance(raw_input, Tensor) and isinstance(raw_weight, Tensor):

            ### Determine repeats
            repeats_input: Tuple[Union[None, int], ...]
            repeats_input = determnine_repeats(
                shape=input.shape, raw_shape=raw_input.shape
            )
            repeats_weight: Tuple[Union[None, int], ...]
            repeats_weight = determnine_repeats(
                shape=weight.shape, raw_shape=raw_weight.shape
            )
            tensors_repeats: list[Tuple[Union[None, int], ...]] = list()
            for i in inp_id:
                match i:
                    case 0:
                        tensors_repeats.append(repeats_input)
                    case 1:
                        tensors_repeats.append(repeats_weight)

            ### Unbroadcast IDData
            tensor_unbroadcast: bool = False
            tensor_unbroadcast |= input.shape != raw_input.shape
            tensor_unbroadcast |= weight.shape != raw_weight.shape
            if ID_data != (None, None) and tensor_unbroadcast:
                ID_data = unbroadcast_IDData(
                    ID_data=ID_data,
                    tensors_repeats=tensors_repeats,
                )

        return ID_data

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()
            case (0, (1,)):
                ID_data = self._compute_internal_0_1()
            case (0, (0, 1)):
                ID_data = self._compute_internal_0_01()
            case (0, (1, 0)):
                ID_data = self._compute_internal_0_10()
        ID_data = self._unbroadcast(ID_data=ID_data, inp_id=inp_id)

        return ID_data
