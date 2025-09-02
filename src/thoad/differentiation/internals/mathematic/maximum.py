# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import (
    shape_align_indep,
    shape_broadcastable,
)
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.broadcasting import (
    unbroadcast_IDData,
    determnine_repeats,
)
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape, Indep, Notation, IDData


class MaximumXBackward0(ContractiveFunction):

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
        raw_other: Tensor = self._processed_context["raw_other"]
        # initialize shape and indep projections
        projected_shape: Shape = broadcasted_shape
        projected_indep: Indep = indep
        broadcastable: bool = shape_broadcastable(shape=projected_shape, target=shape)
        if None in (raw_input, raw_other) and broadcastable:
            projected_shape = shape
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # adjust indeps to broadcasting requirements
        if isinstance(raw_input, Tensor) and isinstance(raw_other, Tensor):
            input_repeats: Tuple[Union[None, int], ...] = determnine_repeats(
                shape=projected_shape, raw_shape=raw_input.shape
            )
            other_repeats: Tuple[Union[None, int], ...] = determnine_repeats(
                shape=projected_shape, raw_shape=raw_other.shape
            )
            aux: list[Union[None, int]] = list(projected_indep)
            for i, d in enumerate(projected_indep):
                if inp_id == 0 and input_repeats[i] != 1:
                    aux[i] = None
                if inp_id == 1 and other_repeats[i] != 1:
                    aux[i] = None
            projected_indep = tuple(aux)
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_other: Tensor = getattr(self._grad_fn, "_saved_other")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        saved_other = saved_other.to(dtype=self._dtype, device=self._device)
        getattr(saved_other, "_fix_weakref")()
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_other"] = saved_other
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_other: Tensor = self._context["saved_other"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        raw_input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        raw_other: Tensor = denull_tensor(
            tensor=saved_other, dtype=self._dtype, device=self._device
        )
        tensors_shapes: list[Shape]
        tensors_shapes = [T.shape for T in (raw_input, raw_other) if T is not None]
        broadcasted_shape: Shape = infer_broadcast(shapes=tensors_shapes)
        input: Tensor = raw_input.broadcast_to(size=broadcasted_shape)
        other: Tensor = raw_other.broadcast_to(size=broadcasted_shape)
        input_weight: Tensor
        other_weight: Tensor
        input_weight = (input >= other).to(dtype=self._dtype, device=self._device)
        other_weight = (other >= input).to(dtype=self._dtype, device=self._device)
        weight_sum: Tensor = input_weight + other_weight
        input_weight /= weight_sum
        other_weight /= weight_sum
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["input"] = input
        processed_context["raw_input"] = raw_input
        processed_context["other"] = other
        processed_context["raw_other"] = raw_other
        processed_context["input_weight"] = input_weight
        processed_context["other_weight"] = other_weight
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        input_weight: Tensor = self._processed_context["input_weight"]

        ### Instantiate derivative
        derivative: Tensor = input_weight

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
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        other_weight: Tensor = self._processed_context["other_weight"]

        ### Instantiate derivative
        derivative: Tensor = other_weight

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

    def _unbroadcast(self, ID_data: IDData, inp_id: Tuple[int, ...]) -> IDData:
        assert self._processed_context is not None
        ### Read context
        input: Tensor = self._processed_context["input"]
        raw_input: Tensor = self._processed_context["raw_input"]
        other: Tensor = self._processed_context["other"]
        raw_other: Tensor = self._processed_context["raw_other"]

        if isinstance(raw_input, Tensor) and isinstance(raw_other, Tensor):

            ### Determine repeats
            repeats_input: Tuple[Union[None, int], ...]
            repeats_input = determnine_repeats(
                shape=input.shape, raw_shape=raw_input.shape
            )
            repeats_other: Tuple[Union[None, int], ...]
            repeats_other = determnine_repeats(
                shape=other.shape, raw_shape=raw_other.shape
            )
            tensors_repeats: list[Tuple[Union[None, int], ...]] = list()
            for i in inp_id:
                match i:
                    case 0:
                        tensors_repeats.append(repeats_input)
                    case 1:
                        tensors_repeats.append(repeats_other)

            ### Unbroadcast IDData
            tensor_unbroadcast: bool = False
            tensor_unbroadcast |= input.shape != raw_input.shape
            tensor_unbroadcast |= other.shape != raw_other.shape
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
        ID_data = self._unbroadcast(ID_data=ID_data, inp_id=inp_id)

        return ID_data
