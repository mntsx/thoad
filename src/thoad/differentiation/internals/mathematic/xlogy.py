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
from thoad.differentiation.internals.utils.derivate import log_derivate
from thoad.typing import Shape, Indep, Notation, IDData


class XlogyXBackward0(ContractiveFunction):

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
        if saved_other is not None:
            saved_other = saved_other.to(dtype=self._dtype, device=self._device)
            getattr(saved_other, "_fix_weakref")()
        if saved_self is not None:
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
        input: Union[None, Tensor] = None
        other: Union[None, Tensor] = None
        if isinstance(raw_input, Tensor):
            input = raw_input.broadcast_to(size=broadcasted_shape)
        if isinstance(raw_other, Tensor):
            other = raw_other.broadcast_to(size=broadcasted_shape)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["input"] = input
        processed_context["raw_input"] = raw_input
        processed_context["other"] = other
        processed_context["raw_other"] = raw_other
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        other: Tensor = self._processed_context["other"]

        ### Carry out instrumental operations
        broadcasted_other: Tensor = other.broadcast_to(self._shape0)

        ### Instantiate derivative
        derivative: Tensor = torch.log(broadcasted_other)

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

    def _compute_internal_0(self, order_0: int, order_1: int) -> IDData:
        assert self._processed_context is not None
        assert order_0 == 0 or order_0 == 1
        ### Read context
        input: Tensor = self._processed_context["input"]
        other: Tensor = self._processed_context["other"]

        ### Carry out instrumental operations
        broadcasted_input: Tensor = input.broadcast_to(self._shape0)
        broadcasted_other: Tensor = other.broadcast_to(self._shape0)

        ### Instantiate derivative
        derivative: Tensor = log_derivate(tensor=other, order=order_1)
        if order_0 == 0:
            derivative *= broadcasted_input

        ### Create einstein notation
        ndim: int = len(self._shape0)
        order: int = order_0 + order_1
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
            case _:
                order_0: int = inp_id.count(0)
                order_1: int = inp_id.count(1)
                if order_0 == 0 or order_0 == 1:
                    ID_data = self._compute_internal_0(
                        order_0=order_0,
                        order_1=order_1,
                    )
                else:
                    pass
        ID_data = self._unbroadcast(ID_data=ID_data, inp_id=inp_id)

        return ID_data
