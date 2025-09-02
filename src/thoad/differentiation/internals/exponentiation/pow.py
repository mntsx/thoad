# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.broadcasting import (
    unbroadcast_IDData,
    determnine_repeats,
)
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.differentiation.internals.utils.derivate import (
    pow0_derivate,
    pow1_derivate,
    pow1_base_derivate,
    pow1_exponent_derivate,
)
from thoad.typing import Shape, Indep, Notation, IDData


class PowXBackward0(ContractiveFunction):

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
        # save input
        saved_exponent: float = getattr(self._grad_fn, "_saved_exponent")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict["str", Any] = dict()
        context["saved_exponent"] = saved_exponent
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        # load context
        saved_exponent: float = self._context["saved_exponent"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict["str", Any] = dict()
        processed_context["exponent"] = saved_exponent
        processed_context["input"] = input
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> IDData:
        assert self._processed_context is not None
        ### Read context
        exponent: float = self._processed_context["exponent"]
        input: Tensor = self._processed_context["input"]

        ### Instantiate internal derivative derivative
        derivative: Tensor = pow0_derivate(
            tensor=input,
            exponent=exponent,
            order=order,
        )

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = tuple(tuple(range(ndim)) for _ in range(order))
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
        ID_data: IDData
        if out_id == 0 and all(i == 0 for i in inp_id):
            order: int = len(inp_id)
            ID_data = self._compute_internal_0(order=order)
        else:
            ID_data = (None, None)
        return ID_data


class PowXBackward1(ContractiveFunction):

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
        output: Tensor = self._processed_context["output"]
        raw_input: Tensor = self._processed_context["raw_input"]
        raw_exponent: Tensor = self._processed_context["raw_exponent"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(output.shape)
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # adjust indeps to broadcasting requirements
        if isinstance(raw_input, Tensor) and isinstance(raw_exponent, Tensor):
            input_repeats: Tuple[Union[None, int], ...] = determnine_repeats(
                shape=output.shape, raw_shape=raw_input.shape
            )
            exponent_repeats = determnine_repeats(
                shape=output.shape, raw_shape=raw_exponent.shape
            )
            aux: list[Union[None, int]] = list(projected_indep)
            for i, d in enumerate(projected_indep):
                if inp_id == 0 and input_repeats[i] != 1:
                    aux[i] = None
                if inp_id == 1 and exponent_repeats[i] != 1:
                    aux[i] = None
            projected_indep = tuple(aux)
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_exponent: Tensor = getattr(self._grad_fn, "_saved_exponent")
        saved_result: Tensor = getattr(self._grad_fn, "_saved_result")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        if saved_exponent is not None:
            saved_exponent = saved_exponent.to(dtype=self._dtype, device=self._device)
            getattr(saved_exponent, "_fix_weakref")()
        if saved_result is not None:
            saved_result = saved_result.to(dtype=self._dtype, device=self._device)
            getattr(saved_result, "_fix_weakref")()
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_exponent"] = saved_exponent
        context["saved_result"] = saved_result
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_exponent: Tensor = self._context["saved_exponent"]
        saved_result: Tensor = self._context["saved_result"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        raw_exponent: Tensor = denull_tensor(
            tensor=saved_exponent, dtype=self._dtype, device=self._device
        )
        raw_input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        output: Tensor = denull_tensor(
            tensor=saved_result, dtype=self._dtype, device=self._device
        )
        input: Tensor = raw_input.broadcast_to(size=output.shape)
        exponent: Tensor = raw_exponent.broadcast_to(size=output.shape)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["exponent"] = exponent
        processed_context["raw_exponent"] = raw_exponent
        processed_context["input"] = input
        processed_context["raw_input"] = raw_input
        processed_context["output"] = output
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self, order: int) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        assert order >= 1
        ### Read context
        exponent: Tensor = self._processed_context["exponent"]
        input: Tensor = self._processed_context["input"]

        ### Instantiate derivative
        derivative: Tensor = pow1_base_derivate(
            input=input,
            exponent=exponent,
            order=order,
            dtype=self._dtype,
            device=self._device,
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

    def _compute_internal_0_1(self, order: int) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        assert order >= 1
        ### Read context
        input: Tensor = self._processed_context["input"]
        output: Tensor = self._processed_context["output"]

        ### Instantiate derivative
        derivative: Tensor = pow1_exponent_derivate(
            input=input,
            output=output,
            order=order,
            dtype=self._dtype,
            device=self._device,
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

    def _compute_internal_0(self, derivations: Tuple[int, ...]) -> IDData:
        assert self._processed_context is not None
        assert all(i in (0, 1) for i in derivations)

        ### Read context
        input: Tensor = self._processed_context["input"]
        exponent: Tensor = self._processed_context["exponent"]

        ### Instantiate derivative
        derivative: Tensor = pow1_derivate(
            base=input,
            exponent=exponent,
            derivations=derivations,
            dtype=self._dtype,
            device=self._device,
        )

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(range(ndim))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = tuple(tuple(range(ndim)) for _ in derivations)
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
        output: Tensor = self._processed_context["output"]
        raw_input: Tensor = self._processed_context["raw_input"]
        raw_exponent: Tensor = self._processed_context["raw_exponent"]

        if isinstance(raw_input, Tensor) and isinstance(raw_exponent, Tensor):

            ### Determine repeats
            repeats_input: Tuple[Union[None, int], ...]
            repeats_input = determnine_repeats(
                shape=output.shape, raw_shape=raw_input.shape
            )
            repeats_exponent: Tuple[Union[None, int], ...]
            repeats_exponent = determnine_repeats(
                shape=output.shape, raw_shape=raw_exponent.shape
            )
            tensors_repeats: list[Tuple[Union[None, int], ...]] = list()
            for i in inp_id:
                match i:
                    case 0:
                        tensors_repeats.append(repeats_input)
                    case 1:
                        tensors_repeats.append(repeats_exponent)

            ### Unbroadcast IDData
            tensor_unbroadcast: bool = False
            tensor_unbroadcast |= output.shape != raw_input.shape
            tensor_unbroadcast |= output.shape != raw_exponent.shape
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
        order_0: int = inp_id.count(0)
        order_1: int = inp_id.count(1)
        if order_1 == 0:
            ID_data = self._compute_internal_0_0(order=order_0)
        if order_0 == 0:
            ID_data = self._compute_internal_0_1(order=order_1)
        if order_0 > 0 and order_1 > 0:
            ID_data = self._compute_internal_0(derivations=inp_id)
        if order_0 == 0 and order_1 == 0:
            pass
        ID_data = self._unbroadcast(ID_data=ID_data, inp_id=inp_id)

        return ID_data
