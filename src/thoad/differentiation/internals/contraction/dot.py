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


class DotXBackward0(ContractiveFunction):

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
        # initialize shape and indep projections
        projected_shape: Shape = (1,)
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
        # adjust projected_indep to input
        projected_indep = tuple(None for _ in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_tensor: Tensor = getattr(self._grad_fn, "_saved_tensor")
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        if saved_tensor is not None:
            saved_tensor = saved_tensor.to(dtype=self._dtype, device=self._device)
            getattr(saved_tensor, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_tensor"] = saved_tensor
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_tensor: Tensor = self._context["saved_tensor"]
        # process context
        v1: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        v2: Tensor = denull_tensor(
            tensor=saved_tensor, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["v1"] = v1
        processed_context["v2"] = v2
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        v2: Tensor = self._processed_context["v2"]

        ### Instantiate derivative
        internal_shape: Tuple[int, ...] = (1, v2.numel())
        derivative: Tensor = v2.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (0, 1)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1,),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        v1: Tensor = self._processed_context["v1"]

        ### Instantiate derivative
        internal_shape: Tuple[int, ...] = (1, v1.numel())
        derivative: Tensor = v1.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (0, 1)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1,),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_01(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        v1: Tensor = self._processed_context["v1"]

        ### Instrumental operations
        dual_size: int = v1.numel()
        internal_shape: Tuple[int, ...] = (1, v1.numel(), v1.numel())

        ### Instantiate derivative
        derivative: Tensor = torch.eye(
            n=dual_size,
            dtype=self._dtype,
            device=self._device,
        )
        derivative = derivative.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (0, 1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1,), (2,))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_10(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        v1: Tensor = self._processed_context["v1"]

        ### Instrumental operations
        dual_size: int = v1.numel()
        internal_shape: Tuple[int, ...] = (1, v1.numel(), v1.numel())

        ### Instantiate derivative
        derivative: Tensor = torch.eye(
            n=dual_size,
            dtype=self._dtype,
            device=self._device,
        )
        derivative = derivative.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (0, 1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((2,), (1,))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True, True)))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()
            case (0, (1,)):
                ID_data = self._compute_internal_0_1()
            case (
                0,
                (
                    0,
                    1,
                ),
            ):
                ID_data = self._compute_internal_0_01()
            case (0, (1, 0)):
                ID_data = self._compute_internal_0_10()

        return ID_data
