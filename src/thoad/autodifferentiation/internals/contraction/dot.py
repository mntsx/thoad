# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class DotXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        projected_shape: Shape = (1,)
        projected_indep: Indep = (None,)
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = self._grad_fn._saved_self
        saved_tensor: Tensor = self._grad_fn._saved_tensor
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        if saved_tensor is not None:
            saved_tensor = saved_tensor.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_tensor"] = saved_tensor
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_tensor: Tensor = self._context["saved_tensor"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["v1"] = saved_self
        processed_context["v2"] = saved_tensor
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        v2: Tensor = self._processed_context["v2"]

        ### Instantiate differential
        internal_shape: Tuple[int, ...] = (1, v2.numel())
        differential: Tensor = v2.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [0, 1]
        einstein_composed: list[list[int]] = [1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        v1: Tensor = self._processed_context["v1"]

        ### Instantiate differential
        internal_shape: Tuple[int, ...] = (1, v1.numel())
        differential: Tensor = v1.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [0, 1]
        einstein_composed: list[list[int]] = [1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_01(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        v1: Tensor = self._processed_context["v1"]

        ### Instrumental operations
        dual_size: int = v1.numel()
        internal_shape: Tuple[int, ...] = (1, v1.numel(), v1.numel())

        ### Instantiate differential
        differential: Tensor = torch.eye(n=dual_size).view(size=internal_shape)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [0, 1, 2]
        einstein_composed: list[list[int]] = [[1], [2]]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([*einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_10(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        v1: Tensor = self._processed_context["v1"]

        ### Instrumental operations
        dual_size: int = v1.numel()
        internal_shape: Tuple[int, ...] = (1, v1.numel(), v1.numel())

        ### Instantiate differential
        differential: Tensor = torch.eye(n=dual_size).view(size=internal_shape)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [0, 1, 2]
        einstein_composed: list[list[int]] = [[2], [1]]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([*einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case (0, (1,)):
                (differential, einstein_notation) = self._compute_internal_0_1()
            case (0, (0, 1)):
                (differential, einstein_notation) = self._compute_internal_0_01()
            case (0, (1, 0)):
                (differential, einstein_notation) = self._compute_internal_0_10()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
