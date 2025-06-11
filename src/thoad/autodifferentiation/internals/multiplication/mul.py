# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class MulXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_other: Tensor = self._context["saved_other"]
        saved_self: Tensor = self._context["saved_self"]
        tensors: list[Tensor] = [saved_other, saved_self]
        tensors_shapes = [tuple(T.shape) for T in tensors if T is not None]
        output_shape: list[int] = infer_broadcast(shapes=tensors_shapes)
        projected_shape: Shape = tuple(output_shape)
        projected_indep: Indep = indep
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
        saved_other: Tensor = self._grad_fn._saved_other
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_other = saved_other.to(dtype=self._dtype, device=self._device)
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_other"] = saved_other
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_other: Tensor = self._context["saved_other"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input"] = saved_self
        processed_context["other"] = saved_other
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        other: Tensor = self._processed_context["other"]

        ### Carry out instrumental operations
        broadcasted_other: Tensor = other.broadcast_to(self._shape)

        ### Instantiate differential
        differential: Tensor = broadcasted_other

        ### Create einstein notation
        ndim: int = len(self._shape)
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim))]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        input: Tensor = self._processed_context["input"]

        ### Carry out instrumental operations
        broadcasted_input: Tensor = input.broadcast_to(self._shape)

        ### Instantiate differential
        differential: Tensor = broadcasted_input

        ### Create einstein notation
        ndim: int = len(self._shape)
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim))]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def _compute_internal_0_01(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        # ...

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1.sum()  # view(size=self._shape)

        ### Create einstein notation
        ndim: int = len(self._shape)
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list()
        einstein_composed: list[list[int]] = [list(range(ndim)) for _ in range(2)]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list()])

        return (differential, einstein_notation)

    def _compute_internal_0_10(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None

        differential: Tensor
        einstein_notation: Notation
        (differential, einstein_notation) = self._compute_internal_0_01()

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
