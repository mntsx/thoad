# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class MvXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        assert len(shape) == 1
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape = projected_shape
        projected_indep = tuple(None for _ in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = self._grad_fn._saved_self
        saved_vec: Tensor = self._grad_fn._saved_vec
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        if saved_vec is not None:
            saved_vec = saved_vec.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_vec"] = saved_vec
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_vec: Tensor = self._context["saved_vec"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["m"] = saved_self
        processed_context["v"] = saved_vec
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        v: Tensor = self._processed_context["v"]

        ### Instantiate differential
        differential: Tensor = v

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [1]
        einstein_composed: list[list[int]] = [0, 1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(v.shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m: Tensor = self._processed_context["m"]

        ### Instantiate differential
        differential: Tensor = m

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [0, 1]
        einstein_composed: list[list[int]] = [1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(m.shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_01(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m: Tensor = self._processed_context["m"]

        ### Instrumental operations
        dual_size: int = m.shape[1]
        internal_shape: Tuple[int, ...] = (m.shape[1],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [1, 2]
        einstein_composed: list[list[int]] = [[0, 1], [2]]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([*einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_10(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m: Tensor = self._processed_context["m"]

        ### Instrumental operations
        dual_size: int = m.shape[1]
        internal_shape: Tuple[int, ...] = (m.shape[1],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0]
        einstein_internal: list[int] = [1, 2]
        einstein_composed: list[list[int]] = [[1], [0, 2]]
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
