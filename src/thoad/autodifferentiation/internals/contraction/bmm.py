# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class BmmXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_mat2: Tensor = self._context["saved_mat2"]
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Tuple[int, ...]
        if saved_mat2 is not None and saved_self is not None:
            projected_shape = (*saved_self.shape[:2], saved_mat2.shape[2])
        else:
            assert len(shape) == 3
            first_size: int
            if saved_mat2 is None:
                first_size = saved_self.shape[0]
            else:
                first_size = saved_mat2.shape[0]
            assert shape[0] == first_size
            projected_shape = shape
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
        aux: list[Union[None, int]] = list(projected_indep)
        for dim in [1, 2]:
            if dim in projected_indep:
                aux[projected_indep.index(dim)] = None
        projected_indep = tuple(aux)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = self._grad_fn._saved_self
        saved_mat2: Tensor = self._grad_fn._saved_mat2
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        if saved_mat2 is not None:
            saved_mat2 = saved_mat2.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_mat2"] = saved_mat2
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_mat2: Tensor = self._context["saved_mat2"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["m1"] = saved_self
        processed_context["m2"] = saved_mat2
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m2: Tensor = self._processed_context["m2"]

        ### Instantiate differential
        differential: Tensor = m2

        ### Create einstein notation
        einstein_external: list[int] = [0, 1, 3]
        einstein_internal: list[int] = [0, 2, 3]
        einstein_composed: list[list[int]] = [0, 1, 2]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(m2.shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m1: Tensor = self._processed_context["m1"]

        ### Instantiate differential
        differential: Tensor = m1

        ### Create einstein notation
        einstein_external: list[int] = [0, 1, 3]
        einstein_internal: list[int] = [0, 1, 2]
        einstein_composed: list[list[int]] = [0, 2, 3]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(m1.shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_01(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m1: Tensor = self._processed_context["m1"]

        ### Instrumental operations
        dual_size: int = m1.shape[2]
        internal_shape: Tuple[int, ...] = (m1.shape[2],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0, 1, 3]
        einstein_internal: list[int] = [2, 4]
        einstein_composed: list[list[int]] = [[0, 1, 2], [0, 4, 3]]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([*einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_10(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m1: Tensor = self._processed_context["m1"]

        ### Instrumental operations
        dual_size: int = m1.shape[2]
        internal_shape: Tuple[int, ...] = (m1.shape[2],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0, 1, 3]
        einstein_internal: list[int] = [2, 4]
        einstein_composed: list[list[int]] = [[0, 2, 3], [0, 1, 4]]
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
