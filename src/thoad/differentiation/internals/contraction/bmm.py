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


class BmmXBackward0(ContractiveFunction):

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
        m2: Tensor = self._processed_context["m2"]
        m1: Tensor = self._processed_context["m1"]
        projected_shape: Tuple[int, ...]
        if m2 is not None and m1 is not None:
            projected_shape = (*m1.shape[:2], m2.shape[2])
        else:
            assert len(shape) == 3
            first_size: int
            if m2 is None:
                first_size = m1.shape[0]
            else:
                first_size = m2.shape[0]
            assert shape[0] == first_size
            projected_shape = shape
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
        match inp_id:
            case 0:
                projected_indep = tuple(
                    d if d in (0, 1) else None for d in projected_indep
                )
            case 1:
                projected_indep = tuple(
                    d if d in (0, 2) else None for d in projected_indep
                )
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_mat2: Tensor = getattr(self._grad_fn, "_saved_mat2")
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        if saved_mat2 is not None:
            saved_mat2 = saved_mat2.to(dtype=self._dtype, device=self._device)
            getattr(saved_mat2, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_mat2"] = saved_mat2
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_mat2: Tensor = self._context["saved_mat2"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        m1: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        m2: Tensor = denull_tensor(
            tensor=saved_mat2, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["m1"] = m1
        processed_context["m2"] = m2
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        m2: Tensor = self._processed_context["m2"]

        ### Instantiate derivative
        derivative: Tensor = m2

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 1, 3)
        einstein_internal: Tuple[int, ...] = (0, 2, 3)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1, 2),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(m2.shape), (True, False, False)))

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1: Tensor = self._processed_context["m1"]

        ### Instantiate derivative
        derivative: Tensor = m1

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 1, 3)
        einstein_internal: Tuple[int, ...] = (0, 1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 2, 3),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(m1.shape), (True, False, False)))

        return (derivative, einstein_notation)

    def _compute_internal_0_01(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1: Tensor = self._processed_context["m1"]
        m2: Tensor = self._processed_context["m2"]

        ### Instrumental operations
        dual_size: int = m1.shape[2]
        internal_shape: Tuple[int, ...] = (m1.shape[2],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 1, 3)
        einstein_internal: Tuple[int, ...] = (2, 4)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1, 2), (0, 4, 3))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_10(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1: Tensor = self._processed_context["m1"]
        m2: Tensor = self._processed_context["m2"]

        ### Instrumental operations
        dual_size: int = m1.shape[2]
        internal_shape: Tuple[int, ...] = (m1.shape[2],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 1, 3)
        einstein_internal: Tuple[int, ...] = (2, 4)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 2, 3), (0, 1, 4))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

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
