# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor, denull_shape
from thoad.typing import Shape, Indep, Notation, IDData


class MmXBackward0(ContractiveFunction):

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
        m2_shape: Shape = self._processed_context["m2_shape"]
        m1_shape: Shape = self._processed_context["m1_shape"]
        # initialize shape and indep projections
        projected_shape: Shape = (m1_shape[0], m2_shape[1])
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
                projected_indep = tuple(0 if d == 0 else None for d in projected_indep)
            case 1:
                projected_indep = tuple(1 if d == 1 else None for d in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        saved_self_sym_strides: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_strides"
        )
        saved_mat2: Tensor = getattr(self._grad_fn, "_saved_mat2")
        saved_mat2_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_mat2_sym_sizes"
        )
        saved_mat2_sym_strides: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_mat2_sym_strides"
        )
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
        context["saved_mat2_sym_sizes"] = saved_mat2_sym_sizes
        context["saved_mat2_sym_strides"] = saved_mat2_sym_strides
        context["saved_self"] = saved_self
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        context["saved_self_sym_strides"] = saved_self_sym_strides
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_mat2: Tensor = self._context["saved_mat2"]
        saved_mat2_sym_sizes: Tuple[int, ...] = self._context["saved_mat2_sym_sizes"]
        saved_self: Tensor = self._context["saved_self"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        m1: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        m1_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        m2: Tensor = denull_tensor(
            tensor=saved_mat2, dtype=self._dtype, device=self._device
        )
        m2_shape: Shape = denull_shape(shape=saved_mat2_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["m1"] = m1
        processed_context["m1_shape"] = m1_shape
        processed_context["m2"] = m2
        processed_context["m2_shape"] = m2_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        m2: Tensor = self._processed_context["m2"]
        m2_shape: Shape = self._processed_context["m2_shape"]

        ### Instantiate derivative
        derivative: Tensor = m2

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 2)
        einstein_internal: Tuple[int, ...] = (1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(m2_shape), (False, False)))

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1: Tensor = self._processed_context["m1"]
        m1_shape: Shape = self._processed_context["m1_shape"]

        ### Instantiate derivative
        derivative: Tensor = m1

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 2)
        einstein_internal: Tuple[int, ...] = (0, 1)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1, 2),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(m1_shape), (False, False)))

        return (derivative, einstein_notation)

    def _compute_internal_0_01(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1_shape: Shape = self._processed_context["m1_shape"]
        m2_shape: Shape = self._processed_context["m2_shape"]

        ### Instrumental operations
        dual_size: int = m1_shape[1]
        internal_shape: Tuple[int, ...] = (m1_shape[1],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 2)
        einstein_internal: Tuple[int, ...] = (1, 3)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1), (3, 2))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_10(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m1_shape: Shape = self._processed_context["m1_shape"]
        m2_shape: Shape = self._processed_context["m2_shape"]

        ### Instrumental operations
        dual_size: int = m1_shape[1]
        internal_shape: Tuple[int, ...] = (m1_shape[1],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0, 2)
        einstein_internal: Tuple[int, ...] = (1, 3)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1, 2), (0, 3))
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
