# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class AddmmXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_mat1_sym_sizes: Tuple[int, ...] = self._context["saved_mat1_sym_sizes"]
        saved_mat2_sym_sizes: Tuple[int, ...] = self._context["saved_mat2_sym_sizes"]
        projected_shape: Shape = (saved_mat1_sym_sizes[0], saved_mat2_sym_sizes[1])
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
        projected_indep = tuple(None for _ in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_alpha: float = self._grad_fn._saved_alpha
        saved_beta: float = self._grad_fn._saved_beta
        saved_mat1: Tensor = self._grad_fn._saved_mat1
        saved_mat1_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_mat1_sym_sizes
        saved_mat1_sym_strides: Tuple[int, ...] = self._grad_fn._saved_mat1_sym_strides
        saved_mat2: Tensor = self._grad_fn._saved_mat2
        saved_mat2_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_mat2_sym_sizes
        saved_mat2_sym_strides: Tuple[int, ...] = self._grad_fn._saved_mat2_sym_strides
        # ensure proper tensor configuration
        if saved_mat1 is not None:
            saved_mat1 = saved_mat1.to(dtype=self._dtype, device=self._device)
        if saved_mat2 is not None:
            saved_mat2 = saved_mat2.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_alpha"] = saved_alpha
        context["saved_beta"] = saved_beta
        context["saved_mat1"] = saved_mat1
        context["saved_mat1_sym_sizes"] = saved_mat1_sym_sizes
        context["saved_mat1_sym_strides"] = saved_mat1_sym_strides
        context["saved_mat2"] = saved_mat2
        context["saved_mat2_sym_sizes"] = saved_mat2_sym_sizes
        context["saved_mat2_sym_strides"] = saved_mat2_sym_strides

        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_alpha: float = self._context["saved_alpha"]
        saved_beta: float = self._context["saved_beta"]
        saved_mat1: Tensor = self._context["saved_mat1"]
        saved_mat1_sym_sizes: Tuple[int, ...] = self._context["saved_mat1_sym_sizes"]
        saved_mat2: Tensor = self._context["saved_mat2"]
        saved_mat2_sym_sizes: Tuple[int, ...] = self._context["saved_mat2_sym_sizes"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["alpha"] = saved_alpha
        processed_context["beta"] = saved_beta
        processed_context["m1"] = saved_mat1
        processed_context["m1_shape"] = saved_mat1_sym_sizes
        processed_context["m2"] = saved_mat2
        processed_context["m2_shape"] = saved_mat2_sym_sizes
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        beta: float = self._processed_context["beta"]
        m1_shape: Tuple[int, ...] = self._processed_context["m1_shape"]
        m2_shape: Tuple[int, ...] = self._processed_context["m2_shape"]

        ### Carry out instrumental opertoins
        # create dummy tensor
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor
        internal_indices: list[int]
        internal_shape: list[int]
        if beta == 1.0:
            internal_indices = list()
            internal_shape = list()
            differential: Tensor = t1.sum()
        else:
            internal_indices = [0, 1]
            internal_shape = [m1_shape[0], m2_shape[1]]
            differential: Tensor = beta * t1.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: list[int] = [0, 1]
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = [0, 1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([internal_shape])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        alpha: float = self._processed_context["alpha"]
        m2: Tensor = self._processed_context["m2"]
        m2_shape: Tuple[int, ...] = self._processed_context["m2_shape"]

        ### Instantiate differential
        differential: Tensor = alpha * m2

        ### Create einstein notation
        einstein_external: list[int] = [0, 2]
        einstein_internal: list[int] = [1, 2]
        einstein_composed: list[list[int]] = [0, 1]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(m2_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_2(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        alpha: float = self._processed_context["alpha"]
        m1: Tensor = self._processed_context["m1"]
        m1_shape: Tuple[int, ...] = self._processed_context["m1_shape"]

        ### Instantiate differential
        differential: Tensor = alpha * m1

        ### Create einstein notation
        einstein_external: list[int] = [0, 2]
        einstein_internal: list[int] = [0, 1]
        einstein_composed: list[list[int]] = [1, 2]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([list(m1_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_12(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m1_shape: Tuple[int, ...] = self._processed_context["m1_shape"]

        ### Instrumental operations
        dual_size: int = m1_shape[1]
        internal_shape: Tuple[int, ...] = (m1_shape[1],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0, 2]
        einstein_internal: list[int] = [1, 3]
        einstein_composed: list[list[int]] = [[0, 1], [3, 2]]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([*einstein_composed])
        einstein_notation.append([list(internal_shape)])

        return (differential, einstein_notation)

    def _compute_internal_0_21(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        m1_shape: Tuple[int, ...] = self._processed_context["m1_shape"]

        ### Instrumental operations
        dual_size: int = m1_shape[1]
        internal_shape: Tuple[int, ...] = (m1_shape[1],) * 2

        ### Instantiate differential
        differential: Tensor = torch.eye(dual_size)

        ### Create einstein notation
        einstein_external: list[int] = [0, 2]
        einstein_internal: list[int] = [1, 3]
        einstein_composed: list[list[int]] = [[1, 2], [0, 3]]
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
            case (0, (2,)):
                (differential, einstein_notation) = self._compute_internal_0_2()
            case (0, (1, 2)):
                (differential, einstein_notation) = self._compute_internal_0_12()
            case (0, (2, 1)):
                (differential, einstein_notation) = self._compute_internal_0_21()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
