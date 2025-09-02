# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape, Indep, Notation, IDData


class MvXBackward0(ContractiveFunction):

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
        assert len(shape) == 1
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape0 = projected_shape
        # adjust projected_indep to input
        match inp_id:
            case 0:
                projected_indep = tuple(0 if d == 0 else None for d in projected_indep)
            case 1:
                projected_indep = tuple(None for _ in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        saved_vec: Tensor = getattr(self._grad_fn, "_saved_vec")
        # ensure proper tensor configuration
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        if saved_vec is not None:
            saved_vec = saved_vec.to(dtype=self._dtype, device=self._device)
            getattr(saved_vec, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_vec"] = saved_vec
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_vec: Tensor = self._context["saved_vec"]
        # process context
        m: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        v: Tensor = denull_tensor(
            tensor=saved_vec, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["m"] = m
        processed_context["v"] = v
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        v: Tensor = self._processed_context["v"]

        ### Instantiate derivative
        derivative: Tensor = v

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (1,)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(v.shape), (True,)))

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m: Tensor = self._processed_context["m"]

        ### Instantiate derivative
        derivative: Tensor = m

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (0, 1)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1,),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(m.shape), (False, False)))

        return (derivative, einstein_notation)

    def _compute_internal_0_01(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m: Tensor = self._processed_context["m"]

        ### Instrumental operations
        dual_size: int = m.shape[1]
        internal_shape: Tuple[int, ...] = (m.shape[1],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((0, 1), (2,))
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (True, True)))

        return (derivative, einstein_notation)

    def _compute_internal_0_10(self) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        m: Tensor = self._processed_context["m"]

        ### Instrumental operations
        dual_size: int = m.shape[1]
        internal_shape: Tuple[int, ...] = (m.shape[1],) * 2

        ### Instantiate derivative
        derivative: Tensor
        derivative = torch.eye(dual_size, dtype=self._dtype, device=self._device)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = (0,)
        einstein_internal: Tuple[int, ...] = (1, 2)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = ((1,), (0, 2))
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
