# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_shape
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class IndexSelectXBackward0(ContractiveFunction):

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
        dim: int = self._processed_context["dim"]
        index: Tensor = self._processed_context["index"]
        input_shape: Shape = self._processed_context["input_shape"]
        # initialize shape and indep projections
        index_size: int = max(1, index.shape[0])
        projected_shape: Shape = tuple(
            sz if d != dim else index_size for d, sz in enumerate(input_shape)
        )
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
        projected_indep = tuple(d if d != dim else None for d in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        saved_index: Tensor = getattr(self._grad_fn, "_saved_index")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        saved_index = saved_index.to(dtype=torch.long, device=self._device)
        getattr(saved_index, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_index"] = saved_index
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_dim: int = self._context["saved_dim"]
        saved_index: Tensor = self._context["saved_index"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dim: int = recalibrate_dim(dim=saved_dim, shape=saved_self_sym_sizes)
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = dim
        processed_context["index"] = saved_index
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dim: int = self._processed_context["dim"]
        index: Tensor = self._processed_context["index"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        external_indices: list[int] = list(range(ndim))
        composed_indices: list[int]
        composed_indices = [d if d != dim else ndim for d in external_indices]

        # create auxiliar tensors
        internal_shape: Tuple[int, ...] = (index.shape[0], input_shape[dim])
        base: Tensor = torch.zeros(
            size=internal_shape, dtype=self._dtype, device=self._device
        )
        scatter_indices = index.unsqueeze(1)
        t1: Tensor = torch.ones_like(
            input=scatter_indices, dtype=self._dtype, device=self._device
        )

        ### Instantiate derivative
        derivative: Tensor
        internal_indices: list[int] = [dim, ndim]
        derivative = base.scatter_(dim=1, index=scatter_indices, src=t1)
        derivative = derivative.to(device=self._device, dtype=self._dtype)
        derivative = derivative.view(size=internal_shape)

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(internal_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(internal_shape), (False, True)))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data
