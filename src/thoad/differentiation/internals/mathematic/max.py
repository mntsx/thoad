# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor, denull_shape
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class MaxXBackward0(ContractiveFunction):

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
        keepdim: bool = self._processed_context["keepdim"]
        indices: Tensor = self._processed_context["indices"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]
        output_shape: Tuple[int, ...] = self._processed_context["output_shape"]
        # initialize shape and indep projections
        projected_shape: Shape = output_shape
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=projected_shape,
                indep=projected_indep,
                expected_shape=projected_shape,
            )
        # save as class attributes
        self._shape0 = projected_shape
        if keepdim:
            projected_indep = tuple(d if d != dim else None for d in projected_indep)
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        saved_indices: Tensor = getattr(self._grad_fn, "_saved_indices")
        saved_keepdim: bool = getattr(self._grad_fn, "_saved_keepdim")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        if saved_indices is not None:
            saved_indices = saved_indices.to(dtype=torch.long, device=self._device)
            getattr(saved_indices, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_indices"] = saved_indices
        context["saved_keepdim"] = saved_keepdim
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
        saved_indices: Tensor = self._context["saved_indices"]
        saved_keepdim: bool = self._context["saved_keepdim"]
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        dim: int = recalibrate_dim(dim=saved_dim, shape=saved_self_sym_sizes)
        indices: Tensor = denull_tensor(
            tensor=saved_indices, dtype=torch.long, device=self._device
        )
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        output_shape: Tuple[int, ...] = tuple(
            sz if d != dim else 1
            for d, sz in enumerate(input_shape)
            if d != dim or saved_keepdim
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = dim
        processed_context["indices"] = indices
        processed_context["keepdim"] = saved_keepdim
        processed_context["input_shape"] = input_shape
        processed_context["output_shape"] = output_shape
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dim: int = self._processed_context["dim"]
        indices: Tensor = self._processed_context["indices"]
        keepdim: bool = self._processed_context["keepdim"]
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]
        output_shape: Tuple[int, ...] = self._processed_context["output_shape"]

        ### Carry out instrumental operations
        ndim: int = len(input_shape)
        # calculate indices for each stage
        external_indices: list[int] = [d for d in range(ndim) if d != dim or keepdim]
        internal_indices: list[int] = [*external_indices, ndim]
        composed_indices: list[int] = [d if d != dim else ndim for d in range(ndim)]
        internal_shape: Tuple[int, ...] = (*output_shape, input_shape[dim])
        internal_independencies: list[bool] = [True for _ in internal_indices]

        ### Instantiate derivative
        derivative: Tensor
        derivative = F.one_hot(input=indices, num_classes=input_shape[dim])
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
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data


class MaxXBackward1(ContractiveFunction):

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
        # initialize shape and indep projections
        projected_shape: Shape = tuple()
        projected_indep: Indep = tuple()
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_result: Tensor = getattr(self._grad_fn, "_saved_result")
        saved_self: Tensor = getattr(self._grad_fn, "_saved_self")
        # ensure proper tensor configuration
        if saved_result is not None:
            saved_result = saved_result.to(dtype=self._dtype, device=self._device)
            getattr(saved_result, "_fix_weakref")()
        if saved_self is not None:
            saved_self = saved_self.to(dtype=self._dtype, device=self._device)
            getattr(saved_self, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_result"] = saved_result
        context["saved_self"] = saved_self
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_result: Tensor = self._context["saved_result"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        output: Tensor = denull_tensor(
            tensor=saved_result, dtype=self._dtype, device=self._device
        )
        condition: Tensor = input >= output
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        weight: Tensor = torch.where(condition=condition, input=t1, other=t0)
        maximum_count: int = int(torch.sum(condition).item())
        assert maximum_count >= 1
        weight /= maximum_count
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input"] = input
        processed_context["output"] = output
        processed_context["weight"] = weight
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        weight: Tensor = self._processed_context["weight"]

        ### Carry out instrumental operations
        ndim: int = len(weight.shape)
        # calculate derivative shape and indices
        internal_shape: Tuple[int, ...] = tuple(weight.shape)
        external_indices: list[int] = list()
        internal_indices: list[int] = list(range(ndim))
        composed_indices: list[int] = list(range(ndim))

        ### Instantiate derivative
        derivative: Tensor = weight

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(external_indices)
        einstein_internal: Tuple[int, ...] = tuple(internal_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(composed_indices),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(True for _ in internal_shape))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data
