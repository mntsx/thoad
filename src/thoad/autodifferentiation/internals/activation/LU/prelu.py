# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class PreluKernelXBackward0(ContractiveFunction):

    def check_shape(
        self,
        shape: Shape,
        indep: Indep,
    ) -> Tuple[Shape, Indep]:
        # extract saved inputs
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
        projected_indep: Indep = indep
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        saved_weight: Tensor = self._context["saved_weight"]
        aux: list[Union[None, int]] = list()
        for dim in projected_indep:
            if dim is None or saved_weight.shape[dim] == 1:
                aux.append(None)
            else:
                aux.append(dim)
        projected_indep = tuple(aux)
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract input and weight tensors
        saved_self: Tensor = self._grad_fn._saved_self
        saved_weight: Tensor = self._grad_fn._saved_weight
        # ensure proper dtype and device
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        saved_weight = saved_weight.to(dtype=self._dtype, device=self._device)
        # save raw context
        context: dict[str, Any] = dict()
        context["saved_self"] = saved_self
        context["saved_weight"] = saved_weight
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self: Tensor = self._context["saved_self"]
        saved_weight: Tensor = self._context["saved_weight"]
        # process context
        condition: Tensor = saved_self > 0
        squeezed_weight: Tensor = saved_weight.squeeze()
        extended_shape: Tuple[int, ...]
        extended_shape = tuple([s if d == 1 else 1 for d, s in enumerate(self._shape)])
        extended_weight: Tensor = squeezed_weight.view(extended_shape)
        expanded_weight: Tensor = extended_weight.expand(self._shape)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["condition"] = condition
        processed_context["expanded_weight"] = expanded_weight
        processed_context["input"] = saved_self
        processed_context["weight"] = saved_weight
        self._processed_context = processed_context
        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        condition: Tensor = self._processed_context["condition"]
        expanded_weight: Tensor = self._processed_context["expanded_weight"]

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate internal differential
        differential: Tensor = torch.where(
            condition=condition,
            input=t1,
            other=expanded_weight,
        )

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
        ### Gather context
        condition: Tensor = self._processed_context["condition"]
        input: Tensor = self._processed_context["input"]
        weight: Tensor = self._processed_context["weight"]

        ### Carry out instrumental operations
        # find the the weight dim
        weight_shape: Tuple[int, ...] = tuple(weight.shape)
        equal_sizes: list[bool] = [w == i for w, i in zip(weight_shape, input.shape)]
        ones: list[bool] = [s == 1 for s in input.shape]
        equal_non_ones: list[bool] = [es and o for es, o in zip(equal_sizes, ones)]
        assert any(equal_sizes) and equal_non_ones.count(True) <= 1
        weight_dim: int
        if equal_non_ones.count(True) == 1:
            weight_dim = equal_non_ones.index(True)
        else:
            weight_dim = equal_sizes.index(True)
        # build internal differential shape and corresponding einsum indices
        differential_shape: list[int] = list()
        external_indices: list[int] = list()
        internal_indices: list[int] = list()
        composed_indices: list[int] = list()
        counter: int = 0
        for dim, size in enumerate(self._shape):
            differential_shape.append(size)
            external_indices.append(counter)
            internal_indices.append(counter)
            counter += 1
            if dim == weight_dim:
                composed_indices.append(counter - 1)
            else:
                differential_shape.append(1)
                internal_indices.append(counter)
                composed_indices.append(counter)
                counter += 1

        # instantiate zero tensor
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate internal differential
        differential: Tensor = torch.where(condition=condition, input=t0, other=input)
        differential = differential.reshape(shape=differential_shape)

        ### Create einstein notation
        einstein_external: list[int] = external_indices
        einstein_internal: list[int] = internal_indices
        einstein_composed: list[list[int]] = [composed_indices]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([differential_shape])

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
            case _:
                (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
