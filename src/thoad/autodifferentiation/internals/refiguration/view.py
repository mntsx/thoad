# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class ViewXBackward0(ContractiveFunction):  # Reimplement as direct

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        assert math.prod(saved_self_sym_sizes) == math.prod(shape)
        projected_shape: Shape = shape
        projected_indep: Indep = tuple(None for _ in indep)
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        # save context
        context: dict[str, Any] = dict()
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = saved_self_sym_sizes
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        input_shape: Tuple[int, ...] = self._processed_context["input_shape"]

        ### Carry out instrumental operations
        input_ndim: int = len(input_shape)
        output_ndim: int = len(self._shape)
        external_range: Tuple[int, ...] = range(output_ndim)
        composed_range: Tuple[int, ...] = range(output_ndim, output_ndim + input_ndim)
        concat_shape: Tuple[int, ...] = (*self._shape, *input_shape)

        ### Instantiate differential
        differential: Tensor
        differential = torch.eye(n=math.prod(input_shape))
        differential = differential.reshape(concat_shape)

        ### Create einstein notation
        einstein_external: list[int] = list(external_range)
        einstein_internal: list[int] = [*external_range, *composed_range]
        einstein_composed: list[list[int]] = [list(composed_range)]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([(*self._shape, *input_shape)])

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
            case _:
                (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
