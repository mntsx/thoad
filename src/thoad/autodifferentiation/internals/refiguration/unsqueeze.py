# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


class UnsqueezeXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_dim: int = self._context["saved_dim"]
        assert shape[saved_dim] == 1
        projected_shape: Shape = shape
        aux: list[Union[None, int]] = list(indep)
        if saved_dim in indep:
            aux[indep.index(saved_dim)] = None
        projected_indep: Indep = tuple(aux)
        # project indep if necesary
        #   -> no useful context info for proyection
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = self._grad_fn._saved_dim
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_dim: int = self._context["saved_dim"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = saved_dim
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        dim: int = self._processed_context["dim"]

        ### Carry out instrumental operations
        unsqueezed_shape: list[int] = list(range(len(self._shape)))
        unsqueezed_shape.pop(dim)
        dim_size: int = self._shape[dim]
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate differential
        differential: Tensor = t1  # (broadcasted to dim_size)

        ### Create einstein notation
        ndim: int = len(self._shape)
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = [dim]
        einstein_composed: list[list[int]] = [unsqueezed_shape]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([[dim_size]])

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
