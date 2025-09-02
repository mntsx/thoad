# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class UnsqueezeXBackward0(ContractiveFunction):

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
        assert shape[dim] == 1
        self._processed_context["dim"] = recalibrate_dim(dim=dim, shape=shape)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        aux: list[Union[None, int]] = list(indep)
        if dim in indep:
            aux[indep.index(dim)] = None
        projected_indep: Indep = tuple(aux)
        # project indep if necesary
        #   -> no useful context info for proyection
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: int = getattr(self._grad_fn, "_saved_dim")
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
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

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        dim: int = self._processed_context["dim"]

        ### Carry out instrumental operations
        unsqueezed_shape: list[int] = list(range(len(self._shape0)))
        unsqueezed_shape.pop(dim)
        dim_size: int = self._shape0[dim]
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1  # (broadcasted to dim_size)

        ### Create einstein notation
        ndim: int = len(self._shape0)
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = (dim,)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(unsqueezed_shape),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(((dim_size,), (True,)))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data
