# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.typing import Shape, Indep, Notation, IDData


class AddXBackward0(ContractiveFunction):

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
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # project indep if necesary
        #   -> no need for projection, shape is returned unchanged
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_alpha: float = float(getattr(self._grad_fn, "_saved_alpha"))
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_alpha"] = saved_alpha
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_alpha: float = self._context["saved_alpha"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["alpha"] = saved_alpha
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        # ...

        ### Carry out instrumental operations
        ndim: int = len(self._shape0)
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1.sum()

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple()
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(), tuple()))

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        alpha: float = self._processed_context["alpha"]

        ### Carry out instrumental operations
        ndim: int = len(self._shape0)
        # calculate internal shape
        internal_shape: Tuple[int, ...] = tuple(self._shape0)
        internal_broadcastable_shape: Tuple[int, ...] = (1,) * ndim
        # create dummy tensor
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate tensor differently dependding on the value of alpha
        derivative: Tensor
        einstein_external: list[int]
        einstein_internal: list[int]
        einstein_composed: Tuple[Tuple[int, ...], ...]
        composed_shape: list[int]

        if alpha == 1.0:
            # instantiate derivative
            derivative = t1.sum()
            # create einstein notation
            einstein_external = list(range(ndim))
            einstein_internal = list()
            einstein_composed = (tuple(range(ndim)),)
            composed_shape = list()
        else:
            # instantiate derivative
            derivative = torch.full(
                fill_value=alpha,
                size=internal_broadcastable_shape,
                dtype=self._dtype,
                device=self._device,
            )
            # create einstein notation
            einstein_external = list(range(ndim))
            einstein_internal = list(range(ndim))
            einstein_composed = (tuple(range(ndim)),)
            composed_shape = list(self._shape0)

        einstein_notation: Notation = list()
        einstein_notation.append((tuple(einstein_external), tuple(einstein_internal)))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(composed_shape), tuple(True for _ in composed_shape))
        )

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

        return ID_data
