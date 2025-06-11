# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.autodifferentiation.internals.utils.polynomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)
from thoad.typing.data import Shape, Indep, Notation, IDData


def tan_derivate(tensor: Tensor, n: int) -> Tensor:
    """
    Returns the n-th derivative of tan(x) evaluated at x,
    expressed as a polynomial in tan(x).
    """

    _tan_poly_cache: dict[int, list[float]] = {}
    _tan_poly_cache[0] = [0.0, 1.0]  # T0(t) = 0 + 1*t

    def get_tan_poly(n: int) -> list[float]:
        if n in _tan_poly_cache:
            return _tan_poly_cache[n]
        max_cached: int = max(_tan_poly_cache.keys())
        for k in range(max_cached, n):
            Tk: list[float] = _tan_poly_cache[k]
            dTk: list[float] = poly_derivative(Tk)
            # (1+t^2) as a polynomial is [1.0, 0.0, 1.0]
            T_next: list[float] = poly_var_mul(dTk, [1.0, 0.0, 1.0])
            _tan_poly_cache[k + 1] = T_next
        return _tan_poly_cache[n]

    if n == 0:
        return tensor
    Tn: list[float] = get_tan_poly(n)

    return poly_eval(Tn, tensor)


class TanXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_result: Tensor = self._context["saved_result"]
        projected_shape: Shape = tuple(saved_result.shape)
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
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_result: Tensor = self._grad_fn._saved_result
        # ensure proper tensor configuration
        saved_result = saved_result.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_result"] = saved_result
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_result: Tensor = self._context["saved_result"]
        # process context
        result: Tensor = saved_result
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["result"] = result
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        result: Tensor = self._processed_context["result"]

        ### Carry out instrumental operations
        # ...

        ### Instantiate differential
        differential: Tensor
        differential = tan_derivate(tensor=result, n=order)

        ### Create einstein notation
        ndim: int = result.ndim
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim)) for _ in range(order)]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

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
        if out_id == 0 and all(i == 0 for i in in_id):
            order: int = len(in_id)
            (differential, einstein_notation) = self._compute_internal_0(order=order)
        else:
            (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
