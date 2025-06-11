# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.autodifferentiation.internals.utils.polynomial import (
    poly_add,
    poly_derivative,
    poly_eval,
    poly_var_mul,
)

from thoad.typing.data import Shape, Indep, Notation, IDData


def sigmoid_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of sigma(x) evaluated at inv_sigmoid(tensor).
    All vectorized in PyTorch.
    """
    # Polynomial cache for Q_n(s). Q_n(s) is stored as a list of coefficients.
    _sigmoid_poly_cache: dict[int, list[float]] = {1: [1.0]}
    # Q_1(s) = 1

    def get_sigmoid_poly(n: int) -> list[float]:
        """
        Returns the list of coefficients of Q_n(s) such that
        sigma^{(n)}(x) = s(1-s)*Q_n(s).
        """
        if n in _sigmoid_poly_cache:
            return _sigmoid_poly_cache[n]

        # We build recursively from what we already have
        max_cached: int = max(_sigmoid_poly_cache.keys())
        for k in range(max_cached, n):
            Qk: list[float] = _sigmoid_poly_cache[k]  # Q_k
            dQk: list[float] = poly_derivative(Qk)  # Q_k'(s)

            # (1 - 2s)*Q_k(s)
            # polynomial (1) - 2*s => [1.0, -2.0]
            part1: list[float] = poly_var_mul(Qk, [1.0, -2.0])

            # s(1-s)*Q_k'(s)
            # s(1-s) => polynomial: [0.0, 1.0, -1.0]
            part_s1s: list[float] = [0.0, 1.0, -1.0]
            part2: list[float] = poly_var_mul(dQk, part_s1s)

            # Q_{k+1}(s) = part1 + part2
            Q_next: list[float] = poly_add(part1, part2)
            _sigmoid_poly_cache[k + 1] = Q_next

        return _sigmoid_poly_cache[n]

    if order == 0:
        # For consistency, the "0-th derivative" is the function itself
        return tensor

    # Q_n(s) in the form of coefficients
    Qn: list[float] = get_sigmoid_poly(n=order)

    # Evaluate Q_n(s) at s
    poly_val: Tensor = poly_eval(Qn, tensor)

    # Multiply by s*(1-s)
    return tensor * (1 - tensor) * poly_val


class SigmoidXBackward0(ContractiveFunction):

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
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["result"] = saved_result
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        ### Gather context
        result: Tensor = self._processed_context["result"]

        ### Instantiate differential
        differential: Tensor = sigmoid_derivate(tensor=result, order=order)

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
