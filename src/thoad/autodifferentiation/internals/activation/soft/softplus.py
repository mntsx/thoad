# Standard Library Dependencies
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.autodifferentiation.internals.utils.polynomial import (
    poly_add,
    poly_eval,
    poly_derivative,
    poly_var_mul,
)
from thoad.typing.data import Shape, Indep, Notation, IDData


def softplus_derivate(tensor: Tensor, beta: float, order: int) -> Tensor:
    """
    Returns the n-th derivative of softplus(x) = (1/β)*ln(1+exp(βx))
    ignoring the threshold. In particular, for n>=1:
      - For n = 1: f'(x) = σ(βx)
      - For n ≥ 2: f^(n)(x) = β^(n-1) σ(βx)(1-σ(βx)) Qₙ₋₁(σ(βx))
    where σ is the sigmoid function.
    """
    _softplus_poly_cache: dict[int, list[float]] = {}
    _softplus_poly_cache[1] = [1.0]

    def get_softplus_poly(n: int) -> list[float]:
        """
        Returns the coefficients of the polynomial Qₙ(s) used in the formula
        f^(n+1)(x) = βⁿ σ(βx)(1-σ(βx)) Qₙ(σ(βx))
        for softplus, with Q₁(s)=1 and the recurrence
        Qₙ₊₁(s) = (s(1-s)) Qₙ'(s) + (1-2s) Qₙ(s).
        """
        if n in _softplus_poly_cache:
            return _softplus_poly_cache[n]
        max_cached: int = max(_softplus_poly_cache.keys())
        for k in range(max_cached, n):
            Qk: list[float] = _softplus_poly_cache[k]
            dQk: list[float] = poly_derivative(Qk)
            # s*(1-s) as polynomial: 0 + 1·s + (-1)·s²  ==> [0.0, 1.0, -1.0]
            part1: list[float] = poly_var_mul(dQk, [0.0, 1.0, -1.0])
            # (1-2s) as polynomial: 1 + (-2)·s  ==> [1.0, -2.0]
            part2: list[float] = poly_var_mul(Qk, [1.0, -2.0])
            Q_next: list[float] = poly_add(part1, part2)
            _softplus_poly_cache[k + 1] = Q_next
        return _softplus_poly_cache[n]

    s: Tensor = torch.sigmoid(beta * tensor)
    if order == 1:
        return s
    else:
        Q: list[float] = get_softplus_poly(order - 1)
        poly_val: Tensor = poly_eval(Q, s)
        return (beta ** (order - 1)) * s * (1 - s) * poly_val


class SoftplusXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
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
        saved_beta: float = self._grad_fn._saved_beta
        saved_self: Tensor = self._grad_fn._saved_self
        saved_threshold: float = self._grad_fn._saved_threshold
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_beta"] = saved_beta
        context["saved_self"] = saved_self
        context["saved_threshold"] = saved_threshold
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_beta: float = self._context["saved_beta"]
        saved_self: float = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["beta"] = saved_beta
        processed_context["input"] = saved_self
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        ### Gather context
        beta: Tensor = self._processed_context["beta"]
        input: Tensor = self._processed_context["input"]

        ### Instantiate differential
        differential: Tensor = softplus_derivate(tensor=input, beta=beta, order=order)

        ### Create einstein notation
        ndim: int = input.ndim
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
