# Standard Library dependencies
import math

# Pytorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_add,
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


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
