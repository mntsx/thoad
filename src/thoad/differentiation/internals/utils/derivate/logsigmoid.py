# Pytorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


def logsigmoid_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of log(sigmoid(x)) evaluated at x=tensor.

    Args:
      tensor: a Tensor of pre-activation values x
      order: derivative order n >= 0

    Returns:
      d^n/dx^n [ log(sigmoid(x)) ] as a Tensor of the same shape
    """
    # First compute s = sigmoid(x)
    s: Tensor = torch.sigmoid(tensor)

    # 0th derivative is log(s)
    if order == 0:
        return torch.log(s)

    # For n>=1 we build R_n(s) via R_1(s)=1−s and R_n = s(1−s)·d/ds[R_{n−1}].
    # We'll cache the polynomials as lists of coefficients:
    _cache: dict[int, list[float]] = {1: [1.0, -1.0]}  # R1(s) = 1 - s

    def get_poly(n: int) -> list[float]:
        # Build up to R_n in the cache
        if n in _cache:
            return _cache[n]
        max_k: int = max(_cache)
        for k in range(max_k, n):
            Rk: list[float] = _cache[k]
            dRk: list[float] = poly_derivative(Rk)
            # Multiply dRk(s) by s(1-s) whose coefficients are [0,1,-1]
            R_next: list[float] = poly_var_mul(dRk, [0.0, 1.0, -1.0])
            _cache[k + 1] = R_next
        return _cache[n]

    # Get coefficients for R_n(s), evaluate at s, and return
    coeffs: list[float] = get_poly(order)
    return poly_eval(coeffs, s)
