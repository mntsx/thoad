# Pytorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


def logsubsigmoid_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of log(1 - sigmoid(x)) evaluated at x=tensor.

    Args:
      tensor: a Tensor of pre-activation values x
      order: derivative order n >= 0

    Returns:
      d^n/dx^n [ log(1 - sigmoid(x)) ] as a Tensor of the same shape
    """
    # compute s = sigmoid(x)
    s: Tensor = torch.sigmoid(tensor)

    # 0th derivative is the function itself
    if order == 0:
        return torch.log(1 - s)

    # For n >= 1 we build T_n(s) via T_1(s)= -s and
    # T_n(s) = s(1-s) * d/ds[T_{n-1}(s)].
    _cache: dict[int, list[float]] = {1: [0.0, -1.0]}  # T1(s) = -s

    def get_poly(n: int) -> list[float]:
        # Build up to T_n in the cache
        if n in _cache:
            return _cache[n]
        max_k: int = max(_cache.keys())
        for k in range(max_k, n):
            Tk: list[float] = _cache[k]
            dTk: list[float] = poly_derivative(Tk)
            # s(1-s) has coeffs [0,1,-1]
            Tnext: list[float] = poly_var_mul(dTk, [0.0, 1.0, -1.0])
            _cache[k + 1] = Tnext
        return _cache[n]

    # Evaluate the n-th polynomial at s
    coeffs: list[float] = get_poly(order)
    return poly_eval(coeffs, s)
