# Pytorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


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
