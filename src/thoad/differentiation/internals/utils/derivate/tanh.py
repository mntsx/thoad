# Pytorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


def tanh_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of tanh(x) evaluated at inv_tanh(tensor).
    """

    # We store T_n(t) as a list of coefficients
    _tanh_poly_cache: dict[int, list[float]] = {}
    # Define T_1(t) = 1 - t^2 => [1.0, 0.0, -1.0] (i.e. 1 + 0*t - 1*t^2)
    _tanh_poly_cache[1] = [1.0, 0.0, -1.0]

    def get_tanh_poly(n: int) -> list[float]:
        """
        Returns the list of coefficients of T_n(t) where
        T_n(t) = d^n/dx^n [tanh(x)], represented in t = tanh(x).
        """
        if n in _tanh_poly_cache:
            return _tanh_poly_cache[n]

        max_cached: int = max(_tanh_poly_cache.keys())
        for k in range(max_cached, n):
            Tk: list[float] = _tanh_poly_cache[k]
            dTk: list[float] = poly_derivative(Tk)  # d/dt [T_k(t)]

            # (1 - t^2) * dTk
            # (1 - t^2) => [1.0, 0.0, -1.0]
            T_next: list[float] = poly_var_mul(dTk, [1.0, 0.0, -1.0])
            _tanh_poly_cache[k + 1] = T_next

        return _tanh_poly_cache[n]

    if order == 0:
        # "0-th derivative" => tanh(x) itself
        return tensor

    # Obtain T_n(t)
    Tn: list[float] = get_tanh_poly(order)
    # Evaluate at t
    return poly_eval(Tn, tensor)
