# Pytorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.utils.polynomial import (
    poly_add,
    poly_derivative,
    poly_eval,
    poly_var_mul,
)


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
