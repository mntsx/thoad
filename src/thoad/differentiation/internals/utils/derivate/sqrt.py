# Standard Library dependencies
import math

# Pytorch dependencies
from torch import Tensor


def sqrt_derivate(
    output: Tensor,
    order: int,
) -> Tensor:
    """
    Compute the `order`-th derivative of sqrt(x) elementwise,
    given output = sqrt(x).  Returns a flattened Tensor of shape (N,).

    d^n/dx^n [sqrt(x)] = (1/2)(1/2-1)…(1/2-n+1) * x^(1/2 - n)
                        = coeff * (sqrt(x))^(1 - 2n)
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    y: Tensor = output.flatten()
    if order == 0:
        return y
    # falling-factorial for exponent=1/2:
    exponent: float = 0.5
    coeff: float = math.prod(exponent - i for i in range(order))
    # since x = y**2, x^(1/2 - order) = y^(2*(1/2 - order)) = y^(1 - 2*order)
    pow_exp: float = float(1 - 2 * order)
    y_pow: Tensor = y.pow(pow_exp)
    y_mul: Tensor = y_pow.mul_(coeff)
    return y_mul.view_as(other=output)
