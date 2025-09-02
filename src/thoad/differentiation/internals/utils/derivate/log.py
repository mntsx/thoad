# Standard Library dependencies
import math

# Pytorch dependencies
import torch
from torch import Tensor


def log_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of log(x) evaluated at x = `tensor`.

    Formula:
      d^n/dx^n [log(x)] =
         log(x),                       n = 0
         (-1)^(n-1) * (n-1)! / x^n,    n >= 1
    """
    if order == 0:
        return torch.log(tensor)  # 0th derivative is log(x) itself
    else:
        # Use the closed-form for n >= 1
        sign: int = (-1) ** (order - 1)
        factorial_term: int = math.factorial(order - 1)
        return sign * factorial_term / (tensor**order)