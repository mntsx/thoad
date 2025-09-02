# Standard Library dependencies
import math

# Pytorch dependencies
import torch
from torch import Tensor


def sin_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of sin(x) evaluated at x = `tensor`.
    Uses the closed-form identity:
      d^n/dx^n [sin(x)] = sin(x + n*pi/2).
    """
    return torch.sin(tensor + order * math.pi / 2)
