# Standard Library dependencies
import math

# Pytorch dependencies
import torch
from torch import Tensor


def cos_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Returns the n-th derivative of cos(x) evaluated at x = `tensor`.
    Uses the closed-form identity:
      d^n/dx^n [cos(x)] = cos(x + n*pi/2).
    """
    return torch.cos(tensor + order * math.pi / 2)
