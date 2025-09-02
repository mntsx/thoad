# Standard Library dependencies
import math

# Pytorch dependencies
import torch
from torch import Tensor


def hermite_prob(x: Tensor, n: int) -> Tensor:
    """
    Compute the probabilists' Hermite polynomial He_n(x) defined by:
         He_0(x) = 1,
         He_1(x) = x,
         He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x).
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        H0: Tensor = torch.ones_like(x)
        H1: Tensor = x
        for k in range(1, n):
            H2: Tensor = x * H1 - k * H0
            H0, H1 = H1, H2
        return H1


def gelu_derivate(tensor: Tensor, order: int) -> Tensor:
    """
    Compute the n-th derivative of GELU(x)= x * Phi(x) with
      Phi(x) = 0.5*(1+erf(x/sqrt(2)))  and  phi(x)= Phi'(x) = exp(-x^2/2)/sqrt(2pi).

    For order==0: returns GELU(x).
    For order==1: returns g'(x)= Phi(x) + x*phi(x).
    For order>=2: returns
         g^(n)(x)= (-1)^(n-2)*phi(x)*[ n*He_{n-2}(x) - x*He_{n-1}(x) ].
    """

    x: Tensor = tensor
    sqrt2: float = math.sqrt(2.0)
    sqrt_2pi: float = math.sqrt(2 * math.pi)
    phi: Tensor = torch.exp(-0.5 * x**2) / sqrt_2pi
    Phi: Tensor = 0.5 * (1 + torch.erf(x / sqrt2))

    if order == 0:
        return x * Phi
    elif order == 1:
        return Phi + x * phi
    else:
        # For order n>=2:
        n: int = order  # alias for clarity
        H_n_minus_1: Tensor = hermite_prob(x, n - 1)  # He_{n-1}(x)
        H_n_minus_2: Tensor  # He_{n-2}(x)
        H_n_minus_2 = hermite_prob(x, n - 2) if (n - 2) >= 0 else torch.ones_like(x)
        sign: int = (-1) ** (n - 2)
        return sign * phi * (n * H_n_minus_2 - x * H_n_minus_1)
