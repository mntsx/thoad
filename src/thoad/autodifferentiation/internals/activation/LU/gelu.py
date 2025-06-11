# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


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
        H_n_minus_2: Tensor = (
            hermite_prob(x, n - 2) if (n - 2) >= 0 else torch.ones_like(x)
        )  # He_{n-2}(x)
        sign: Any = (-1) ** (n - 2)
        return sign * phi * (n * H_n_minus_2 - x * H_n_minus_1)


class GeluXBackward0(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        assert self._context is not None
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        # save as class attributes
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_approximate: str = self._grad_fn._saved_approximate
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict[str, Any] = dict()
        context["saved_approximate"] = saved_approximate
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_approximate: Tensor = self._context["saved_approximate"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["aproximate"] = saved_approximate
        processed_context["input"] = saved_self
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        input: Tensor = self._processed_context["input"]

        ### Carry out instrumental operations
        ndim: int = input.ndim

        ### Instantiate differential
        differential: Tensor = gelu_derivate(tensor=input, order=order)

        ### Create einstein notation
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim)) for _ in range(order)]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([list(self._shape)])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        if out_id == 0 and all(i == 0 for i in in_id):
            order: int = len(in_id)
            (differential, einstein_notation) = self._compute_internal_0(order=order)
        else:
            (differential, einstein_notation) = (None, None)

        return (differential, einstein_notation)
