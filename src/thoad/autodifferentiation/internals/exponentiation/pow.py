# Standard Library Dependencies
import math
from typing import Any, Tuple

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import adjust_indep
from thoad.autodifferentiation.internals.base import ContractiveFunction
from thoad.typing.data import Shape, Indep, Notation, IDData


def pow0_derivate(tensor: Tensor, exponent: float, order: int) -> Tensor:
    """
    Compute the `order`-th derivative of x**exponent, elementwise
    on `tensor`.  Returns a flattened Tensor of shape (tensor.numel(),).

    d^order/dx^order [x^exponent] =
      (exponent)(exponent-1)…(exponent-order+1) * x**(exponent-order)
    """
    if order < 0:
        raise ValueError("order must be non-negative")

    # flatten so we match the original behaviour
    x: Tensor = tensor.flatten()

    # trivial 0-th derivative
    if order == 0:
        return x.pow(exponent)

    # falling-factorial coefficient: exponent * (exponent-1) * … * (exponent-order+1)
    coeff: float = math.prod(exponent - i for i in range(order))

    return coeff * x.pow(exponent - order)


class PowXBackward0(ContractiveFunction):

    def check_shape(
        self,
        shape: Shape,
        indep: Indep,
    ) -> Tuple[Shape, Indep]:
        # extract input shape
        saved_self: Tensor = self._context["saved_self"]
        projected_shape: Shape = tuple(saved_self.shape)
        projected_indep: Indep = indep
        if shape != projected_shape:
            projected_indep = adjust_indep(
                shape=shape,
                indep=indep,
                projected_shape=projected_shape,
            )
        self._shape = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # save input
        saved_exponent: float = self._grad_fn._saved_exponent
        saved_self: Tensor = self._grad_fn._saved_self
        # ensure proper tensor configuration
        saved_self = saved_self.to(dtype=self._dtype, device=self._device)
        # save context
        context: dict["str", Any] = dict()
        context["saved_exponent"] = saved_exponent
        context["saved_self"] = saved_self
        self._context = context

        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        # load context
        saved_exponent: Tensor = self._context["saved_exponent"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        # ...
        # save processed context
        processed_context: dict["str", Any] = dict()
        processed_context["exponent"] = saved_exponent
        processed_context["input"] = saved_self
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> Tuple[Tensor, Notation]:
        ### Gather context
        exponent: Tensor = self._processed_context["exponent"]
        input: Tensor = self._processed_context["input"]

        ## Instantiate internal differential derivative via product rule
        differential: Tensor = pow0_derivate(
            tensor=input,
            exponent=exponent,
            order=order,
        )
        differential = differential.view(size=input.shape)

        ### Create einstein notation
        ndim: int = input.ndim
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim)) for _ in range(order)]
        einstein_notation: Notation = []
        einstein_notation.append(
            [
                einstein_external,
                einstein_internal,
            ]
        )
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

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


class PowXBackward1(ContractiveFunction):

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        raise NotImplementedError()
        assert self._context is not None
        # ...
        projected_shape: Shape = shape
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
        raise NotImplementedError()
        # extract info
        # ...
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        # ...
        self._context = context

        return None

    def _process_context(self) -> None:
        raise NotImplementedError()
        # checks
        assert self._shape is not None
        assert self._context is not None
        # load context
        # ...
        # process context
        # ...
        # save processed context
        processed_context: dict[str, Any] = dict()
        # ...
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert self._processed_context is not None
        ### Gather context
        # ...

        ### Carry out instrumental operations

        ### Instantiate differential
        differential: Tensor = None

        ### Create einstein notation
        ndim: int = None
        einstein_external: list[int] = list(range(ndim))
        einstein_internal: list[int] = list(range(ndim))
        einstein_composed: list[list[int]] = [list(range(ndim))]
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append(einstein_composed)
        einstein_notation.append([self._shape])

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
        match (out_id, in_id):
            case (0, (0,)):
                raise NotImplementedError()
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)
