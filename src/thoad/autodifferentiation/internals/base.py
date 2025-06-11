# Standard Library Dependencies
from typing import Any, Sequence, Tuple, Union
from abc import ABC, abstractmethod

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing.data import AutogradFunction, Shape, Indep, IDData


class ExtendedAutogradFunction(ABC):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:

        # plain attributes
        self._grad_fn: AutogradFunction = grad_fn
        self._order: int = order
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device

        # processed attributes
        self._method: list[str]  # Override
        self._nin: int  # Override
        self._nout: int  # Override
        self._context: Union[None, dict[str, Any]] = None
        self._processed_context: Union[None, dict[str, Any]] = None
        self._shape: Union[None, Tuple[Shape, ...]] = None  # define in check_shape

        # context initialization
        self._extract_context()

        return None

    def __name__(self) -> str:
        return "ExtendedAutogradFunction"

    @property
    def method(self) -> list[str]:
        return self._method

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    @abstractmethod
    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        pass
        #   1. Runs checks on shape:
        #      - some XAFs may have an exact defined shape
        #      - some XAFs may have no requirement on shape
        #      - soma XAFs may have some (but not complete) requirements on shape
        #   2. Calculate closest feasible shape and returns it

    @abstractmethod
    def _extract_context(self) -> None:
        pass
        #   1. Obtains all possible context data
        #   2. Returns each context data object (or None, if not available)

    @abstractmethod
    def _process_context(self) -> None:
        assert self._shape is not None
        #   Process context to extract data for computation of internals
        #   Returns resulting objects & tuple indicating which inputs require grad


class DirectFunction(ExtendedAutogradFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._method: list[str] = ["direct"]
        return None

    @abstractmethod
    def transform(
        self,
        differential: Tensor,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]:
        pass  # TODO


class ContractiveFunction(ExtendedAutogradFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._method: list[str] = ["contractive"]
        return None

    def __getitem__(self, index: Tuple[int, Tuple[int, ...]]) -> IDData:
        assert isinstance(index[0], int)
        assert isinstance(index[1], Sequence)
        assert all(isinstance(i, int) for i in index[1])
        return self.compute_internal(out_id=index[0], in_id=index[1])

    @abstractmethod
    def compute_internal(self, out_id: int, in_id: Tuple[int, ...]) -> IDData:
        self._process_context()
        # Computes internal differential for a specific pair of:
        #   1 external variable
        #   N internal variables


class DirectContractiveFunction(DirectFunction, ContractiveFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._method: list[str] = ["direct", "contractive"]
        return None


class ConvContractiveFunction(ContractiveFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._method: list[str] = ["convolutional", "contractive"]
        return None

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        # Kernel internals would be computed here
        pass

    def compute_sliding_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        # Feautures internals would be computed here
        pass
