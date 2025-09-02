# Standard Library Dependencies
from typing import Any, Sequence, Tuple, Union
from abc import ABC, abstractmethod

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing import (
    AutogradFunction,
    Indep,
    IDData,
    Shape,
    StaticEDData,
)


class ExtendedAutogradFunction(ABC):

    schwarz: bool = True

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
        self._method: Tuple[str, ...]  # Override
        self._nin: int  # Override
        self._nout: int  # Override
        self._context: Union[None, dict[str, Any]] = None
        self._processed_context: Union[None, dict[str, Any]] = None
        self._shape0: Shape  # define in check_shape

        # context initialization
        self._extract_context()

        return None

    @property
    def name(self) -> str:
        return "ExtendedAutogradFunction"

    @property
    def method(self) -> Tuple[str, ...]:
        return self._method

    @property
    def grad_fn(self) -> AutogradFunction:
        return self._grad_fn

    @property
    def context(self) -> dict[str, Any]:
        assert self._context is not None
        return self._context

    @abstractmethod
    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
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
        assert self._context is not None
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
        self._indeps: list[Union[None, Indep]]
        self._method: Tuple[str, ...] = ("direct",)
        return None

    def _check_transform(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        out_id: Tuple[Union[None, int], ...],
        inp_id: Tuple[Union[None, int], ...],
    ) -> None:
        # check length coherence between shapes and indeps
        assert len(indeps) == len(shapes)
        # check coherence between saved and received indeps
        order: int = len(indeps)
        unique_input_count: int = len(self._indeps)
        assert all(ii in range(unique_input_count) for ii in inp_id if ii is not None)
        # check alignment between output id & input id
        for i in range(order):
            oo: Union[None, int] = out_id[i]
            ii: Union[None, int] = inp_id[i]
            if ii is not None:
                _indep: Union[None, Indep] = self._indeps[ii]
                assert _indep is not None
                assert tuple(indeps[i]) == tuple(_indep)
                assert oo is not None
            else:
                assert oo is None
        # check the derivative shape matches the expected
        distributed_shapes: list[list[int]] = [list() for _ in shapes]
        for (
            i,
            (shape, indep),
        ) in enumerate(zip(shapes, indeps)):
            for j, sz in enumerate(shape):
                if j not in indep:
                    distributed_shapes[i].append(sz)
        indep_sizes: list[int] = list()
        for i, row in enumerate(zip(*indeps)):
            row_sizes: list[int]
            row_sizes = [shapes[j][d] for j, d in enumerate(row) if d is not None]
            indep_sizes.append(max([1, *row_sizes]))
        XX: int = derivative.shape[0]
        flat_distributed: list[int] = [ii for i in distributed_shapes for ii in i]
        expected_shape: Tuple[int, ...] = (XX, *indep_sizes, *flat_distributed)
        assert derivative.shape == expected_shape

        return None

    @abstractmethod
    def transform(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        out_id: Tuple[Union[None, int], ...],
        inp_id: Tuple[Union[None, int], ...],
    ) -> StaticEDData:
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
        self._method: Tuple[str, ...] = ("contractive",)
        return None

    def __getitem__(self, index: Tuple[int, Tuple[int, ...]]) -> IDData:
        assert isinstance(index[0], int)
        assert isinstance(index[1], Sequence)
        assert all(isinstance(i, int) for i in index[1])
        return self.compute_internal(out_id=index[0], inp_id=index[1])

    @abstractmethod
    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        self._process_context()
        # Computes internal derivative for a specific pair of:
        #   1 external variable
        #   N internal variables
