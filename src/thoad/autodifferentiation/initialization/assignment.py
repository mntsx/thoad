# Standard Library Dependencies
from typing import Type

# PyTorch dependencies
import torch

# Internal dependencies
from thoad.autodifferentiation.initialization.mapping import acquire_gfn_map
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction


class FunctionTranscoder:

    def __init__(self) -> None:
        self._index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
        self._index = acquire_gfn_map()
        return None

    @property
    def index(
        self,
    ) -> dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]:
        return self._index

    @index.setter
    def index(
        self,
        index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]],
    ) -> None:
        assert isinstance(index, dict)
        assert all([issubclass(v, ExtendedAutogradFunction) for v in index.values()])
        self._index = index
        return None

    def map(
        self,
        grad_fn: Type[torch.autograd.Function],
    ) -> Type[ExtendedAutogradFunction]:
        if type(grad_fn) not in self._index:
            raise NotImplementedError(f"{grad_fn.name()} is not supported.")
        return self._index[type(grad_fn)]

    def supports(self, grad_fn: torch.autograd.Function) -> bool:
        return type(grad_fn) in self._index
