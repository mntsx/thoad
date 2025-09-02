# Standard Library Dependencies
from typing import Type

# PyTorch dependencies
import torch

# Internal dependencies
from thoad.differentiation.initialization.mapping import acquire_gfn_map
from thoad.differentiation.internals.base import ExtendedAutogradFunction
from thoad.typing import AutogradFunction


class FunctionTranscoder:

    def __init__(self) -> None:
        self._index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
        self._index = acquire_gfn_map()
        return None

    @property
    def index(self) -> dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]:
        return self._index

    @index.setter
    def index(
        self,
        index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]],
    ) -> None:
        assert isinstance(index, dict)
        assert all(issubclass(v, ExtendedAutogradFunction) for v in index.values())
        self._index = index
        return None

    def map(
        self,
        gfn_type: Type[AutogradFunction],
    ) -> Type[ExtendedAutogradFunction]:
        if gfn_type not in self._index:
            raise NotImplementedError(f"{gfn_type.__name__} is not supported.")
        return self._index[gfn_type]

    def supports(self, grad_fn: AutogradFunction) -> bool:
        return type(grad_fn) in self._index
