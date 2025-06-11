# Standard Library Dependencies
from typing import Iterable, Optional, Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.graph.structures import Node
from thoad.typing.data import EDData, Shape, Indep


class DifferentialGrid:
    def __init__(self) -> None:
        self._external_differentials: dict[Tuple[Node, ...], EDData] = {}

    @property
    def variables(self) -> set[Node]:
        variables: set[Node] = set()
        for key in self._external_differentials.keys():
            variables = variables.union(key)
        return variables

    def get(self, key: Tuple[Node, ...]) -> Optional[EDData]:
        substitute_data: Tuple[None, None, None] = (None, None, None)
        return self._external_differentials.get(key, substitute_data)

    def __getitem__(self, key: Tuple[Node, ...]) -> Optional[EDData]:
        return self.get(key)

    def set(
        self,
        key: Tuple[Node, ...],
        data: EDData,  # Tensor, Tuple[Shape, ...], Tuple[Indep, ...]
    ) -> None:

        ###  Key checks
        assert isinstance(key, Tuple)
        assert all(isinstance(N, Node) for N in key)
        # unique-valued key
        unique_key: Tuple[Node, ...] = tuple(dict.fromkeys(key))

        null: bool = data[0] is None

        ### Data checks
        assert isinstance(data, Tuple)
        assert len(data) == 3
        assert not null or all(d is None for d in data)
        if not null:
            # data[0] (Tensor)
            assert isinstance(data[0], Tensor)
            # data[1] (Shapes)
            assert isinstance(data[1], Tuple)
            assert len(unique_key) == len(data[1])
            assert all(isinstance(T, Tuple) for T in data[1])
            assert all(isinstance(i, int) for T in data[1] for i in T)
            # data[2] (Indeps)
            assert isinstance(data[2], Tuple)
            assert len(unique_key) == len(data[2])
            assert all(isinstance(T, Tuple) for T in data[2])
            assert all(isinstance(i, (type(None), int)) for T in data[2] for i in T)
            for i, indep in enumerate(data[2]):
                for dim in indep:
                    assert dim is None or dim in range(len(data[1][i]))
            assert len(set(len(i) for i in data[2])) == 1

        # Save data
        self._external_differentials[key] = data
        return None

    def __setitem__(self, key: Tuple[Node, ...], data: EDData) -> None:
        self.set(key, data)

    def remove(self, variables: Iterable[Node]) -> None:
        keys_to_remove: list[Tuple[Node, ...]] = [
            key
            for key in self._external_differentials.keys()
            if len(set(variables).intersection(set(key))) > 0
        ]
        for key in keys_to_remove:
            self._external_differentials.pop(key)


def initialize_differential(
    order: int,
    tensor: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> EDData:
    numel: int = max(1, tensor.numel())
    ndim: int = max(1, tensor.ndim)
    shape: Tuple[int, ...] = (1,) if tensor.ndim == 0 else tuple(tensor.shape)
    grad_shape: Tuple[int, ...] = (numel, *shape)
    # create differential tensor
    differential: Tensor
    assert order >= 1
    if order == 1:
        differential = torch.eye(numel, dtype=dtype, device=device)
        differential = differential.reshape(shape=grad_shape)
    else:
        differential = torch.zeros(size=grad_shape, dtype=dtype, device=device)
    # initialize shapes and indeps
    shapes: Tuple[Shape, ...] = (tuple(shape),)
    indeps: Tuple[Indep, ...] = (tuple(range(ndim)),)

    return (differential, shapes, indeps)
