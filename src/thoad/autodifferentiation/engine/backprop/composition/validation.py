# Standard Library Dependencies
from typing import Sequence, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.backprop.composition.combination import (
    generate_permutation_keys,
)
from thoad.typing.data import Shape, Indep, Notation


def check_variables(variables: Tuple[int, int, Tuple[int, ...]]) -> None:

    assert isinstance(variables, Sequence)
    assert len(variables) == 3
    assert isinstance(variables[0], int)
    assert isinstance(variables[1], int)
    assert isinstance(variables[2], Sequence)
    assert len(variables[2]) > 0
    assert all([isinstance(var, int) for var in variables[2]])

    return None


def check_external_differentials(
    variables: Tuple[int, int, Tuple[int, ...]],
    external_differentials: dict[Tuple[int, ...], Union[None, Tensor]],
    external_shapes: dict[int, Shape],
    external_indeps: dict[int, Indep],
) -> None:

    unique_indeps: set[Indep] = set(indep for indep in external_indeps.values())
    assert len(set(len(ui) for ui in unique_indeps)) == 1
    assert isinstance(external_differentials, dict)
    for key, val in external_differentials.items():
        # key cheks
        assert isinstance(key, Sequence)
        assert all([isinstance(var, int) for var in key])
        assert all([var >= 0 and var <= (variables[0] - 1) for var in key])
        # get indeps and shapes
        shapes: list[Shape] = [external_shapes[v] for v in key]
        indeps: list[Indep] = [external_indeps[v] for v in key]
        # value cheks
        assert isinstance(val, (type(None), Tensor))
        if val is not None:
            distributed_shapes: list[list[int]] = [list() for _ in shapes]
            for (
                i,
                (shape, indep),
            ) in enumerate(zip(shapes, indeps)):
                for j, sz in enumerate(shape):
                    if j not in indep:
                        distributed_shapes[i].append(sz)
            # check coherence between indeps
            indep_sizes: list[int] = list()
            for i, row in enumerate(zip(*indeps)):
                row_sizes: list[int]
                row_sizes = [shapes[j][d] for j, d in enumerate(row) if d is not None]
                indep_sizes.append(max([1, *row_sizes]))
            XX: int = val.shape[0]
            flat_distributed: list[int] = [ii for i in distributed_shapes for ii in i]
            expected_shape: Tuple[int, ...] = (XX, *indep_sizes, *flat_distributed)
            assert val.shape == expected_shape

    return None


def _check_keys_appearance(
    variables: Tuple[int, int, Tuple[int, ...]], keys: set[Tuple[int, Tuple[int, ...]]]
) -> None:

    keys: list[Tuple[int, Tuple[int, ...]]]
    keys = generate_permutation_keys(
        external_size=variables[0],
        internal_size=variables[1],
        max_order=len(variables[2]),
    )
    for key in keys:
        assert key in keys, key

    return None


def check_internal_differentials(
    variables: Tuple[int, int, Tuple[int, ...]],
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]],
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]],
) -> None:

    _check_keys_appearance(variables=variables, keys=set(internal_differentials.keys()))
    assert isinstance(internal_differentials, dict)
    for key, val in internal_differentials.items():
        # key cheks
        assert isinstance(key, Tuple)
        assert len(key) == 2
        assert isinstance(key[0], int)
        assert key[0] >= 0 and key[0] <= (variables[0] - 1)
        assert isinstance(key[1], Sequence)
        assert all(isinstance(i, int) for i in key[1])
        assert all([var >= 0 and var <= (variables[1] - 1) for var in key[1]])
        # value cheks
        assert isinstance(val, (type(None), Tensor))
        # notations checks
        assert key in einstein_notations
        if val is not None:
            notation: Notation = einstein_notations[key]
            assert notation is not None
            assert isinstance(notation, Sequence)
            assert len(notation) == 3, (len(notation), notation)
            assert len(notation[0]) == 2
            assert len(notation[1]) >= 1
            assert len(notation[2]) == 1
            assert all(isinstance(n, Sequence) for n in notation)
            assert all(isinstance(m, Sequence) for n in notation[0:2] for m in n)
            assert all(isinstance(i, int) for n in notation[0:2] for m in n for i in m)
            assert all(isinstance(i, int) for i in notation[2][0])
            assert len(val.shape) == len(notation[0][1]), (val.shape, notation[0][1])

    return None
