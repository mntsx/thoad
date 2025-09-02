# Standard Library Dependencies
import itertools
from typing import Any, Sequence, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.composition.numeric.combination import (
    generate_permutation_keys,
)
from thoad.differentiation.engine.control.symmetry import reverse_permutation
from thoad.typing import Shape, Indep, Notation, VPerm


def _populated(obj: Union[None, Any]) -> Any:
    assert obj is not None
    return obj


def check_variables(variables: Tuple[int, int, Tuple[int, ...]]) -> None:

    assert isinstance(variables, Sequence)
    assert len(variables) == 3
    assert isinstance(variables[0], int)
    assert isinstance(variables[1], int)
    assert isinstance(variables[2], Sequence)
    assert len(variables[2]) > 0
    assert all(isinstance(var, int) for var in variables[2])

    return None


def check_external_derivatives(
    variables: Tuple[int, int, Tuple[int, ...]],
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]],
    external_shapes: dict[int, Union[None, Shape]],
    external_indeps: dict[int, Union[None, Indep]],
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]],
) -> None:

    # precompute independent size once
    unique_indeps: set[Union[None, Indep]] = set(external_indeps.values())
    unique_indeps -= {None}
    sizes: set[int] = {len(_populated(ui)) for ui in unique_indeps}
    assert len(sizes) == 1
    independent_size: int = next(iter(sizes))
    assert isinstance(external_derivatives, dict)
    n_external: int = variables[0]
    for key, val in external_derivatives.items():
        # key cheks
        assert isinstance(key, Sequence)
        for var in key:
            assert isinstance(var, int)
            assert 0 <= var < n_external
        # get indeps and shapes
        _shapes: list[Union[None, Shape]] = [external_shapes[v] for v in key]
        _indeps: list[Union[None, Indep]] = [external_indeps[v] for v in key]
        vperm: Union[None, VPerm] = external_vperms[key]  # asdf
        # value cheks
        assert isinstance(val, (type(None), Tensor))
        if val is None:
            continue
        shapes: list[Shape] = [_populated(S) for S in _shapes]
        indeps: list[Indep] = [_populated(I) for I in _indeps]
        assert vperm is not None
        # compute distributed shapes (dims not bound to indeps)
        distributed_shapes: list[list[int]] = list()
        reverse_vperm: Tuple[int, ...] = reverse_permutation(permutation=vperm)
        pairs: list[Tuple[Shape, Indep]] = list(zip(shapes, indeps))
        for shape, indep in (pairs[d] for d in reverse_vperm):
            row: list[int] = [sz for j, sz in enumerate(shape) if j not in indep]
            distributed_shapes.append(row)
        # check coherence between indeps (avoid zip(*) materialization)
        indep_sizes: list[int] = []
        for d in range(independent_size):
            row_sizes: list[int] = []
            for j, indep in enumerate(indeps):
                idx: Union[None, int] = indep[d]
                if idx is not None:
                    row_sizes.append(shapes[j][idx])
            indep_sizes.append(max([1, *row_sizes]))
        XX: int = val.shape[0]
        # flatten distributed_shapes without nested loops
        flat_distributed: list[int]
        flat_distributed = list(itertools.chain.from_iterable(distributed_shapes))
        expected_view: Tuple[int, ...] = (XX, *indep_sizes, *flat_distributed)
        assert val.shape == expected_view, (val.shape, expected_view)

    return None


def _check_keys_appearance(
    variables: Tuple[int, int, Tuple[int, ...]], keys: set[Tuple[int, Tuple[int, ...]]]
) -> None:

    generated_keys: list[Tuple[int, Tuple[int, ...]]]
    generated_keys = generate_permutation_keys(
        external_size=variables[0],
        internal_size=variables[1],
        max_order=len(variables[2]),
    )
    for key in keys:
        assert key in generated_keys, key

    return None


def check_internal_derivatives(
    variables: Tuple[int, int, Tuple[int, ...]],
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]],
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]],
) -> None:

    _check_keys_appearance(variables=variables, keys=set(internal_derivatives.keys()))

    assert isinstance(internal_derivatives, dict)
    for key, val in internal_derivatives.items():
        # key cheks
        assert isinstance(key, Tuple)
        assert len(key) == 2
        assert isinstance(key[0], int)
        assert key[0] >= 0 and key[0] <= (variables[0] - 1)
        assert isinstance(key[1], Sequence)
        assert all(isinstance(i, int) for i in key[1])
        assert all(var >= 0 and var <= (variables[1] - 1) for var in key[1])
        # value cheks
        assert isinstance(val, (type(None), Tensor))
        # notations checks
        assert key in einstein_notations
        if val is not None:
            notation: Union[None, Notation] = einstein_notations[key]
            assert notation is not None
            assert isinstance(notation, Sequence)
            assert len(notation) == 3, (len(notation), notation)
            assert len(notation[0]) == 2
            assert len(notation[1]) >= 1
            assert len(notation[2]) == 2, notation[2]
            assert all(isinstance(n, Sequence) for n in notation)
            assert all(isinstance(m, Sequence) for n in notation[0:2] for m in n)
            assert all(isinstance(i, int) for n in notation[0:2] for m in n for i in m)
            assert all(isinstance(i, int) for i in notation[2][0])
            assert all(isinstance(i, bool) for i in notation[2][1])
            assert len(val.shape) == len(notation[0][1]), (val.shape, notation[0][1])

    return None
