# Standard Library Dependencies
from typing import Tuple

# Internal dependencies
from thoad.typing import (
    Indep,
    Shape,
)


def find_variable_permutation(
    init: Tuple[int, ...],
    target: Tuple[int, ...],
) -> Tuple[int, ...]:
    n: int = len(init)
    assert n == len(target)
    # linear multiset check
    counts: dict[int, int] = {}
    for v in init:
        counts[v] = counts.get(v, 0) + 1
    for v in target:
        counts[v] = counts.get(v, 0) - 1
        if counts[v] == 0:
            del counts[v]
    assert not counts  # same multiset
    # map value -> stack of indices where it appears in 'init' (leftmost at the end)
    pos_in_init: dict[int, list[int]] = {}
    for i in range(n - 1, -1, -1):
        v = init[i]
        bucket = pos_in_init.get(v)
        if bucket is None:
            pos_in_init[v] = [i]
        else:
            bucket.append(i)
    # build permutation: for each target position, pick an index from init
    perm: list[int] = [pos_in_init[v].pop() for v in target]
    return tuple(perm)


def reverse_permutation(permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    assert all(i in permutation for i in range(len(permutation)))
    permutation_reverse: list[int] = [0] * len(permutation)
    for i, p in enumerate(permutation):
        permutation_reverse[p] = i
    return tuple(permutation_reverse)


def depermute_variables(
    variables: Tuple[int, ...],
    permutation: Tuple[int, ...],
) -> Tuple[int, ...]:
    permutation_reverse: Tuple[int, ...] = reverse_permutation(permutation=permutation)
    depermuted_variables: Tuple[int, ...] = tuple(
        variables[d] for d in permutation_reverse
    )
    return depermuted_variables


def depermute_metadata(
    variables: Tuple[int, ...],
    indeps: Tuple[Indep, ...],
    shapes: Tuple[Shape, ...],
    permutation: Tuple[int, ...],
) -> Tuple[Tuple[Shape, ...], Tuple[Indep, ...]]:
    # depermute variables
    depermuted_variables: Tuple[int, ...] = depermute_variables(
        variables=variables, permutation=permutation
    )
    # reduce variables to unique values
    unique_variables: Tuple[int, ...] = tuple(
        v for _, v in enumerate(dict.fromkeys(variables))
    )
    # reduce depermuted variables to unique values
    unique_depermuted_variables: Tuple[int, ...] = tuple(
        v for _, v in enumerate(dict.fromkeys(depermuted_variables))
    )
    # depermute indeps
    depermuted_indeps: list[Indep] = list()
    depermuted_shapes: list[Shape] = list()
    for v in unique_variables:
        idx: int = unique_depermuted_variables.index(v)
        depermuted_indeps.append(indeps[idx])
        depermuted_shapes.append(shapes[idx])

    return (tuple(depermuted_shapes), tuple(depermuted_indeps))
