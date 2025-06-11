# Standard Library Dependencies
import itertools
from typing import Iterator, Sequence, Tuple


def produce_variations(elements: Sequence[int], size: int) -> list[Tuple[int, ...]]:
    """
    Generate all fixed-length tuples of given elements with repetition.

    Args:
        elements (Sequence[int]): Sequence of values to draw from.
        size (int): Length of each variation tuple.

    Returns:
        List[Tuple[int, ...]]: All ordered tuples of length `size`, where each position
        is any element from `elements`.
    """
    assert isinstance(elements, Sequence)
    variations: Iterator[Tuple[int, ...]] = itertools.product(elements, repeat=size)
    result: list[Tuple[int, ...]] = [tuple(item) for item in variations]
    return result


def generate_permutation_keys(
    external_size: int,
    internal_size: int,
    max_order: int,
) -> list[Tuple[int, Tuple[int, ...]]]:
    """
    Build all keys for external→internal differential permutations.

    Args:
        variables (Tuple[int, int, Tuple[int, ...]]):
            A triple (n_external, n_internal, var_map) where:
            - n_external is number of external variables,
            - n_internal is number of internal variables,
            - var_map maps composition positions to variable IDs.

    Returns:
        List[Tuple[int, Tuple[int, ...]]]:
            All pairs of (external_index, internal_variation), where internal_variation
            is any tuple of length 1..max_order over range(n_internal).
    """

    external_variables: Tuple[int, ...] = tuple(range(external_size))
    internal_variables: Tuple[int, ...] = tuple(range(internal_size))
    internal_keys: list[Tuple[int, ...]] = list()
    for suborder in range(1, max_order + 1):
        variations: list[Tuple[int, ...]] = produce_variations(
            elements=internal_variables, size=suborder
        )
        internal_keys.extend(variations)
    combinations: list[Iterator]
    combinations = itertools.product(external_variables, internal_keys)
    keys: list[Tuple[int, Tuple[int, ...]]]
    keys = [tuple(combination) for combination in combinations]

    return keys
