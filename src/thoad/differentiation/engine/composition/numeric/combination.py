# Standard Library Dependencies
import itertools
from typing import Sequence, Tuple


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
    return list(itertools.product(elements, repeat=size))


def generate_permutation_keys(
    external_size: int,
    internal_size: int,
    max_order: int,
) -> list[Tuple[int, Tuple[int, ...]]]:
    """
    Build all keys for external→internal derivative permutations.

    Args:
        variables (Tuple[int, int, Tuple[int, ...]]):
            A triple (n_external, n_internal, var_map) where:
            - n_external is number of external variables,
            - n_internal is number of internal variables,
            - var_map maps composition positions to variable IDs.

    Returns:
        List[Tuple[int, Tuple[int, ...]]]:
            All pairs of (external_index, internal_variation), whereinternal_variation
            is any tuple of length 1..max_order over range(n_internal).
    """
    external_variables = range(external_size)
    internal_variables = range(internal_size)

    # Build all internal variations for orders 1..max_order in one iterator
    all_internal_keys: itertools.chain[Tuple[int, ...]] = itertools.chain.from_iterable(
        produce_variations(internal_variables, suborder)
        for suborder in range(1, max_order + 1)
    )

    # Build product of external × internal directly without storing internal_keys first
    return list(itertools.product(external_variables, all_internal_keys))
