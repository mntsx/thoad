# Standard Library dependencies
from typing import Any, Tuple, Union

# Pytorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.typing import IDData, Notation, Shape


def determnine_repeats(
    shape: Shape,
    raw_shape: Shape,
) -> Tuple[Union[None, int], ...]:
    """
    returns a tuple of len(shape) containing:
        int: number of times the corresponding dimension in raw shape has been repeated
        None: when the dimension has no corresponding peer in raw_shape
    """
    repeats: list[Union[None, int]] = [None for _ in shape]
    assert len(shape) >= len(raw_shape)
    for i, raw_size in enumerate(raw_shape[::-1]):
        j: int = len(shape) - i - 1
        assert shape[j] % raw_size == 0
        repeats[j] = shape[j] // raw_size
    return tuple(repeats)


def _bool(b: Any) -> bool:
    assert isinstance(b, bool)
    return b


def unbroadcast_IDData(
    ID_data: IDData,
    tensors_repeats: list[Tuple[Union[None, int], ...]],
) -> IDData:
    ### Definitions and checks
    assert ID_data[0] is not None
    assert ID_data[1] is not None
    derivative: Tensor = ID_data[0]
    einstein_notation: Notation = ID_data[1]
    new_internal_indices: list[int] = [*einstein_notation[0][1]]
    new_composed_indices: list[list[int]]
    new_composed_indices = [[*indices] for indices in einstein_notation[1]]
    assert len(new_composed_indices) == len(tensors_repeats)
    # determine wich indices require to be reduced / eliminated
    elimination_indices: set[int] = set()
    reduction_indices: dict[int, int] = dict()
    for i, repeats in enumerate(tensors_repeats):
        assert len(new_composed_indices[i]) == len(repeats)
        new_idx: int = max(*einstein_notation[0][0], *einstein_notation[0][1]) + 1
        for d, rep in enumerate(repeats):
            if rep is not None and rep != 1:
                idx: int = new_composed_indices[i][d]
                if idx not in reduction_indices:
                    reduction_indices[idx] = new_idx
                    new_internal_indices.append(new_idx)
                    new_idx += 1
                new_composed_indices[i][d] = reduction_indices[idx]
        for d, rep in enumerate(repeats):  # important to keep 2 separated loops
            if rep is None:
                elimination_indices.add(new_composed_indices[i].pop(d))
    remaining_indices: set[int]
    remaining_indices = set(idx for indices in new_composed_indices for idx in indices)
    remaining_indices |= set(einstein_notation[0][0])
    elimination_indices -= remaining_indices
    elimination_indices |= set(einstein_notation[0][1]) - remaining_indices
    # determine the positions that elimination indices occupy in internal indices
    elimination_dims: set[int] = set()
    for old_idx in elimination_indices:
        elimination_dims.add(new_internal_indices.index(old_idx))
    # eliminate / reduce internal indices
    new_internal_shape: list[int] = [*einstein_notation[2][0]]
    new_internal_indep: list[bool] = [_bool(b) for b in einstein_notation[2][1]]
    new_internal_shape.extend((1,) * len(reduction_indices))
    new_internal_indep.extend((True,) * len(reduction_indices))
    for d in sorted(elimination_dims)[::-1]:
        new_internal_indices.pop(d)
        new_internal_shape.pop(d)
        new_internal_indep.pop(d)
    # eliminate / reduce dimensions in derivative
    new_derivative: Tensor = derivative
    if len(reduction_indices) > 0:
        reduced_view: Tuple[int, ...]
        reduced_view = (*new_derivative.shape, *((1,) * len(reduction_indices)))
        new_derivative = new_derivative.view(size=reduced_view)
    if len(elimination_dims) > 0:
        new_derivative = new_derivative.sum(
            dim=tuple(elimination_dims),
            keepdim=False,
        )
    # construct new ID_data
    new_einstein_notation: Notation = [
        (tuple(einstein_notation[0][0]), tuple(new_internal_indices)),
        (*(tuple(indices) for indices in new_composed_indices),),
        (tuple(new_internal_shape), tuple(new_internal_indep)),
    ]

    return (new_derivative, new_einstein_notation)
