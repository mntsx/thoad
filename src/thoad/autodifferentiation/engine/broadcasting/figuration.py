# Standard Library dependencies
import math
from typing import Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor


def infer_broadcast(shapes: list[Tuple[int, ...]]) -> Tuple[int, ...]:
    output_shape: list[int] = list()
    for shape in shapes:
        for i, sz in enumerate(shape[::-1]):
            if len(output_shape) <= i:
                output_shape.insert(0, sz)
            else:
                current_sz: int = output_shape[len(output_shape) - i - 1]
                assert sz == current_sz or sz == 1 or current_sz == 1
                output_shape[len(output_shape) - i - 1] = max(current_sz, sz)
    return tuple(output_shape)


def _calculate_shapes(
    first_size: int,
    variables: Tuple[int, ...],
    shapes: Sequence[Tuple[int, ...]],
    indeps: Sequence[Tuple[Union[None, int], ...]],
    indeps_squeezed: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute scattered and compact shapes for a tensor differential.

    Args:
        first_size (int): Size of the first tensor dimension.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Sequence[Tuple[int]]): Sizes for each tensor axis.
        indeps (Sequence[Tuple[Tint]]): Flags for independent dimensions.

    Returns:
        Tuple[Tuple[int, ...], Tuple[int, ...]]: scattered and compact shapes.
    """
    # precalculations
    independent_sizes: list[list[int]] = [1 for _ in enumerate(indeps[0])]
    for i, indep in enumerate(indeps):
        for j, dim in enumerate(indep):
            if dim is not None:
                independent_sizes[j] = max(independent_sizes[j], shapes[i][dim])
    if indeps_squeezed:
        independent_sizes = list()
    distributed_sizes: list[list[int]] = list()
    for v in variables:
        sublist: list[int] = list()
        for dim, size in enumerate(shapes[v]):
            if dim not in indeps[v]:
                sublist.append(size)
        distributed_sizes.append(sublist)

    lists: list[list[int]] = [[first_size], independent_sizes, *distributed_sizes]
    # compute scattered shape
    scattered_shape: Tuple[int, ...] = tuple([ss for s in lists for ss in s])
    # compute compact shape
    compact_distributed: list[int] = [math.prod(s) for s in distributed_sizes]
    compact_shape = Tuple[int, ...]
    compact_shape = (first_size, *independent_sizes, *compact_distributed)

    return (scattered_shape, compact_shape)


def compact_differential(
    differential: Tensor,
    variables: Tuple[int, ...],
    shapes: Sequence[Tuple[int, ...]],
    indeps: Sequence[Tuple[int, ...]],
    indeps_squeezed: bool,
) -> Tensor:
    """
    Reshape differential tensor from scattered to compact layout.

    Args:
        differential (Tensor): Input tensor with scattered shape.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Sequence[Tuple[int, ...]]): Sizes for each tensor axis.
        indeps (Sequence[Tuple[int, ...]]): Flags for independent dimensions.

    Returns:
        Tensor: Tensor reshaped to compact layout.
    """
    expected_shape: Tuple[int, ...]
    new_shape: Tuple[int, ...]
    expected_shape, new_shape = _calculate_shapes(
        first_size=differential.shape[0],
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=indeps_squeezed,
    )
    assert differential.shape == expected_shape
    compact_differential: Tensor = differential.reshape(shape=new_shape)
    return compact_differential


def scatter_differential(
    differential: Tensor,
    variables: Tuple[int, ...],
    shapes: Tuple[int, ...],
    indeps: Tuple[int, ...],
    indeps_squeezed: bool,
) -> Tensor:
    """
    Reshape differential tensor from compact to scattered layout.

    Args:
        differential (Tensor): Input tensor with compact shape.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Tuple[int, ...]): Sizes for each tensor axis.
        indeps (Tuple[int, ...]): Flags for independent dimensions.

    Returns:
        Tensor: Tensor reshaped to scattered layout.
    """
    expected_shape: Tuple[int, ...]
    new_shape: Tuple[int, ...]
    new_shape, expected_shape = _calculate_shapes(
        first_size=differential.shape[0],
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=indeps_squeezed,
    )
    assert differential.shape == expected_shape
    scattered_differential: Tensor = differential.reshape(shape=new_shape)
    return scattered_differential


def denull_differential(
    differential: Tensor,
    variables: Tuple[int, ...],
    shapes: Tuple[Tuple[int, ...], ...],
    indeps: Tuple[Tuple[Union[None, int], ...], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[
    Tensor,
    Tuple[Tuple[int, ...], ...],
    Tuple[Tuple[Union[None, int], ...], ...],
]:

    # constants
    XX: int = differential.shape[0]
    INDEPS: int = len(indeps[0])

    ### Obtain descriptive info about independent dimensions
    NULL_INDEPS: list[bool] = [True for _ in range(INDEPS)]
    INDEP_MAX_SHAPE: list[int] = [0 for _ in range(INDEPS)]
    for i, indep in enumerate(indeps):
        for j, dim in enumerate(indep):
            if dim is not None:
                NULL_INDEPS[j] = False
                INDEP_MAX_SHAPE[j] = max(INDEP_MAX_SHAPE[j], shapes[i][dim])
            else:
                INDEP_MAX_SHAPE[j] = max(INDEP_MAX_SHAPE[j], 1)

    ### Inital checks
    assert all([var in range(len(shapes)) for var in variables])
    # check that every variable shares the same independent dimensions
    assert len(set(len(indep) for indep in indeps)) == 1
    for j, step in enumerate(zip(*indeps)):
        assert all([isinstance(i, (int, type(None))) for i in step])
        size: set[int] = {shapes[i][ii] for i, ii in enumerate(step) if ii is not None}
        assert len(size) <= 1
        if len(size) == 1:
            sz: int = size.pop()
            assert sz == differential.shape[1 + j]
    # check coherence in number of dimensions
    distributed_ndim: int = 0
    for v in variables:
        distributed_ndim += len(shapes[v]) - INDEPS + indeps[v].count(None)
    assert differential.ndim == (1 + INDEPS + distributed_ndim)
    # check coherence in size of dimensions
    expected_shape, _ = _calculate_shapes(
        first_size=XX,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert differential.shape == expected_shape
    assert XX > 0
    assert 0 not in differential.shape[1 : (1 + INDEPS)]

    ### Eliminate size zero dimensions
    # Determine nullity contition of shapes
    null_shapes: bool = False  # list[bool] = [False for _ in shape]
    denullfied_shapes: list[list[int]] = [list(shape) for shape in shapes]
    denullfied_indeps: list[list[int]] = [list(indep) for indep in indeps]
    for i, shape in enumerate(shapes):
        for dim, size in enumerate(shape):
            if size == 0:
                null_shapes: bool = True
                denullfied_shapes[i][j] = 1
                if j in indeps[i]:
                    idx: int = indeps[i].index(j)
                    denullfied_indeps[idx] = None
                    INDEP_MAX_SHAPE[idx] = 1

    _shapes: Tuple[Tuple[int, ...], ...]
    _shapes = tuple(tuple(s) for s in denullfied_shapes)
    _indeps: Tuple[Tuple[Union[None, int], ...]]
    _indeps = tuple(tuple(i) for i in denullfied_indeps)

    # Correct differential (if necessary)
    denulled_differential: Tensor
    if null_shapes:
        new_shape: Tuple[int, ...]
        new_shape, _ = _calculate_shapes(
            first_size=XX,
            variables=variables,
            shapes=_shapes,
            indeps=_indeps,
            indeps_squeezed=False,
        )
        denulled_differential = torch.zeros(
            size=new_shape,
            dtype=dtype,
            device=device,
        )
    else:
        denulled_differential = differential

    return (denulled_differential, _shapes, _indeps)
