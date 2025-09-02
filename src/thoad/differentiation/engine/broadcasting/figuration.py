# Standard Library dependencies
import math
from typing import Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing import Indep, Shape


def construct_nd_identity(
    n: int,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Constructs an n^dims-sized identity shaped as [n, n, ..., n] (ndim times).
    For example, construct_nd_identity(2, 2) -> shape [2,2].
    """
    size: int = n**ndim
    IDn: Tensor = torch.zeros((size,), dtype=dtype, device=device)
    idx: Tensor = torch.arange(0, n)
    factor: int = 0
    for i in range(ndim):
        factor += n**i
    idx *= factor
    IDn[idx] = 1

    return IDn.view(*([n] * ndim))


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


def calculate_shapes(
    first_size: int,
    variables: Tuple[int, ...],
    shapes: Sequence[Tuple[int, ...]],
    indeps: Sequence[Tuple[Union[None, int], ...]],
    indeps_squeezed: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute scattered and compact shapes for a tensor derivative.

    Args:
        first_size (int): Size of the first tensor dimension.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Sequence[Tuple[int]]): Sizes for each tensor axis.
        indeps (Sequence[Tuple[Tint]]): Flags for independent dimensions.

    Returns:
        Tuple[Tuple[int, ...], Tuple[int, ...]]: scattered and compact shapes.
    """
    # precalculations

    shapes = tuple(shape for i, shape in enumerate(shapes) if i in variables)
    indeps = tuple(indep for i, indep in enumerate(indeps) if i in variables)
    independent_sizes: list[int] = [1 for _ in enumerate(indeps[0])]
    for i, indep in enumerate(indeps):
        for j, dim in enumerate(indep):
            if dim is not None:
                independent_sizes[j] = max(independent_sizes[j], shapes[i][dim])
    if indeps_squeezed:
        assert all(sz == 1 for sz in independent_sizes)
        independent_sizes = []
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
    compact_shape: Tuple[int, ...]
    compact_shape = (first_size, *independent_sizes, *compact_distributed)

    return (scattered_shape, compact_shape)


def compact_derivative(
    derivative: Tensor,
    variables: Tuple[int, ...],
    shapes: Sequence[Tuple[int, ...]],
    indeps: Sequence[Tuple[Union[None, int], ...]],
    indeps_squeezed: bool,
) -> Tensor:
    """
    Reshape derivative tensor from scattered to compact layout.

    Args:
        derivative (Tensor): Input tensor with scattered shape.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Sequence[Tuple[int, ...]]): Sizes for each tensor axis.
        indeps (Sequence[Tuple[int, ...]]): Flags for independent dimensions.

    Returns:
        Tensor: Tensor reshaped to compact layout.
    """
    expected_shape: Tuple[int, ...]
    new_shape: Tuple[int, ...]
    expected_shape, new_shape = calculate_shapes(
        first_size=derivative.shape[0],
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=indeps_squeezed,
    )
    assert derivative.shape == expected_shape
    compact_derivative: Tensor = derivative.reshape(shape=new_shape)
    return compact_derivative


def scatter_derivative(
    derivative: Tensor,
    variables: Tuple[int, ...],
    shapes: Tuple[Shape, ...],
    indeps: Tuple[Indep, ...],
    indeps_squeezed: bool,
) -> Tensor:
    """
    Reshape derivative tensor from compact to scattered layout.

    Args:
        derivative (Tensor): Input tensor with compact shape.
        variables (Tuple[int, ...]): Indices of variable dimensions.
        shapes (Tuple[int, ...]): Sizes for each tensor axis.
        indeps (Tuple[int, ...]): Flags for independent dimensions.

    Returns:
        Tensor: Tensor reshaped to scattered layout.
    """
    expected_shape: Tuple[int, ...]
    new_shape: Tuple[int, ...]
    new_shape, expected_shape = calculate_shapes(
        first_size=derivative.shape[0],
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=indeps_squeezed,
    )
    assert derivative.shape == expected_shape
    scattered_derivative: Tensor = derivative.reshape(shape=new_shape)
    return scattered_derivative


def denull_derivative(
    derivative: Tensor,
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
    primal_size: int = derivative.shape[0]
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
    assert all(var in range(len(shapes)) for var in variables), (len(shapes), variables)
    # check that every variable shares the same independent dimensions
    assert len(set(len(indep) for indep in indeps)) == 1
    for j, step in enumerate(zip(*indeps)):
        assert all(isinstance(i, (int, type(None))) for i in step)
        size: set[int] = {shapes[i][ii] for i, ii in enumerate(step) if ii is not None}
        assert len(size) <= 1
        if len(size) == 1:
            sz: int = size.pop()
            assert sz == derivative.shape[1 + j]
    # check coherence in number of dimensions
    distributed_ndim: int = 0
    for v in variables:
        distributed_ndim += len(shapes[v]) - INDEPS + indeps[v].count(None)
    assert derivative.ndim == (1 + INDEPS + distributed_ndim)
    # check coherence in size of dimensions
    expected_view: Tuple[int, ...]
    expected_view, _ = calculate_shapes(
        first_size=primal_size,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative.shape == expected_view, (derivative.shape, expected_view)
    assert 0 not in derivative.shape[1 : (1 + INDEPS)]

    ### Augment length 0 shapes (to length 1 size 1)
    augmented_shapes: list[Shape] = [shape for shape in shapes]
    augmented_view: list[int] = list(expected_view)
    insertion_index: int = 1 + len(INDEP_MAX_SHAPE)
    for v in variables:
        if len(shapes[v]) > 0:
            insertion_index += len(shapes[v])
        else:
            augmented_shapes[v] = (1,)
            augmented_view.insert(insertion_index, 1)
            insertion_index += 1
    shapes = tuple(augmented_shapes)
    derivative = derivative.view(size=augmented_view)

    ### Bound size 0 dimensions (to size 1)
    # Determine nullity contition of shapes
    null_shapes: bool = False  # list[bool] = [False for _ in shape]
    bounded_shapes: list[list[int]] = [list(shape) for shape in shapes]
    bounded_indeps: list[list[Union[None, int]]] = [list(indep) for indep in indeps]
    for i, shape in enumerate(shapes):
        for dim, sz in enumerate(shape):
            if sz == 0:
                null_shapes: bool = True
                bounded_shapes[i][dim] = 1
                if dim in indeps[i]:
                    idx: int = indeps[i].index(dim)
                    bounded_indeps[i][idx] = None
                    INDEP_MAX_SHAPE[idx] = 1
    _shapes: Tuple[Tuple[int, ...], ...]
    _shapes = tuple(tuple(S) for S in bounded_shapes)
    _indeps: Tuple[Tuple[Union[None, int], ...], ...]
    _indeps = tuple(tuple(I) for I in bounded_indeps)
    # Correct derivative (if necessary)
    denulled_derivative: Tensor
    if null_shapes:
        new_view: Tuple[int, ...]
        new_view, _ = calculate_shapes(
            first_size=primal_size,
            variables=variables,
            shapes=_shapes,
            indeps=_indeps,
            indeps_squeezed=False,
        )
        denulled_derivative = torch.zeros(
            size=new_view,
            dtype=dtype,
            device=device,
        )
    else:
        denulled_derivative = derivative

    return (denulled_derivative, _shapes, _indeps)
