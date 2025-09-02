# Standard Library dependencies
import itertools
import math
import warnings
from typing import Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.figuration import calculate_shapes
from thoad.typing import Indep, Shape, StaticEDData
import thoad.config as config


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


def unify_indeps(
    indeps: Sequence[Indep],
    shapes_ndims: Sequence[int],
    inclusive: bool,
) -> Indep:
    assert isinstance(indeps, Sequence)
    assert all(isinstance(indep, Tuple) for indep in indeps)
    assert all(isinstance(dim, (type(None), int)) for indep in indeps for dim in indep)
    assert len(set(len(indep) for indep in indeps)) == 1

    ### Rules all indeps must follow
    #   1. indep offset must be constant along all indep values
    #   2. indep values must not reference any dimensions not present in all indeps
    #      i.e. all values must be lower than offset

    # obtain offset (gap) of each indep
    gaps: Tuple[int, ...] = tuple(ndim - min(shapes_ndims) for ndim in shapes_ndims)
    # merge indep values following inclusive or exclusive criteria
    unified_indep: list[Union[None, int]] = [None for _ in indeps[0]]
    for i, row in enumerate(zip(*indeps)):
        if inclusive or None not in row:
            for j, dim in enumerate(row):
                if dim is not None:
                    if unified_indep[i] is None:
                        unified_indep[i] = dim - gaps[j]
                    else:
                        assert unified_indep[i] == dim - gaps[j]

    return tuple(unified_indep)


def shape_align_indep(
    shape: Tuple[int, ...],
    indep: Tuple[Union[None, int], ...],
    expected_shape: Tuple[int, ...],
) -> Tuple[Union[None, int], ...]:
    projected_indep: Indep = indep
    if shape != expected_shape:
        if len(shape) == len(expected_shape):
            if math.prod(shape) == math.prod(expected_shape):
                # same numel -> permute
                permutation: Tuple[int, ...] = _find_shape_permutation(
                    shape=shape,
                    target=expected_shape,
                )
                _, projected_indep = _permute_shape(
                    shape=shape,
                    indep=indep,
                    permutation=permutation,
                )
            else:
                # different numel -> assert repeat
                aux: list[Union[None, int]] = [i for i in indep]
                for i, (s, p) in enumerate(zip(shape, expected_shape)):
                    assert math.gcd(s, p) == p
                    if i in indep:
                        aux[indep.index(i)] = aux[indep.index(i)] if s == p else None
                projected_indep = tuple(aux)
        else:
            # truncate left dimensions (torch only broadcasts in the left)
            drop: int = len(shape) - len(expected_shape)
            assert tuple(shape[drop:]) == tuple(expected_shape)
            trucated_indep: list[Union[None, int]] = [None for _ in indep]
            for i, dim in enumerate(indep):
                if dim is not None and dim >= drop:
                    trucated_indep[i] = dim - drop
            projected_indep = tuple(trucated_indep)

    return projected_indep


def _permute_shape(
    shape: Tuple[int, ...],
    indep: Tuple[Union[int, None], ...],
    permutation: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[Union[int, None], ...]]:
    """
    Permute a shape and update independent‐axis indices.

    Args:
        shape (Tuple[int, ...]): Original dimensions.
        indep (Tuple[Optional[int], ...]): Indices of independent axes.
        permutation (Sequence[int]): New axis ordering.

    Returns:
        Tuple[int, ...]: permuted shape
        Tuple[Optional[int], ...]: updated indep indices
    """
    assert isinstance(permutation, Tuple)
    assert all(isinstance(i, int) for i in permutation)
    assert set(permutation) == set(range(len(permutation)))
    assert len(shape) == len(permutation)
    # permute shape and adjust indep
    permuted_shape: list[int] = [-1 for _ in permutation]
    adjusted_indep: list[Union[None, int]] = [None for _ in indep]
    for i, p in enumerate(permutation):
        permuted_shape[i] = shape[p]
        if p in indep:
            adjusted_indep[indep.index(p)] = i
    assert -1 not in permuted_shape
    return (tuple(permuted_shape), tuple(adjusted_indep))


def shape_broadcastable(shape: Tuple[int, ...], target: Tuple[int, ...]) -> bool:
    """
    Check if two shapes are broadcastable via gcd rule.

    Args:
        shape (tuple[int, ...]): The original shape.
        target (tuple[int, ...]): The target shape to broadcast to.

    Returns:
        bool: True if for every dimension i,
            gcd(shape[i], target[i]) == target[i], else False.
    """
    broadcastable: bool = True
    broadcastable &= len(shape) <= len(target)
    pruned_target: Tuple[int, ...] = target[max(0, len(target) - len(shape)) :]
    broadcastable &= all(math.gcd(s, t) == s for s, t in zip(shape, pruned_target))
    return broadcastable


def _assert_permutation_options(options: int) -> None:
    """
    Validate the count of possible permutations and warn or error.

    Args:
        options (int): Number of valid permutations found.

    Raises:
        ValueError: If no valid permutations exist (options == 0).
        RuntimeWarning: If multiple ambiguous permutations exist (options > 1).
    """
    match options:
        case 0:
            raise ValueError(
                "Engine found an intractable combination of permutation "
                "and broadcasting. Consider being more explicit in "
                "the arrangement of dimensions."
            )

        case _:
            warnings.warn(
                "Engine found an ambiguous combination of permutation "
                "and broadcasting. This can lead to errors in partials "
                "computations. Consider being more explicit in the "
                "arrangement of dimensions.",
                RuntimeWarning,
            )
    return None


def _shape_distance(
    shape: Tuple[int, ...],
    target: Tuple[int, ...],
) -> Tuple[int, int]:
    """
    Compute a distance score between two shapes for permutation ranking.

    Score attends to 2 criteria of similitude to target:
      1. the fewer swaps the better
      2. swaps in the last dimensions are better

    Args:
        shape (tuple[int, ...]): The current shape tuple.
        target (tuple[int, ...]): The target shape tuple.

    Returns:
        tuple[int, int]: A pair (movement, position_sum) used for scoring.
    """
    movement: int = sum(1 for s, a in zip(shape, target) if s != a)
    positions: list[int] = [i for i, (s, a) in enumerate(zip(shape, target)) if s != a]
    return (movement, sum(positions))


def _solve_permutation(
    shape: Tuple[int, ...], target: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Determine index permutation mapping `shape` to `target`.

    Each element in `target` must appear in `shape`. Returns a tuple p
    such that target[i] == shape[p[i]] for all i.

    Args:
        shape (tuple[int, ...]): Original tuple of values.
        target (tuple[int, ...]): Desired ordering of same values.

    Returns:
        tuple[int, ...]: Indices mapping `shape` to `target`.
    """
    assert len(shape) == len(target)
    # Build a mapping from each value to a list of its positions in `shape`
    value_to_indices: dict[int, list[int]] = {}
    for idx, val in enumerate(shape):
        value_to_indices.setdefault(val, []).append(idx)
    # for each value in `target`, pop the next available index
    permutation: list[int] = []
    for val in target:
        # pop(0) retrieves the earliest unused index
        permutation.append(value_to_indices[val].pop(0))

    return tuple(permutation)


def _find_shape_permutation(
    shape: Tuple[int, ...], target: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Find the optimal index permutation mapping `shape` to `target`.

    Considers all permutations of `shape` whose dimensions multiply to the same
    total as `target`, scores each by (1) the number of moved axes and (2)
    the sum of their positions (favoring moves toward the end), and selects
    the best.  Finally computes the index mapping from the original `shape`
    to this best‐matched ordering.

    Args:
        shape (Tuple[int, ...]): Original dimension sizes.
        target (Tuple[int, ...]): Desired dimension sizes; must have the same
            total product as some permutation of `shape`.

    Returns:
        Tuple[int, ...]: A permutation `p` such that
            `target[i] == shape[p[i]]` for all i.
    """
    assert len(shape) == len(target)
    permuted_shapes: list[Tuple[int, ...]]
    permuted_shapes = list(itertools.permutations(shape))

    def _target_permutable(shape: Tuple[int, ...]) -> bool:
        return math.prod(shape) == math.prod(target)

    permuted_shapes = list(filter(_target_permutable, permuted_shapes))
    _assert_permutation_options(options=len(permuted_shapes))

    def _score_to_target(shape: Tuple[int, ...]) -> Tuple[int, int]:
        return _shape_distance(shape=shape, target=target)

    best_shape: Tuple[int, ...] = min(permuted_shapes, key=_score_to_target)
    best_permutation: Tuple[int, ...]
    best_permutation = _solve_permutation(shape=shape, target=best_shape)

    return best_permutation


class SymIndex:

    def __init__(self) -> None:
        self._id: Union[None, int] = None
        self._size: Union[None, int] = None

    def assert_size(self, size: int, override: bool = False) -> None:
        if self._size is None or (override and 1 in (size, self._size)):
            current_size: int = 1 if self._size is None else self._size
            size = max(size, current_size) if override else size
            self._size = size
        else:
            assert self._size == size, (self._size, size)
        return None

    @property
    def src(self) -> "SymIndex":
        return self

    @property
    def id(self) -> int:
        assert self._id is not None
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = value
        return None

    @property
    def size(self) -> int:
        assert self._size is not None
        return self._size


class LinkedSymIndex(SymIndex):
    """
    A SymIndex that shares its size with another SymIndex.

    LinkedSymIndex wraps an existing SymIndex and ensures that any size
    assertions apply to the linked index rather than to this instance.
    """

    def __init__(self, sym_index: SymIndex) -> None:
        """
        Initialize a LinkedSymIndex.

        Args:
            sym_index (SymIndex):
                The SymIndex to link to. All size assertions and queries will be
                forwarded to this linked index.
        """
        super().__init__()
        self._link: SymIndex = sym_index

    def assert_size(self, size: int, override: bool = False) -> None:
        """
        Assert that the linked SymIndex has the given size.

        If the linked SymIndex has no size yet, set it to `size`. Otherwise, verify
        that its existing size matches `size`.

        Args:
            size (int): The dimension size to assert on the linked index.
        """
        if self._link._size is None or (override and 1 in (size, self._link._size)):
            if override:
                link_size: Union[None, int] = self._link._size
                assert link_size is not None
                size = max(size, link_size)
            self._link._size = size
        else:
            assert self._link._size == size, (self._link._size, size)
        return None

    @property
    def src(self) -> "SymIndex":
        return self._link

    @property
    def size(self) -> int:
        """
        Get the size of the linked SymIndex.

        Returns:
            int or None: The size of the linked SymIndex, or None if not set.
        """
        size: Union[None, int] = self._link._size
        assert size is not None
        return size


def _numerize_indices(nested_indices: list[list[list[SymIndex]]]) -> None:
    flat_indices: set[SymIndex] = set()
    for sub in nested_indices:
        for subsub in sub:
            for sym_idx in subsub:
                flat_indices.add(sym_idx)
    for i, sym_idx in enumerate(flat_indices):
        sym_idx.id = i
    return None


def _einsum(
    in_tensor: Tensor,
    nested_indices: list[list[list[SymIndex]]],
) -> Tensor:

    def _insert_numerized(syms: list[SymIndex]) -> Tuple[int, ...]:
        return tuple(sym.id for sym in syms)

    def _flatten(seq: Sequence[Sequence[SymIndex]]) -> Tuple[SymIndex, ...]:
        return tuple(x for sub in seq for x in sub)

    ### Sort einsum args
    einsum_args: list[Union[Tensor, Tuple[int, ...]]] = [in_tensor]
    external_syms: list[SymIndex]
    external_syms = [*_flatten(nested_indices[0]), *_flatten(nested_indices[1])]
    assert len(external_syms) == in_tensor.ndim
    einsum_args.append(_insert_numerized(external_syms))
    for syms in nested_indices[2]:
        ndim: int = len(syms)
        if ndim > 1:
            assert len(set([sym.size for sym in syms])) == 1
            identity: Tensor = construct_nd_identity(
                n=syms[0].size,
                ndim=ndim,
                dtype=in_tensor.dtype,
                device=in_tensor.device,
            )
            einsum_args.append(identity)
            einsum_args.append(_insert_numerized(syms))
    output_syms: list[SymIndex]
    output_syms = [*_flatten(nested_indices[3]), *_flatten(nested_indices[4])]
    einsum_args.append(_insert_numerized(output_syms))

    ### Compute composed derivative
    # Note. torch einsum does not reallocate memory if not necesary (e.g. "ij->ij")
    result: Tensor = torch.einsum(*einsum_args)

    return result


def _num(i: Union[None, int]) -> int:
    assert isinstance(i, int)
    return i


def unbroadcast_derivative(
    derivative: Tensor,
    variables: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_shapes: Sequence[Shape],
) -> Tuple[
    Tensor,
    Tuple[Shape, ...],
    Tuple[Indep, ...],
]:

    assert len(set((len(indep) for indep in indeps))) == 1
    assert all(sz in (1, *S) for (S, XS) in zip(shapes, expected_shapes) for sz in XS)
    INDEP_SIZE: int = set((len(indep) for indep in indeps)).pop()

    ### Determine which dimensions must be kept
    # create registries to mark wich dimensions must be kept in output data
    independent_markers: list[list[bool]]
    independent_markers = [[True for _ in indep] for indep in indeps]
    distributed_markers_a: list[list[bool]]  # dimensions to sum to size 1
    distributed_markers_a = [[False for _ in shape] for shape in shapes]
    distributed_markers_b: list[list[bool]]  # dimensions to sum and remove
    distributed_markers_b = [[True for _ in shape] for shape in shapes]
    # mark dimensions
    for i, (shape, xshape, indep) in enumerate(zip(shapes, expected_shapes, indeps)):
        # mark dimensions to be summed and removed
        gap: int = len(shape) - len(xshape)
        counts: dict[int, int] = {sz: xshape.count(sz) for sz in (*shape, *xshape)}
        diffs: dict[int, int] = {sz: shape.count(sz) - counts[sz] for sz in counts}
        for d, sz in enumerate(shape[::-1]):
            mirror_d: int = len(shape) - d - 1
            xmirror_d: int = len(xshape) - d - 1
            skip: bool = xmirror_d >= 0 and xshape[xmirror_d] == 1 and diffs[sz] > 0
            prox: Tuple[int, ...] = tuple(range(max(0, xmirror_d)))
            statics: int = sum(shape[gap + j] == sz and xshape[j] == sz for j in prox)
            if sz in counts and counts[sz] - statics > 0 and not skip:
                distributed_markers_b[i][mirror_d] = False
                if mirror_d in indep:
                    independent_markers[i][indep.index(mirror_d)] = False
                counts[sz] -= 1
            diffs[sz] -= int(skip)
        # mark dimensions to be summed to size 1
        xones: int = min(xshape.count(1) - shape.count(1), len(xshape))
        for d, _ in enumerate(shape[::-1]):
            mirror_d: int = len(shape) - d - 1
            if distributed_markers_b[i][mirror_d] and xones > 0:
                distributed_markers_b[i][mirror_d] = False
                distributed_markers_a[i][mirror_d] = True
                xones -= 1
    # create registry to mark which independent positions arent "used" by any variable
    shared_independent_marker: list[bool]
    shared_independent_marker = [all(iT) for iT in zip(*independent_markers)]

    ### Construct new derivative data (Tensor, shapes & indeps) removing dimensions
    # update shapes
    unbroadcasted_shapes: list[Shape] = list()
    for shape, marker_a, marker_b in zip(
        shapes, distributed_markers_a, distributed_markers_b
    ):
        unbroadcasted_shapes.append(
            tuple(
                1 if ka else sz
                for sz, ka, kb in zip(shape, marker_a, marker_b)
                if not kb
            )
        )
    # update indeps
    unbroadcasted_indeps: list[Indep] = list()
    for i, (indep, marker) in enumerate(zip(indeps, independent_markers)):
        dist_marker: list[bool] = distributed_markers_b[i]
        unbroadcasted_indep: Indep = tuple(
            None if k else _num(d) - sum(dist_marker[:d]) for d, k in zip(indep, marker)
        )
        unbroadcasted_indeps.append(unbroadcasted_indep)
    # update tensor
    unbroadcasted_derivative: Tensor = derivative
    dims: list[int] = [1 + i for i, k in enumerate(shared_independent_marker) if k]
    acc: int = INDEP_SIZE + 1
    for v in variables:
        indep: Indep = indeps[v]
        marker: list[bool] = distributed_markers_a[v]
        dims.extend(
            [
                acc + sum(d not in indep for d in range(d))
                for d, k in enumerate(marker)
                if k and d not in indep
            ]
        )
        acc += len(marker) - INDEP_SIZE + indep.count(None)
    if len(dims) > 0:
        unbroadcasted_derivative = unbroadcasted_derivative.sum(
            dim=dims,
            keepdim=True,
        )
    dims.clear()
    acc: int = INDEP_SIZE + 1
    for v in variables:
        indep: Indep = indeps[v]
        marker: list[bool] = distributed_markers_b[v]
        dims.extend(
            [
                acc + sum(d not in indep for d in range(d))
                for d, k in enumerate(marker)
                if k and d not in indep
            ]
        )
        acc += len(marker) - INDEP_SIZE + indep.count(None)
    if len(dims) > 0:
        unbroadcasted_derivative = unbroadcasted_derivative.sum(
            dim=dims,
            keepdim=False,
        )

    view: Tuple[int, ...] = calculate_shapes(
        first_size=derivative.shape[0],
        variables=tuple(variables),
        shapes=tuple(unbroadcasted_shapes),
        indeps=tuple(unbroadcasted_indeps),
        indeps_squeezed=False,
    )[0]

    unbroadcasted_derivative = unbroadcasted_derivative.reshape(shape=view)

    return (
        unbroadcasted_derivative,
        tuple(unbroadcasted_shapes),
        tuple(unbroadcasted_indeps),
    )


def permute_derivative(
    derivative: Tensor,
    variables: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_shapes: Sequence[Shape],
) -> Tuple[
    Tensor,
    Tuple[Shape, ...],
    Tuple[Indep, ...],
]:

    assert len(set((len(indep) for indep in indeps))) == 1
    INDEP_SIZE: int = set((len(indep) for indep in indeps)).pop()
    assert all(len(s) == len(xs) for s, xs in zip(shapes, expected_shapes))

    ### Calculate permutations for each shape
    permutations: list[Tuple[int, ...]] = list()
    for shape, xshape in zip(shapes, expected_shapes):
        if shape == xshape:
            permutations.append(tuple(range(len(shape))))
        else:
            permutations.append(
                _find_shape_permutation(
                    shape=shape,
                    target=xshape,
                )
            )

    ### Permute dimensions of each variable shape
    # update indeps
    permuted_indeps: list[Indep] = list()
    for indep, permutation in zip(indeps, permutations):
        permuted_indeps.append(
            tuple(None if d is None else permutation.index(d) for d in indep)
        )
    # update tensor
    view: list[int] = list(range(INDEP_SIZE + 1))
    for v in variables:
        indep: Indep = indeps[v]
        permutation: Indep = permutations[v]
        acc: int = len(view)
        skips: list[bool] = [d in indep for d in range(len(shapes[v]))]
        pruned_permutation: list[int]
        pruned_permutation = [d - sum(skips[:d]) for d in permutation if not skips[d]]
        for d in pruned_permutation:
            view.append(acc + d)
    permuted_derivative: Tensor = derivative
    if tuple(view) != tuple(range(len(view))):
        permuted_derivative = permuted_derivative.permute(dims=view)

    return (permuted_derivative, tuple(expected_shapes), tuple(permuted_indeps))


def _distribute_derivative(
    derivative: Tensor,
    variables: Sequence[int],
    variable_perm: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_indeps: Sequence[Tuple[Union[None, int], ...]],
    keepdim: bool,
) -> Tensor:

    assert len(set((len(indep) for indep in indeps))) == 1

    ### Initialize input and output independent sym indices arrays
    inp_independent_syms: list[SymIndex] = list()
    out_independent_syms: list[Union[None, SymIndex]] = list()
    for indepT, xindepT in zip(zip(*indeps), zip(*expected_indeps)):
        sym: SymIndex = SymIndex()
        enum: enumerate = enumerate(indepT)
        size_candidates: set[int] = set(shapes[i][d] for i, d in enum if d is not None)
        assert len(size_candidates) in (0, 1)
        size: int = size_candidates.pop() if len(size_candidates) == 1 else 1
        sym.assert_size(size=size)
        inp_independent_syms.append(sym)
        out_independent_syms.append(
            sym if any(d is not None for d in xindepT) else None
        )

    ### Initialize input and output distributed sym indices arrays
    inp_distributed_syms: list[list[Union[None, SymIndex]]] = list()
    out_distributed_syms: list[list[Union[None, SymIndex]]] = list()
    for v in variables:
        distributed_syms: list[Union[None, SymIndex]] = list()
        for d, size in enumerate(shapes[v]):
            if d not in indeps[v]:
                sym: SymIndex = SymIndex()
                sym.assert_size(size=size)
                distributed_syms.append(sym)
            else:
                distributed_syms.append(None)
        inp_distributed_syms.append([*distributed_syms])
        out_distributed_syms.append([*distributed_syms])

    ### Initialize distribution sym indices array
    out_distribution_syms: list[list[Union[None, SymIndex]]] = list()
    for sym in inp_independent_syms:
        distribution_syms: list[Union[None, SymIndex]] = [sym]
        distribution_syms.extend([None for _ in variables])
        out_distribution_syms.append(distribution_syms)

    ### Replace Nones by sym_link indices (in out_distributed & distribution arrays)
    v_indeps: Tuple[Indep, ...] = tuple(indeps[v] for v in variables)
    xv_indeps: Tuple[Indep, ...] = tuple(expected_indeps[v] for v in variables)
    for i, (v_indepT, xv_indepT) in enumerate(zip(zip(*(v_indeps)), zip(*(xv_indeps)))):
        sym: SymIndex = inp_independent_syms[i]
        for j, (dT, xdT) in enumerate(zip(v_indepT, xv_indepT)):
            assert dT == xdT or xdT is None, (dT, xdT)
            ij_distributed: bool = dT != xdT
            if ij_distributed:
                sym_link: SymIndex = LinkedSymIndex(sym_index=sym)
                out_distributed_syms[j][dT] = sym_link
                out_distribution_syms[i][j + 1] = sym_link

    ### Use variable_perm to determine the OUTPUT order of partialities
    nvars: int = len(variables)
    assert len(variable_perm) == nvars
    assert all(isinstance(x, int) for x in variable_perm)
    is_true_perm: bool = sorted(variable_perm) == list(range(nvars))
    if is_true_perm:
        order_idx: list[int] = list(variable_perm)
    else:
        # stable argsort by priority (descending)
        order_idx = sorted(range(nvars), key=lambda j: variable_perm[j], reverse=True)
    # reorder output distributed blocks by chosen order
    out_distributed_syms = [out_distributed_syms[j] for j in order_idx]
    # reorder variable columns (skip leading indep col at idx 0)
    for row in range(len(out_distribution_syms)):
        head = out_distribution_syms[row][0:1]
        tail = out_distribution_syms[row][1:]
        out_distribution_syms[row] = head + [tail[j] for j in order_idx]

    ### Clean arrays of independent symbolic indices
    # clean independent symbolic indices
    clean_inp_independent_syms: list[SymIndex] = inp_independent_syms
    clean_out_independent_syms: list[SymIndex]
    clean_out_independent_syms = [
        sym for sym in out_independent_syms if sym is not None
    ]
    # clean distributed symbolic indices
    clean_inp_distributed_syms: list[list[SymIndex]] = [
        [sym for sym in syms if sym is not None] for syms in inp_distributed_syms
    ]
    clean_out_distributed_syms: list[list[SymIndex]] = [
        [sym for sym in syms if sym is not None] for syms in out_distributed_syms
    ]
    # clean distribution symbolic indices
    clean_out_distribution_syms: list[list[SymIndex]] = [
        [sym for sym in syms if sym is not None] for syms in out_distribution_syms
    ]

    # insert GO numel sym index
    sym: SymIndex = SymIndex()
    sym.assert_size(size=derivative.shape[0])
    clean_inp_independent_syms.insert(0, sym)
    clean_out_independent_syms.insert(0, sym)

    # numerize indices
    nested_indices: list[list[list[SymIndex]]] = [
        [clean_inp_independent_syms],
        [*clean_inp_distributed_syms],
        [*clean_out_distribution_syms],
        [clean_out_independent_syms],
        [*clean_out_distributed_syms],
    ]
    _numerize_indices(nested_indices=nested_indices)
    # permute and distribute dimensions
    distributed_derivative: Tensor = _einsum(
        in_tensor=derivative,
        nested_indices=nested_indices,
    )
    # restore fully distributed independent dimentions if required
    if keepdim:
        full_count: int = len(out_independent_syms)
        null_count: int = out_independent_syms.count(None)
        diff_count: int = full_count - null_count

        view: Tuple[int, ...] = tuple(distributed_derivative.shape)
        head_new_view: list[int] = [
            derivative.shape[0],
            *[1 if sym is None else sym.size for sym in out_independent_syms],
        ]
        new_view: Tuple[int, ...] = (*head_new_view, *view[(1 + diff_count) :])
        distributed_derivative = distributed_derivative.reshape(shape=new_view)

    return distributed_derivative


def _check_align_arguments(
    derivative: Tensor,
    variables: Sequence[int],
    variable_perm: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_shapes: Sequence[Shape],
    expected_indeps: Sequence[Indep],
) -> None:

    ### Typings & constants
    INDEP_SIZE: int = len(indeps[0])

    ### Inital checks
    assert len(shapes) == len(expected_shapes)
    assert len(indeps) == len(expected_indeps)
    # check that var values are within expected range
    assert all(var in range(len(shapes)) for var in variables)
    # check that var permutation values are within expected range
    assert set(variable_perm) == set(range(len(variables)))
    # check that every variable shares the same independent dimensions
    assert len(set(len(indep) for indep in (*indeps, *expected_indeps))) == 1
    # check that there is no reintegration of dimensions into indeps
    assert all(
        xd is None or d is not None
        for I, XI in zip(indeps, expected_indeps)
        for d, xd in zip(I, XI)
    )
    # check no indep dim is repeated twice
    assert (indep.count(d) == 1 for indep in indeps for d in indep if d is not None)
    # other internal coherence checks on indeps
    for j, step in enumerate(zip(*indeps)):
        assert all(isinstance(i, (int, type(None))) for i in step)
        size: set[int] = {shapes[i][ii] for i, ii in enumerate(step) if ii is not None}
        assert len(size) <= 1
        if len(size) == 1:
            sz: int = size.pop()
            assert sz == derivative.shape[1 + j]
    # check coherence between derivative shape and other arguments
    distributed_ndim: int = 0
    for v in variables:
        distributed_ndim += len(shapes[v]) - INDEP_SIZE + indeps[v].count(None)
    assert derivative.ndim == (1 + INDEP_SIZE + distributed_ndim)

    return None


def align_derivative(
    derivative: Tensor,
    variables: Sequence[int],
    variable_perm: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_shapes: Sequence[Shape],
    expected_indeps: Sequence[Indep],
    keepdim: bool,
) -> Tensor:
    """
    Align and reduce a derivative tensor to match expected shapes.

    This function permutes, collapses, and distributes dimensions of `derivative`
    according to `variables`, `shapes`, and independence masks, producing a tensor
    compatible with `expected_shapes` and `expected_indeps` for further processing.

    Args:
        derivative (Tensor): Input derivative tensor of shape (batch, *shapes...).
        variables (Sequence[int]): Indices mapping distributed dims to variables.
        shapes (Sequence[tuple[int, ...]]): Original shapes per term.
        indeps (Sequence[tuple[bool, ...]]): Boolean masks for
            independent dims per shape.
        expected_shapes (Sequence[tuple[int, ...]]): Desired shapes per term.
        expected_indeps (Sequence[tuple[Union[None, int], ...]]): Expected independent
            dims per term.
        keepdim (bool): Expects (and outputs) size 1 dims in empty independent dims

    Returns:
        Tensor: The aligned derivative tensor.
    """
    if bool(getattr(config, "DEBUG", False)):
        _check_align_arguments(
            derivative=derivative,
            variables=variables,
            variable_perm=variable_perm,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
        )

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]
    if shapes == expected_shapes:
        unbroadcasted_derivative = derivative
        unbroadcasted_shapes = tuple(shapes)
        unbroadcasted_indeps = tuple(indeps)
    else:
        (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
            unbroadcast_derivative(
                derivative=derivative,
                variables=variables,
                shapes=shapes,
                indeps=indeps,
                expected_shapes=expected_shapes,
            )
        )

    assert all(  # check that xd is not None implies ud is not None
        ud is not None or xd is None
        for i, (UI, XI) in enumerate(zip(unbroadcasted_indeps, expected_indeps))
        for ud, xd in zip(UI, XI)
    )  # FIX: summing dimensions to size 1 requires indep distribution

    assert all(  # check that summing dimensions to size 1 requires indep distribution
        xd is None
        or unbroadcasted_shapes[i][_num(ud)] == 1
        or expected_shapes[i][xd] != 1
        for i, (UI, XI) in enumerate(zip(unbroadcasted_indeps, expected_indeps))
        for ud, xd in zip(UI, XI)
    )  # FIX: summing dimensions to size 1 requires indep distribution

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]
    if unbroadcasted_shapes == expected_shapes:
        permuted_derivative = unbroadcasted_derivative
        permuted_shapes = unbroadcasted_shapes
        permuted_indeps = unbroadcasted_indeps
    else:
        (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
            derivative=unbroadcasted_derivative,
            variables=variables,
            shapes=unbroadcasted_shapes,
            indeps=unbroadcasted_indeps,
            expected_shapes=expected_shapes,
        )
        assert permuted_shapes == expected_shapes

    distributed_derivative: Tensor = _distribute_derivative(
        derivative=permuted_derivative,
        variables=variables,
        variable_perm=variable_perm,
        shapes=permuted_shapes,
        indeps=permuted_indeps,
        expected_indeps=expected_indeps,
        keepdim=keepdim,
    )

    distributed_derivative = distributed_derivative.contiguous()

    return distributed_derivative


def semialign_derivative(
    derivative: Tensor,
    variables: Sequence[int],
    shapes: Sequence[Shape],
    indeps: Sequence[Indep],
    expected_shapes: Sequence[Shape],
    keepdim: bool,
) -> StaticEDData:
    """
    Align and reduce a derivative tensor to match expected shapes.

    This function permutes, collapses, and distributes dimensions of `derivative`
    according to `variables`, `shapes`, and independence masks, producing a tensor
    compatible with `expected_shapes` and `expected_indeps` for further processing.

    Args:
        derivative (Tensor): Input derivative tensor of shape (batch, *shapes...).
        variables (Sequence[int]): Indices mapping distributed dims to variables.
        shapes (Sequence[tuple[int, ...]]): Original shapes per term.
        indeps (Sequence[tuple[bool, ...]]): Boolean masks for
            independent dims per shape.
        expected_shapes (Sequence[tuple[int, ...]]): Desired shapes per term.
        expected_indeps (Sequence[tuple[Union[None, int], ...]]): Expected independent
            dims per term.
        keepdim (bool): Expects (and outputs) size 1 dims in empty independent dims

    Returns:
        Tensor: The aligned derivative tensor.
    """
    if bool(getattr(config, "DEBUG", False)):
        _check_align_arguments(
            derivative=derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=indeps,
        )

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]
    if shapes == expected_shapes:
        unbroadcasted_derivative = derivative
        unbroadcasted_shapes = tuple(shapes)
        unbroadcasted_indeps = tuple(indeps)
    else:
        (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
            unbroadcast_derivative(
                derivative=derivative,
                variables=variables,
                shapes=shapes,
                indeps=indeps,
                expected_shapes=expected_shapes,
            )
        )

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]
    if unbroadcasted_shapes == expected_shapes:
        permuted_derivative = unbroadcasted_derivative
        permuted_shapes = unbroadcasted_shapes
        permuted_indeps = unbroadcasted_indeps
    else:
        (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
            derivative=unbroadcasted_derivative,
            variables=variables,
            shapes=unbroadcasted_shapes,
            indeps=unbroadcasted_indeps,
            expected_shapes=expected_shapes,
        )
        assert permuted_shapes == expected_shapes

    view: Tuple[int, ...] = calculate_shapes(
        first_size=derivative.shape[0],
        variables=tuple(variables),
        shapes=tuple(permuted_shapes),
        indeps=tuple(permuted_indeps),
        indeps_squeezed=(not keepdim),
    )[0]

    if permuted_derivative.shape != view:
        permuted_derivative = permuted_derivative.reshape(shape=view)
    else:
        permuted_derivative = permuted_derivative.contiguous()

    return (permuted_derivative, permuted_shapes, permuted_indeps)
