# Standard Library dependencies
import itertools
import math
import warnings
from typing import Any, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing.data import Indep


def construct_nd_identity(
    n: int,
    ndim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
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


def adjust_indep(
    shape: Tuple[int, ...],
    indep: Tuple[Union[None, int], ...],
    expected_shape: Tuple[int, ...],
) -> Tuple[Union[None, int], ...]:

    projected_indep: Indep = indep

    if shape != expected_shape:
        if len(shape) == len(expected_shape):
            if math.prod(shape) == math.prod(expected_shape):
                # same numel -> permute
                permutation: Tuple[int, ...] = _find_best_permutation(
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
            trucated_indep: list[Union[None, int]]
            trucated_indep = [None for _ in range(len(shape) - drop)]
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
    permuted_shape: list[Union[None, int]] = [None for _ in permutation]
    adjusted_indep: list[Union[None, int]] = [None for _ in indep]
    for i, p in enumerate(permutation):
        permuted_shape[i] = shape[p]
        if p in indep:
            adjusted_indep[indep.index(p)] = i
    return (tuple(permuted_shape), tuple(adjusted_indep))


def _shape_broadcastable(shape: Tuple[int, ...], target: Tuple[int, ...]) -> bool:
    """
    Check if two shapes are broadcastable via gcd rule.

    Args:
        shape (tuple[int, ...]): The original shape.
        target (tuple[int, ...]): The target shape to broadcast to.

    Returns:
        bool: True if for every dimension i,
            gcd(shape[i], target[i]) == target[i], else False.
    """
    return all(math.gcd(s, t) == t for s, t in zip(shape, target))


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
    movement = sum(1 for s, a in zip(shape, target) if s != a)
    positions = [i for i, (s, a) in enumerate(zip(shape, target)) if s != a]
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


def _find_best_permutation(
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


def _unbroadcast(
    tensor: Tensor, target_shape: Tuple[int, ...], keepdim: bool
) -> Tensor:
    """
    Reduce a tensor to a target shape by summing out broadcasted dimensions.

    This function reshapes `tensor` and performs a single call to torch.sum
    to eliminate any dimensions that were introduced by broadcasting.

    Args:
        tensor (Tensor): The input tensor.
        target_shape (tuple[int, ...]):
            The desired output shape. Must satisfy `gcd(tensor.shape[i],
            target_shape[i]) == target_shape[i]` for each dimension i.
        keepdim (bool):
            If True, retains singleton dimensions in the result after summation;
            if False, those dimensions are squeezed out (just like torch.sum).
    Returns:
        Tensor: A tensor of shape `target_shape` (if keepdim=True), or
        squeezed to remove size-1 dims (if keepdim=False).
    """
    assert tensor.ndim == len(target_shape), (tensor.ndim, len(target_shape))
    assert all(math.gcd(s, t) == t for s, t in zip(tensor.shape, target_shape))
    # Build a new shape that “splits” any axis where s != t into (factor, t),
    # then collect all the indices of those “factor” axes so we can sum over them.
    new_shape: list[int] = []
    dims_to_sum: list[int] = []
    running_axis = 0

    for s, t in zip(tensor.shape, target_shape):
        if s == t:
            # no broadcasting along this axis: keep it as-is
            new_shape.append(t)
            running_axis += 1
        else:
            factor = s // t
            # split the original size s into (factor, t)
            new_shape.extend([factor, t])
            dims_to_sum.append(running_axis)  # sum out the “factor” axis
            running_axis += 2

    # Now reshape and do one big sum over all factor-axes
    result = tensor.reshape(new_shape)
    if len(dims_to_sum) > 0:
        # always use keepdim=False here, so the “factor” axes are removed entirely
        result = result.sum(dim=tuple(dims_to_sum), keepdim=False)

    # At this point, `result.shape` should be exactly `target_shape`.
    # If keepdim=False, we additionally want to squeeze away any size-1 dims.
    if not keepdim:
        result = result.squeeze()

    return result


def _divide_permutations(
    permutations: Sequence[Tuple[int, ...]],
    divisions: Sequence[Tuple[bool, ...]],
) -> list[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Split each permutation into independent and distributed parts.

    Args:
        permutations (Sequence[tuple[int, ...]]): List of permutations.
        divisions (Sequence[tuple[bool, ...]]): Boolean masks indicating
            independent dimensions for each permutation.

    Returns:
        list[tuple[tuple[int, ...], tuple[int, ...]]]: A list of pairs
            (independent_perm, distributed_perm) per original permutation.
    """
    # checks
    assert len(permutations) == len(divisions)
    assert all([isinstance(i, int) for p in permutations for i in p])
    assert all([isinstance(b, bool) for p in divisions for b in p])
    # divide permuations
    divided_permutations: list[Tuple[Tuple[int, ...], Tuple[int, ...]]] = list()
    for perm, div in zip(permutations, divisions):
        a_permutation: list[int] = list()
        b_permutation: list[int] = list()
        for i, p in enumerate(perm):
            if div[i]:
                a_permutation.append(p)
            else:
                b_permutation.append(p)
        divided_permutations.append((tuple(a_permutation), tuple(b_permutation)))
    return divided_permutations


def _match_sequences(
    sequences: Sequence[Sequence[Any]], default: Sequence[Any]
) -> list[Any]:
    """
    Merge N equal-length sequences and enforce given default order.

    Asserts all sequences same length and consistent partial orders, then
    returns list of default objects seen in sequence constraints.

    Args:
        sequences: sequences of objects or None, all same length.
        default: sequence of all objects in desired fallback order.
    """
    # determine which default objects actually appear
    filtered = [o for o in default if any(o in seq for seq in sequences)]
    if not filtered:
        return []
    # build precedence graph among filtered objects
    graph: dict[Any, set[Any]] = {o: set() for o in filtered}
    indegree: dict[Any, int] = {o: 0 for o in filtered}
    for seq in sequences:
        prev: Any = None
        for obj in seq:
            if obj is None or obj not in graph:
                continue
            if prev is not None and obj not in graph[prev]:
                graph[prev].add(obj)
                indegree[obj] += 1
            prev = obj
    # topological sort, using default order as tie-breaker
    order_idx = {o: i for i, o in enumerate(filtered)}
    zeros: list[Any] = sorted(
        (o for o, d in indegree.items() if d == 0), key=lambda o: order_idx[o]
    )
    result: list[Any] = []
    while zeros:
        u = zeros.pop(0)
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                i = 0
                while i < len(zeros) and order_idx[zeros[i]] <= order_idx[v]:
                    i += 1
                zeros.insert(i, v)
    assert len(result) == len(filtered)

    return result


class SymIndex:
    """
    Symbolic index tracker that enforces consistent size and assigns IDs.
    """

    def __init__(self) -> None:
        self._id: Union[int, None] = None
        self._size: Union[int, None] = None

    def assert_size(self, size: int) -> None:
        """
        Enforce or set the size of this symbolic index.

        Args:
            size (int): The dimension size to assert.
        """
        if self._size is None:
            self._size = size
        else:
            assert self._size == size
        return None

    @property
    def id(self) -> int:
        """
        Get the assigned integer ID of this SymIndex.

        Returns:
            int: Assigned ID.
        """
        assert self._id is not None
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """
        Assign an integer ID to this SymIndex.

        Args:
            value (int): The ID to assign.
        """
        self._id = value

    @property
    def size(self) -> int:
        """
        Get the enforced size of this SymIndex.

        Returns:
            int: The dimension size.
        """
        assert self._size is not None
        return self._size


class LinkedSymIndex(SymIndex):
    """
    A SymIndex linked to another, sharing its size.
    """

    def __init__(self, sym_index: SymIndex) -> None:
        super().__init__()
        self._link: SymIndex = sym_index

    def assert_size(self, size: int) -> None:
        """
        Delegate size assertion to linked SymIndex.

        Args:
            size (int): The dimension size to assert.
        """
        self._link.assert_size(size)

    @property
    def size(self) -> int:
        """
        Get size from linked SymIndex.

        Returns:
            int: The dimension size.
        """
        return self._link.size


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

    def _flatten(seq: Sequence[Sequence[int]]) -> Tuple[int]:
        return tuple(x for sub in seq for x in sub)

    ### Sort einsum args
    einsum_args: list[Union[Tensor, Tuple[int, ...]]] = [in_tensor]
    external_syms: list[list[SymIndex]]
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
    output_syms: list[list[SymIndex]]
    output_syms = [*_flatten(nested_indices[3]), *_flatten(nested_indices[4])]
    einsum_args.append(_insert_numerized(output_syms))

    ### Compute composed differential
    result: Tensor = torch.einsum(*einsum_args)

    return result


def align_differential(
    differential: Tensor,
    variables: Sequence[int],
    shapes: Sequence[Tuple[int, ...]],
    indeps: Sequence[Tuple[Union[None, int], ...]],
    expected_shapes: Sequence[Tuple[int, ...]],
    expected_indeps: Sequence[Tuple[Union[None, int], ...]],
    keepdim: bool,
) -> Tensor:
    """
    Align and reduce a differential tensor to match expected shapes.

    This function permutes, collapses, and distributes dimensions of `differential`
    according to `variables`, `shapes`, and independence masks, producing a tensor
    compatible with `expected_shapes` and `expected_indeps` for further processing.

    Args:
        differential (Tensor): Input differential tensor of shape (batch, *shapes...).
        variables (Sequence[int]): Indices mapping distributed dims to variables.
        shapes (Sequence[tuple[int, ...]]): Original shapes per term.
        indeps (Sequence[tuple[bool, ...]]): Boolean masks for
            independent dims per shape.
        expected_shapes (Sequence[tuple[int, ...]]): Desired shapes per term.
        expected_indeps (Sequence[tuple[Union[None, int], ...]]): Expected independent
            dims per term.
        keepdim (bool): Expects (and outputs) size 1 dims in empty independent dims

    Returns:
        Tensor: The aligned differential tensor.
    """

    ### Typings & constants
    zip3: zip
    counter: int
    variable_repetitions: int
    XX: int = differential.shape[0]
    INDEPS: int = len(indeps[0])  # asdf

    ### Obtain descriptive info about independent dimensions
    NULL_INDEPS: list[bool] = [True for _ in range(INDEPS)]
    XNULL_INDEPS: list[bool] = [True for _ in range(INDEPS)]
    INDEP_MAX_SHAPE: list[int] = [1 for _ in range(INDEPS)]
    XINDEP_MAX_SHAPE: list[int] = [1 for _ in range(INDEPS)]
    for i, (indep, xindep) in enumerate(zip(indeps, expected_indeps)):
        for j, (dim, xdim) in enumerate(zip(indep, xindep)):
            if dim is not None:
                NULL_INDEPS[j] = False
                INDEP_MAX_SHAPE[j] = max(INDEP_MAX_SHAPE[j], shapes[i][dim])
            if xdim is not None:
                XNULL_INDEPS[j] = False
                XINDEP_MAX_SHAPE[j] = max(XINDEP_MAX_SHAPE[j], expected_shapes[i][dim])

    ### Inital checks
    assert len(shapes) == len(expected_shapes)
    assert len(indeps) == len(expected_indeps)
    # assert
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
    # check coherence between differential shape and other arguments
    distributed_ndim: int = 0
    for v in variables:
        distributed_ndim += len(shapes[v]) - INDEPS + indeps[v].count(None)
    # obtain indep max
    if keepdim:
        assert differential.ndim == (1 + INDEPS + distributed_ndim)
    else:
        ndim: int = 1 + INDEPS + distributed_ndim
        assert differential.ndim == ndim
        expected_shape: list[int] = [differential.shape[0]]
        expected_shape.extend(INDEP_MAX_SHAPE)
        expected_shape.extend(differential.shape[(1 + INDEPS) :])
        differential = differential.reshape(shape=tuple(expected_shape))

    ### Calculate partial target shapes & partial target indeps (for reduction)
    target_shapes: list[Tuple[int, ...]] = list()
    permutations: list[Union[None, Tuple[int, ...]]] = list()
    for shape, expected_shape in zip(shapes, expected_shapes):
        padding: int = max((len(shape) - len(expected_shape)), 0)
        target_shape: Tuple[int, ...] = (1,) * padding + expected_shape
        # calculate possible permutations if no broadcasting is posible
        #   (note. "broadcastable" and "require permutation" are mutually exclusive)
        broadcastable: bool = _shape_broadcastable(shape=shape, target=target_shape)
        if shape == target_shape or broadcastable:
            permutations.append(tuple(range(len(expected_shape))))
        else:
            print("permute")
            target_shape = shape
            permutations.append(
                _find_best_permutation(
                    shape=shape,
                    target=expected_shape,
                )
            )
        # save target shape
        target_shapes.append(target_shape)

    ### Reduction
    shape: Tuple[int, ...]
    indep: Tuple[Union[None, int], ...]
    reduced_complete_shape: list[int] = [XX, *[1 for _ in range(INDEPS)]]  # ???
    reduced_shapes: list[list[int]] = [list() for _ in shapes]
    reduced_indeps: list[list[Union[None, int]]] = [list() for _ in shapes]
    eliminated_indices: list[list[int]] = [list() for _ in shapes]
    # add independent dimensions
    for i, (shape, indep) in enumerate(zip(target_shapes, indeps)):
        reduced_indeps[i] = [None for _ in indep]
        for j, indep_idx in enumerate(indep):
            if indep_idx is not None:
                current: int = reduced_complete_shape[j + 1]
                reduced_complete_shape[j + 1] = max(current, shape[indep_idx])
        counter = 0
        for j, dim_size in enumerate(shape):
            j_not_in_padding: bool = j >= (len(shape) - len(expected_shapes[i]))
            if dim_size > 1 or shapes[i][j] == 1 or j_not_in_padding:
                reduced_shapes[i].append(dim_size)
                if j in indep:
                    reduced_indeps[i][indep.index(j)] = counter
                counter += 1
            elif j in indep:
                eliminated_indices[i].append(indep.index(j))
    # add non-independent dimensions
    for var in variables:
        shape = target_shapes[var]
        indep = indeps[var]
        for j, dim_size in enumerate(shape):
            if j not in indep:
                reduced_complete_shape.append(dim_size)
    # unbroadcast
    reduced_differential: Tensor = _unbroadcast(
        tensor=differential,
        target_shape=reduced_complete_shape,
        keepdim=True,
    )
    INDEPS = max(len(i) for i in reduced_indeps)
    squeezed_shape: Tuple[int, ...] = reduced_differential.shape
    assert squeezed_shape == tuple(reduced_complete_shape)

    ### Calculate indices for Distribution & Permutation
    # instantiate einsum sym-index groups
    in_independent_sym_indices: list[SymIndex] = [SymIndex() for _ in range(INDEPS)]
    in_distributed_sym_indices: list[list[SymIndex]] = [list() for _ in variables]
    identities_sym_indices: list[list[SymIndex]] = list()
    out_independent_sym_indices: list[SymIndex]
    out_distributed_sym_indices: list[SymIndex] = [list() for _ in variables]
    # create dimension sym indices
    partial_sym_indices: list[list[list[SymIndex]]] = list()
    for i, (shape, indep) in enumerate(zip(reduced_shapes, reduced_indeps)):
        variable_repetitions: int = variables.count(i)
        sub_list: list[list[SymIndex]] = list()
        counter = 0
        for j, size in enumerate(shape):
            dim_independent: bool = j in indep
            n: int = 1 if dim_independent else variable_repetitions
            subsub_list: list[SymIndex] = list()
            for _ in range(n):
                sym_index: SymIndex = SymIndex()
                if dim_independent:
                    idx: int = indep.index(j)
                    sym_index = in_independent_sym_indices[idx]
                    counter += 1
                sym_index.assert_size(size=size)
                subsub_list.append(sym_index)
            sub_list.append(subsub_list)
        partial_sym_indices.append(sub_list)
    # divide permutations into independent and regular
    divisions: list[list[bool]] = list()
    for i, shape in enumerate(expected_shapes):
        divisions.append([j in expected_indeps[i] for j, _ in enumerate(shape)])
    divided_permutations: list[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    divided_permutations = _divide_permutations(
        permutations=permutations,
        divisions=divisions,
    )
    # add indentity sym indices
    aux_out: list[list[SymIndex]] = [[None for _ in s] for s in reduced_shapes]
    require_distribution: list[list[bool]]
    require_distribution = [[False for _ in shapes] for _ in range(INDEPS)]
    for i, sym_idx in enumerate(in_independent_sym_indices):
        for j, _ in enumerate(shapes):
            in_independent: bool = reduced_indeps[j][i] is not None
            out_independent: bool = expected_indeps[j][i] is not None
            require_distribution[i][j] |= in_independent ^ out_independent
        identity_sym_indices: list[SymIndex]
        identity_sym_indices = [sym_idx] if any(require_distribution[i]) else list()
        if any(require_distribution[i]):
            n: int = sum([require_distribution[i][v] for v in variables]) - 1
            n += int(not all([require_distribution[i][v] for v in variables]))
            for _ in range(n):
                identity_sym_indices.append(LinkedSymIndex(sym_index=sym_idx))
        identities_sym_indices.append(identity_sym_indices)
    # add out independent sym indices
    for i, indep in enumerate(reduced_indeps):
        counter = 0
        for dim in indep:
            if dim is not None:
                if not require_distribution[counter][i]:
                    t: int = permutations[i].index(dim)
                    aux_out[i][t] = in_independent_sym_indices[dim]
            counter += 1
    out_independent_sym_indices = _match_sequences(
        sequences=aux_out,
        default=in_independent_sym_indices,
    )
    # add distributed dimensions
    counters: list[int] = [0 for _ in shapes]
    zip3: zip = zip(partial_sym_indices, reduced_indeps, divided_permutations)
    for i, (subindices, indep, perms) in enumerate(zip3):
        variable_repetitions: int = variables.count(i)
        # first nested loop structure to place input indices
        for j, subsubindices in enumerate(subindices):
            dim_independent: bool = j in indep
            assert isinstance(subsubindices, list)
            assert all([isinstance(SI, SymIndex) for SI in subsubindices])
            if not dim_independent:
                assert len(subsubindices) == variable_repetitions
                counter = 0
                for k, var in enumerate(variables):
                    if i == var:
                        sym_idx: SymIndex = subsubindices[counter]
                        in_distributed_sym_indices[k].append(sym_idx)
                        counter += 1
        # second nested loop structure to place output indices
        counter = 0
        for k, var in enumerate(variables):
            if i == var:
                for p in perms[1]:
                    sym_idx: SymIndex
                    # get identity linked sym-index (distribution)
                    if p in indep:
                        j: int = indep.index(p)
                        t: int = 0
                        all_distributed: bool = True
                        for v, _ in enumerate(shapes):
                            t += counters[v] if require_distribution[j][v] else 0
                            all_distributed *= require_distribution[j][v]
                        t += int(not all_distributed)
                        sym_idx = identities_sym_indices[j][t]
                    # get pre-distributed sym-index (no distribution)
                    elif p is not None:
                        sym_idx = subindices[p][counter]
                    out_distributed_sym_indices[k].append(sym_idx)
                counter += 1
                counters[var] += 1

    # Distribute and permute
    # insert GO numel sym index
    sym_index: SymIndex = SymIndex()
    sym_index.assert_size(size=XX)
    in_independent_sym_indices.insert(0, sym_index)
    out_independent_sym_indices.insert(0, sym_index)
    # numerize indices
    nested_indices: list[list[SymIndex]] = [
        [in_independent_sym_indices],
        in_distributed_sym_indices,
        identities_sym_indices,
        [out_independent_sym_indices],
        out_distributed_sym_indices,
    ]
    _numerize_indices(nested_indices=nested_indices)
    # permute and ditribute dimensions
    aligned_differential: Tensor = _einsum(
        in_tensor=reduced_differential,
        nested_indices=nested_indices,
    )
    # include size 1 aligned differential independent dims
    out_distributed_ndim: int = 0
    for v in variables:
        out_distributed_ndim += len(expected_shapes[v]) - len(indeps[v])
        out_distributed_ndim += expected_indeps[v].count(None)
    expected_not_nulls: int = XNULL_INDEPS.count(False)
    assert aligned_differential.ndim == (1 + expected_not_nulls + out_distributed_ndim)
    if keepdim:
        new_shape: list[int] = [aligned_differential.shape[0]]
        new_shape.extend(XINDEP_MAX_SHAPE)
        new_shape.extend(aligned_differential.shape[(1 + expected_not_nulls) :])
        aligned_differential = aligned_differential.reshape(shape=new_shape)

    return aligned_differential
