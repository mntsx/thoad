# python 3.12

# Standard Library dependencies
from typing import Any, Optional, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing import Shape, Indep, Notation
from thoad.differentiation.engine.broadcasting.figuration import (
    calculate_shapes,
    construct_nd_identity,
)
import thoad.config as config


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


def _check_symbolizer_arguments(
    notation: Sequence[Sequence[Sequence[int]]],
    external_indep: Indep,
    external_indep_syms: Sequence[SymIndex],
    expected_indeps: Sequence[Indep],
    included_indep_inds: Sequence[int],
    included_indep_syms: Sequence[SymIndex],
) -> None:
    # check notation
    assert len(notation) == 3
    assert len(notation[0]) == 2
    assert len(notation[2]) == 2
    assert all(isinstance(iii, int) for i in notation for ii in i for iii in ii)
    # check external_indep
    assert all(isinstance(d, (type(None), int)) for d in external_indep)
    # check expected_indep
    assert len(expected_indeps) == len(notation[1])
    for xindep in expected_indeps:
        assert all(isinstance(d, (type(None), int)) for d in xindep)
        assert all(
            external_indep[i] is not None or xd is None for i, xd in enumerate(xindep)
        )
    # check external_indep_syms
    assert len(external_indep_syms) == len(external_indep)
    assert all(isinstance(sym, SymIndex) for sym in external_indep_syms)
    # check included_indep_inds
    assert all(i in notation[0][1] for i in included_indep_inds)
    assert all(i in sub for i in included_indep_inds for sub in notation[1])
    # check included_indep_syms
    assert len(included_indep_syms) == len(included_indep_inds)

    return None


def _assemble_indep_map(
    indep: Indep,
    indep_syms: Sequence[SymIndex],
    external_notation: Sequence[int],
) -> dict[int, SymIndex]:
    indep_map: dict[int, SymIndex] = dict()
    for j, dim in enumerate(indep):
        if dim is not None:
            indep_map[external_notation[dim]] = indep_syms[j]
    return indep_map


def _symbolize_array(
    array: Sequence[int],
    replace_map: Union[None, dict[int, SymIndex]] = None,
) -> Tuple[list[SymIndex], dict[int, SymIndex]]:

    ### checks
    assert isinstance(array, Sequence)
    assert all(isinstance(i, int) for i in array)
    assert isinstance(replace_map, (type(None), dict))
    if replace_map is not None:
        assert all(isinstance(i, int) for i in replace_map.keys())
        assert all(
            isinstance(sym, (type(None), SymIndex)) for sym in replace_map.values()
        )

    ### map ints to symbolic indices
    # create record to progressively store read indices
    record: dict[int, SymIndex] = dict()
    if replace_map is not None:
        for i, sym in replace_map.items():
            record[i] = sym
    # create new array mapping original integer indices to symbolic indices
    symbolic_array: list = list()
    for i in array:
        assert isinstance(i, int)
        sym: SymIndex
        if i in record:
            sym = record[i]
        else:
            sym = SymIndex()
            record[i] = sym
        symbolic_array.append(sym)

    return (symbolic_array, record)


def _symbolize_notation(
    notation: Sequence[Sequence[Sequence[int]]],
    indep_map: dict[int, SymIndex],
) -> Tuple[
    list[SymIndex],
    list[SymIndex],
    list[list[SymIndex]],
]:

    ### Symbolize notation
    record: dict[int, SymIndex] = {i: sym for i, sym in indep_map.items()}

    external_syms: list[SymIndex]
    external_syms, record = _symbolize_array(
        array=notation[0][0],
        replace_map=record,
    )

    internal_syms: list[SymIndex]
    internal_syms, record = _symbolize_array(
        array=notation[0][1],
        replace_map=record,
    )

    composed_syms: list[list[SymIndex]] = list()
    for subnotation in notation[1]:
        composed_sub_syms: list[SymIndex]
        composed_sub_syms, record = _symbolize_array(
            array=subnotation,
            replace_map=record,
        )
        composed_syms.append(composed_sub_syms)

    return (external_syms, internal_syms, composed_syms)


class SymInterface:

    def __init__(
        self,
        notation: Notation,
        external_indep: Indep,
        external_indep_syms: Sequence[SymIndex],
        expected_indeps: Sequence[Indep],
        included_indep_inds: Sequence[int],
        included_indep_syms: Sequence[SymIndex],
    ) -> None:

        ### Check and save passed arguments
        if bool(getattr(config, "DEBUG", False)):
            _check_symbolizer_arguments(
                notation=notation,
                external_indep=external_indep,
                external_indep_syms=external_indep_syms,
                expected_indeps=expected_indeps,
                included_indep_inds=included_indep_inds,
                included_indep_syms=included_indep_syms,
            )
        self._notation: Notation = notation
        self._external_indep: Indep = external_indep
        self._external_indep_syms: list[SymIndex] = list(external_indep_syms)

        ### Create symbolic arrays for index manipulation
        indep_map: dict[int, SymIndex] = _assemble_indep_map(
            indep=external_indep,
            indep_syms=external_indep_syms,
            external_notation=notation[0][0],
        )
        indep_map.update(
            {i: sym for i, sym in zip(included_indep_inds, included_indep_syms)}
        )
        raw_notation_syms: Tuple[list[SymIndex], list[SymIndex], list[list[SymIndex]]]
        raw_notation_syms = _symbolize_notation(
            notation=notation,
            indep_map=indep_map,
        )

        ### Initialize arrays of symbolic indices
        # initialize raw syms
        self._raw_external_syms: list[SymIndex] = raw_notation_syms[0]
        self._raw_internal_syms: list[SymIndex] = raw_notation_syms[1]
        self._raw_composed_syms: list[list[SymIndex]] = raw_notation_syms[2]
        # initialize mod syms with same values as raw
        self._mod_external_syms: list[SymIndex] = raw_notation_syms[0]
        self._mod_internal_syms: list[SymIndex] = raw_notation_syms[1]
        self._mod_composed_syms: list[list[SymIndex]] = raw_notation_syms[2]

        ### Gather symbolic indices to be included and excluded
        self._included_indep_syms: list[SymIndex] = list(included_indep_syms)
        self._excluded_indep_syms: list[list[SymIndex]] = [
            [
                sym
                for d, xd, sym in zip(external_indep, xindep, external_indep_syms)
                if d is not None and xd is None
            ]
            for xindep in expected_indeps
        ]
        self._inactive_indep_syms: list[SymIndex] = [
            sym for d, sym in zip(external_indep, external_indep_syms) if d is None
        ]
        self._unvaried_indep_syms: list[list[SymIndex]] = [
            [
                sym
                for d, xd, sym in zip(external_indep, xindep, external_indep_syms)
                if d is not None and xd is not None
            ]
            for xindep in expected_indeps
        ]
        self._composed_indep_syms: list[list[SymIndex]] = [
            [*syms, *self._included_indep_syms] for syms in self._unvaried_indep_syms
        ]

        return None

    @property
    def raw_external_syms(self) -> list[SymIndex]:
        return self._raw_external_syms

    @property
    def raw_internal_syms(self) -> list[SymIndex]:
        return self._raw_internal_syms

    @property
    def raw_composed_syms(self) -> list[list[SymIndex]]:
        return self._raw_composed_syms

    @property
    def mod_external_syms(self) -> list[SymIndex]:
        return self._mod_external_syms

    @property
    def mod_internal_syms(self) -> list[SymIndex]:
        return self._mod_internal_syms

    @property
    def mod_composed_syms(self) -> list[list[SymIndex]]:
        return self._mod_composed_syms

    @property
    def excluded_indep_syms(self) -> list[list[SymIndex]]:
        return self._excluded_indep_syms

    @property
    def composed_indep_syms(self) -> list[list[SymIndex]]:
        return self._composed_indep_syms

    def size_external_syms(self, external_shape: Shape) -> None:
        for size, sym in zip(external_shape, self._raw_external_syms):
            sym.assert_size(size=size, override=True)
        return None

    def size_internal_syms(self, internal_shape: Shape) -> None:
        for size, sym in zip(internal_shape, self._raw_internal_syms):
            sym.assert_size(size=size, override=True)
        return None

    def remove_external_indeps(self) -> None:
        self._mod_external_syms = [
            sym
            for sym in self._mod_external_syms
            if sym not in self._external_indep_syms
        ]
        return None

    def remove_composed_indeps(self) -> None:
        self._mod_composed_syms = [
            [
                sym
                for sym in composed_sub_syms
                if sym not in self._composed_indep_syms[i]
                and sym not in self._included_indep_syms
            ]
            for i, composed_sub_syms in enumerate(self._mod_composed_syms)
        ]
        return None

    def link_spreaded_syms(self) -> list[list[SymIndex]]:
        identity_syms: list[list[SymIndex]] = list()
        identity_tensors: list[Tensor] = list()
        for sym in set((*self._mod_external_syms, *self._mod_internal_syms)):
            require_distribution: bool = True
            require_distribution &= all(sym in sub for sub in self._mod_composed_syms)
            require_distribution &= sym not in self._external_indep_syms
            require_distribution &= sym not in self._included_indep_syms
            require_distribution &= len(self._mod_composed_syms) > 1
            if require_distribution:
                # save symbolic indices associated to the identity tensor
                identity_sub_syms: list[SymIndex] = [sym]
                for composed_sub_syms in self._mod_composed_syms:
                    new_sym: SymIndex = LinkedSymIndex(sym_index=sym)
                    composed_sub_syms[composed_sub_syms.index(sym)] = new_sym
                    identity_sub_syms.append(new_sym)
                identity_syms.append(identity_sub_syms)
        return identity_syms

    def extend_excluded_syms(
        self, identity_syms: list[list[SymIndex]]
    ) -> list[list[SymIndex]]:
        assert len(identity_syms) == len(self._external_indep_syms)
        for excluded_indep_sub_syms, composed_sub_syms in zip(
            self._excluded_indep_syms, self._mod_composed_syms
        ):
            # save symbolic indices associated to the identity tensor
            for i, sym in enumerate(self._external_indep_syms):
                if sym in excluded_indep_sub_syms and sym in composed_sub_syms:
                    new_sym: SymIndex = LinkedSymIndex(sym_index=sym)
                    composed_sub_syms[composed_sub_syms.index(sym)] = new_sym
                    identity_syms[i].append(new_sym)
        return identity_syms


def _populated(obj: Union[None, Any]) -> Any:
    assert obj is not None
    return obj


def calculate_composed_shapes(
    composed_permutation: Sequence[int],
    raw_composed_syms: Sequence[Sequence[SymIndex]],
) -> Tuple[Shape, ...]:
    # construct composed shapes and indeps
    composed_shapes: list[Union[None, Shape]] = [None for _ in composed_permutation]
    for i, idx in enumerate(composed_permutation):
        # add shape
        sub_composed_syms: list[SymIndex] = list(raw_composed_syms[idx])
        composed_shape: list[int] = list()
        for sym in sub_composed_syms:
            composed_shape.append(sym.size)
        composed_shapes[i] = tuple(composed_shape)
    return tuple(_populated(S) for S in composed_shapes)


def calculate_composed_indeps(
    composed_permutation: Sequence[int],
    external_indep_syms: Sequence[SymIndex],
    included_indep_syms: Sequence[SymIndex],
    composed_indep_syms: Sequence[Sequence[SymIndex]],
    raw_composed_syms: Sequence[Sequence[SymIndex]],
) -> Tuple[Indep, ...]:
    # construct composed shapes and indeps
    composed_indeps: list[Union[None, Indep]] = [None for _ in composed_permutation]
    for i, _ in enumerate(composed_permutation):
        composed_sub_indep: list[Union[None, int]] = list()
        for dim, sym in enumerate((*external_indep_syms, *included_indep_syms)):
            if sym in composed_indep_syms[i]:
                composed_sub_indep.append(raw_composed_syms[i].index(sym))
            else:
                composed_sub_indep.append(None)
        composed_indeps[composed_permutation.index(i)] = tuple(composed_sub_indep)
    return tuple(_populated(I) for I in composed_indeps)


def _numerize_indices(nested_indices: list[list[list[SymIndex]]]) -> None:
    flat_indices: set[SymIndex] = set()
    for sub in nested_indices:
        for subsub in sub:
            for sym_idx in subsub:
                flat_indices.add(sym_idx)
    for i, sym_idx in enumerate(flat_indices):
        sym_idx.id = i
    return None


def _contract(
    nested_indices: list[list[list[SymIndex]]],
    external_tensor: Tensor,
    internal_tensors: list[Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:

    def _insert_numerized(syms: list[SymIndex]) -> Tuple[int, ...]:
        return tuple(sym.id for sym in syms)

    def _flatten(seq: Sequence[Sequence[SymIndex]]) -> Tuple[SymIndex, ...]:
        return tuple(x for sub in seq for x in sub)

    ### Create identity tensors
    identity_indices: list[list[SymIndex]] = list()
    identity_tensors: list[Tensor] = list()
    for syms in nested_indices[3]:
        assert len(set((sym.size for sym in syms))) == 1
        match len(syms):
            case 1:
                pass
            case 2:
                for mutating_syms in [
                    *nested_indices[0],
                    *nested_indices[2],
                    *nested_indices[4],
                ]:
                    for i, sym in enumerate(mutating_syms):
                        if sym == syms[0]:
                            mutating_syms[i] = syms[1]
            case _:
                identity: Tensor = construct_nd_identity(
                    n=syms[0].size, ndim=len(syms), dtype=dtype, device=device
                )
                identity_indices.append(syms)
                identity_tensors.append(identity)

    ### Assemble einsum arguments
    # insert external tensor and external indices
    einsum_args: list[Union[Tensor, Tuple[int, ...]]] = [external_tensor]
    external_syms: list[SymIndex]
    external_syms = [*_flatten(nested_indices[0]), *_flatten(nested_indices[1])]
    einsum_args.append(_insert_numerized(external_syms))
    # insert tensors and indices of both internals and identities
    loop_tensors: list[Tensor] = [*internal_tensors, *identity_tensors]
    loop_syms: list[list[SymIndex]] = [*nested_indices[2], *identity_indices]
    for tensor, syms in zip(loop_tensors, loop_syms):
        if tensor.ndim > 0:
            einsum_args.append(tensor)
            einsum_args.append(_insert_numerized(syms))
    # insert composed indices
    composed_syms: list[SymIndex]
    composed_syms = [*_flatten(nested_indices[4]), *_flatten(nested_indices[5])]
    einsum_args.append(_insert_numerized(composed_syms))

    ### Compute composed derivative
    # determine if the einsum operator is necesary
    clone_tensor: bool = False
    assert len(einsum_args) >= 3
    if len(einsum_args) == 3:
        assert isinstance(einsum_args[0], Tensor)
        assert isinstance(einsum_args[1], Tuple)
        assert isinstance(einsum_args[2], Tuple)
        clone_tensor = einsum_args[1] == einsum_args[2]
    # execute the einsum operator
    result: Tensor
    if clone_tensor:
        assert isinstance(einsum_args[0], Tensor)
        result = einsum_args[0].clone()
    else:
        result = torch.einsum(*einsum_args)

    return result


def _bool(obj: Union[bool, Any]) -> bool:
    assert isinstance(obj, bool)
    return obj


def contract_derivatives(
    composed_permutation: Tuple[int, ...],
    external_derivative: Tensor,
    external_shapes: Tuple[Shape, ...],
    external_indeps: Tuple[Indep, ...],
    expected_indeps: Tuple[Tuple[Indep, ...], ...],
    internal_derivatives: Tuple[Tensor, ...],
    einstein_notations: Sequence[Notation],
    effective_order: int,
    flexible_indeps: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]:

    ### Constants
    primal_size: int = external_derivative.shape[0]

    ### Checks
    # check notation coherence with shapes and indeps
    distributed_shapes: list[list[int]] = [list() for _ in external_shapes]
    for i, (shape, indep, notation) in enumerate(
        zip(external_shapes, external_indeps, einstein_notations)
    ):
        distributed_shapes[i] = [sz for j, sz in enumerate(shape) if j not in indep]
    # check coherence between indeps
    indep_sizes: list[int] = list()
    for i, row in enumerate(zip(*external_indeps)):
        row_sizes: list[int]
        row_sizes = [external_shapes[j][d] for j, d in enumerate(row) if d is not None]
        indep_sizes.append(max([1, *row_sizes]))
    assert external_derivative.shape[1 : (len(indep_sizes) + 1)] == tuple(indep_sizes)
    # check coherence between notation and shapes
    for i, (shape, notation) in enumerate(zip(distributed_shapes, einstein_notations)):
        assert len(external_shapes[i]) == len(notation[0][0])
        assert len(notation[0][1]) == len(notation[2][0])
        assert internal_derivatives[i].ndim == len(notation[0][1])
    # check internal coherence in notations
    for notation in einstein_notations:
        for idx in notation[0][1]:
            assert idx in notation[0][0] or any([idx in sub for sub in notation[1]])
        assert len(notation[0][1]) == len(notation[2][0])
        assert len(notation[0][1]) == len(notation[2][1]), notation[2]
    order: int = len(composed_permutation)
    assert sum([len(notation[1]) for notation in einstein_notations]) == order
    # check expected indeps
    assert len(expected_indeps) == len(einstein_notations)
    assert sum(len(indeps) for indeps in expected_indeps) == order
    # flexible indeps checks
    internal_dual_marker: Union[None, Tuple[bool, ...]] = None
    if flexible_indeps:
        assert len(set(external_shapes)) == 1
        assert len(set((len(nt[2][0]) for nt in einstein_notations))) == 1
        # check that all internal shapes are mutuallly broadcastable
        internal_shape_len: int = set(len(nt[2][0]) for nt in einstein_notations).pop()
        internal_shape_aux: list[int] = [1 for _ in range(internal_shape_len)]
        for notation in einstein_notations:
            assert len(set((tuple(sub) for sub in notation[1]))) == 1  # check that
            #   that all intra-notation composed indices are equal
            for d, size in enumerate(notation[2][0]):
                assert internal_shape_aux[d] in (1, size)
                internal_shape_aux[d] = max(internal_shape_aux[d], size)
        # check that all dual dimensions to be included in indeps occupy the same
        #   relative positions accross all composed indices
        internal_dual_markers: list[Tuple[bool, ...]] = list()
        composed_dual_markers: list[Tuple[bool, ...]] = list()
        for notation in einstein_notations:
            marker: Tuple[bool, ...]
            marker = tuple(
                idx not in notation[0][0] and _bool(notation[2][1][i])
                for i, idx in enumerate(notation[0][1])
            )
            internal_dual_markers.append(marker)
            for composed_sub_indices in notation[1]:
                marker = tuple(i not in notation[0][0] for i in composed_sub_indices)
                composed_dual_markers.append(marker)
        assert len(set(internal_dual_markers)) == 1
        # assert len(internal_dual_markers[0]) > 0
        assert len(set(composed_dual_markers)) == 1
        # assert len(composed_dual_markers[0]) > 0
        internal_dual_marker = internal_dual_markers[0]

    ### Create global lists for symbolic indices and tensors
    # create symbolic independent indices
    external_indep_syms: list[SymIndex] = list()
    for size in indep_sizes:
        sym: SymIndex = SymIndex()
        sym.assert_size(size=size)
        external_indep_syms.append(sym)
    # create independent included symbolic indices
    included_indep_syms: list[SymIndex] = list()
    if flexible_indeps and effective_order == 1:
        assert internal_dual_marker is not None
        included_indep_syms = [SymIndex() for k in internal_dual_marker if k]
    # create list for accumulating the rest of indicesr
    external_syms: list[list[SymIndex]] = list()
    internal_syms: list[list[SymIndex]] = list()
    spreaded_syms: list[list[SymIndex]] = list()
    excluded_syms: list[list[SymIndex]] = [list() for _ in external_indep_syms]
    composed_syms: list[list[SymIndex]] = list()
    raw_excluded_syms: list[list[SymIndex]] = list()
    raw_composed_syms: list[list[SymIndex]] = list()
    sep_composed_indep_syms: list[list[SymIndex]] = list()  # separated (by partiality)
    # create list for accumulating internal tensors
    internal_tensors: list[Tensor] = list()
    spreaded_tensors: list[Tensor] = list()

    ### Fill the lists with local symbolic indices and tensors
    # for each einstein notation -> accumulate corresponding symbolic indices
    for i, notation in enumerate(einstein_notations):
        # assign sizes to independent included symbolic indices
        included_indep_inds: list[int] = list()
        if flexible_indeps and effective_order == 1:
            assert internal_dual_marker is not None
            included_indep_inds.extend(
                [notation[0][1][j] for j, k in enumerate(internal_dual_marker) if k]
            )
            for idx, sym in zip(included_indep_inds, included_indep_syms):
                j: int = notation[0][1].index(idx)
                sym.assert_size(size=notation[2][0][j], override=True)
        # create interface to handle indices symbolically
        symbolizer: SymInterface = SymInterface(
            notation=notation,
            external_indep=external_indeps[i],
            external_indep_syms=external_indep_syms,
            expected_indeps=expected_indeps[i],
            included_indep_inds=included_indep_inds,
            included_indep_syms=included_indep_syms,
        )
        # insert indices for external partialities (for external_derivative)
        symbolizer.size_external_syms(external_shape=external_shapes[i])
        # insert indices for internal tensors (for internal derivatives and identies)
        symbolizer.size_internal_syms(internal_shape=notation[2][0])
        # remove / replace indices corresponding to independent dimensions
        symbolizer.remove_external_indeps()
        symbolizer.remove_composed_indeps()
        # extend global lists of SymIndex(s) aggregating local ones
        external_syms.append(symbolizer.mod_external_syms)
        internal_syms.append(symbolizer.mod_internal_syms)
        spreaded_syms.extend(symbolizer.link_spreaded_syms())
        excluded_syms = symbolizer.extend_excluded_syms(identity_syms=excluded_syms)
        composed_syms.extend(symbolizer.mod_composed_syms)
        raw_composed_syms.extend(symbolizer.raw_composed_syms)
        raw_excluded_syms.extend(symbolizer.excluded_indep_syms)
        sep_composed_indep_syms.extend(symbolizer.composed_indep_syms)
        # extend global lists of Tensor(s) aggregating local ones
        internal_tensors.append(internal_derivatives[i])

    ### Construct the array of composed independent symbolic indices
    # create aux structure to easily match external partiality with composed partiality
    # e.g. cmp_partial_idx <- partial_match[ext_partial_idx]
    match: list[int] = [i for i, nt in enumerate(einstein_notations) for _ in nt[1]]
    # construct the array
    composed_indep_syms: list[SymIndex] = list()
    for i, sym in enumerate(external_indep_syms):
        # if not all(sym in syms for syms in raw_excluded_syms):
        if not all(
            sym in syms or external_indeps[match[j]][i] is None
            for j, syms in enumerate(raw_excluded_syms)
        ):
            composed_indep_syms.append(sym)
    composed_indep_syms.extend(included_indep_syms)

    ### Add excluded syms (& links) and identities to corresponding spread arrays
    for i, (sym, syms) in enumerate(zip(external_indep_syms, excluded_syms)):
        # check coherence between raw_excluded_syms and excluded syms
        #   (<=) because there are 2 types of excluded indeps
        #       1. excluded from indep and distributed to subcomposed (sym_count += 1)
        #       2. excluded from indep and not distributed to subcomposed (... += 0)
        sym_count: int = sum(sym in raw_syms for raw_syms in raw_excluded_syms)
        # aggregate syms
        if len(syms) > 0:
            spreaded_syms.append([sym, *syms])

    ### Reorder output partialities as indicated by variables
    composed_syms = [composed_syms[i] for i in composed_permutation]

    ### Contract external derivatives againts internal derivatives
    primal_sym: SymIndex = SymIndex()
    primal_sym.assert_size(size=primal_size)
    external_head_syms: list[list[SymIndex]] = [[primal_sym], external_indep_syms]
    composed_head_syms: list[list[SymIndex]] = [[primal_sym], composed_indep_syms]
    nested_indices: list[list[list[SymIndex]]] = [
        external_head_syms,
        external_syms,
        internal_syms,
        spreaded_syms,
        composed_head_syms,
        composed_syms,
    ]
    _numerize_indices(nested_indices=nested_indices)
    composed_derivative: Tensor = _contract(
        nested_indices=nested_indices,
        external_tensor=external_derivative,
        internal_tensors=internal_tensors,
        dtype=dtype,
        device=device,
    )

    # Add size 1 dimensions to X and
    composed_view: Tuple[int, ...] = tuple(composed_derivative.shape)
    head_new_view: list[int] = [
        primal_size,
        *(sym.size if sym in composed_indep_syms else 1 for sym in external_indep_syms),
        *(sym.size for sym in included_indep_syms),
    ]
    new_view: Tuple[int, ...]
    new_view = (*head_new_view, *composed_view[(len(composed_indep_syms) + 1) :])
    composed_derivative = composed_derivative.reshape(shape=new_view)

    ### Calculate new shapes and indeps
    composed_shapes: Tuple[Shape, ...] = calculate_composed_shapes(
        composed_permutation=composed_permutation,
        raw_composed_syms=raw_composed_syms,
    )
    composed_indeps: Tuple[Indep, ...] = calculate_composed_indeps(
        composed_permutation=composed_permutation,
        external_indep_syms=external_indep_syms,
        included_indep_syms=included_indep_syms,
        composed_indep_syms=sep_composed_indep_syms,
        raw_composed_syms=raw_composed_syms,
    )

    if flexible_indeps:
        mut_reduced_indeps: list[list[Union[None, int]]]
        # mut_reduced_indeps = [list(indep) for indep in composed_indeps]
        mut_reduced_indeps = [list() for _ in composed_indeps]
        removals: int = 0
        for indepT in zip(*composed_indeps):
            if all(d is not None for d in indepT):
                for i, mut_indep in enumerate(mut_reduced_indeps):
                    mut_indep.append(indepT[i])
        composed_indeps: Tuple[Indep, ...] = tuple(
            tuple(mut) for mut in mut_reduced_indeps
        )
        reduced_view: Tuple[int, ...] = calculate_shapes(
            first_size=composed_derivative.shape[0],
            variables=tuple(range(len(composed_permutation))),
            shapes=composed_shapes,
            indeps=composed_indeps,
            indeps_squeezed=False,
        )[0]

        composed_derivative = composed_derivative.reshape(shape=reduced_view)

    return (composed_derivative, composed_shapes, composed_indeps)
