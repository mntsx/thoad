# python 3.12

# Standard Library dependencies
from typing import Optional, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.backprop.symbolic.structure import SumGroup
from thoad.autodifferentiation.engine.backprop.composition.combination import (
    generate_permutation_keys,
    produce_variations,
)
from thoad.autodifferentiation.engine.backprop.symbolic.construction import (
    assemble_symbolic_composition,
)
from thoad.autodifferentiation.engine.backprop.composition.indexation import (
    SymIndex,
    LinkedSymIndex,
    numerize_indices,
    symbolize_notation,
)
from thoad.autodifferentiation.engine.backprop.composition.validation import (
    check_external_differentials,
    check_internal_differentials,
    check_variables,
)
from thoad.typing.data import Shape, Indep, Notation


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


def _contract(
    external_tensor: Tensor,
    internal_tensors: list[Tensor],
    nested_indices: list[list[list[SymIndex]]],
) -> Tensor:

    def _insert_numerized(syms: list[SymIndex]) -> Tuple[int, ...]:
        return tuple(sym.id for sym in syms)

    def _flatten(seq: Sequence[Sequence[int]]) -> Tuple[int]:
        return tuple(x for sub in seq for x in sub)

    ### Sort einsum args
    einsum_args: list[Union[Tensor, Tuple[int, ...]]] = [external_tensor]
    external_syms: list[list[SymIndex]]
    external_syms = [*_flatten(nested_indices[0]), *_flatten(nested_indices[1])]
    einsum_args.append(_insert_numerized(external_syms))
    for tensor, syms in zip(internal_tensors, nested_indices[2]):
        if tensor.ndim > 0:
            einsum_args.append(tensor)
            einsum_args.append(_insert_numerized(syms))
    output_syms: list[list[SymIndex]]
    output_syms = [*_flatten(nested_indices[0]), *_flatten(nested_indices[3])]
    einsum_args.append(_insert_numerized(output_syms))

    ### Compute composed differential
    result: Tensor = torch.einsum(*einsum_args)

    return result


def _contract_differentials(
    variable_indices: Tuple[int, ...],
    external_differential: Tensor,
    shapes: Tuple[Shape, ...],
    indeps: Tuple[Indep, ...],
    internal_differentials: Tuple[Tensor, ...],
    einstein_notations: Sequence[Tuple[Sequence[int], Sequence[Sequence[int]]]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]:

    ### Constants
    XX: int = external_differential.shape[0]

    ### Checks
    # check notation coherence with shapes and indeps
    distributed_shapes: list[list[int]] = [list() for _ in shapes]
    sh_in_nt: enumerate = enumerate(zip(shapes, indeps, einstein_notations))
    for i, (shape, indep, notation) in sh_in_nt:
        distributed_shapes[i] = [sz for j, sz in enumerate(shape) if j not in indep]
    # check coherence between indeps
    indep_sizes: list[int] = list()
    for i, row in enumerate(zip(*indeps)):
        row_sizes: list[int, ...]
        row_sizes = [shapes[j][d] for j, d in enumerate(row) if d is not None]
        indep_sizes.append(max([1, *row_sizes]))
    # check coherence between notation and shapes
    for i, (shape, notation) in enumerate(zip(distributed_shapes, einstein_notations)):
        assert len(shapes[i]) == len(notation[0][0])
        assert len(notation[0][1]) == len(notation[2][0])
        assert internal_differentials[i].ndim == len(notation[0][1])
    # check internal coherence in notations
    for notation in einstein_notations:
        for idx in notation[0][1]:
            assert idx in notation[0][0] or any([idx in sub for sub in notation[1]])
    order: int = len(variable_indices)
    assert sum([len(notation[1]) for notation in einstein_notations]) == order

    ### Obtain symbolic indices
    # obtain symbolic independent indices
    symbolic_independent: list[SymIndex] = list()
    for size in indep_sizes:
        sym_idx: SymIndex = SymIndex()
        sym_idx.assert_size(size=size)
        symbolic_independent.append(sym_idx)
    # create list for accumulating the rest of indices
    symbolic_external: list[list[SymIndex]] = list()
    symbolic_internal: list[list[SymIndex]] = list()
    symbolic_output: list[list[SymIndex]] = list()
    x_symbolic_output: list[list[SymIndex]] = list()
    # create list for accumulating internal tensors
    internal_tensors: list[Tensor] = list()
    # for each einstein notation -> accumulate corresponding symbolic indices
    for i, notation in enumerate(einstein_notations):
        # symbolize notation
        len_in: int = len(notation[0])
        symbolic_notation: list[list[SymIndex]]
        symbolic_notation = symbolize_notation(
            notation=[*notation[0], *notation[1]],
        )
        symbolic_notation_in: list[list[SymIndex]] = symbolic_notation[:len_in]
        symbolic_notation_out: list[list[SymIndex]] = symbolic_notation[len_in:]
        x_symbolic_notation_out: list[list[SymIndex]]
        x_symbolic_notation_out = [[sym for sym in nt] for nt in symbolic_notation_out]
        # insert indices for external partialities (for external_differential)
        for size, sym in zip(shapes[i], symbolic_notation_in[0]):
            sym.assert_size(size=size)
        symbolic_external.append(symbolic_notation_in[0])
        # insert indices for internal tensors (for internal differentials and identies)
        internal_tensors.append(internal_differentials[i])
        for sym, size in zip(symbolic_notation_in[1], notation[2][0]):
            sym.assert_size(size=size, allow_broadcasting=True)
        symbolic_internal.append(symbolic_notation_in[1])
        # remove independent dimension indices
        syms_to_remove: list[SymIndex] = list()
        for j, dim in enumerate(indeps[i]):
            if dim is not None:
                sym_a: SymIndex = symbolic_independent[j]
                sym_b: SymIndex = symbolic_notation_in[0][dim]
                syms_to_remove.append(sym_b)
                for k, subnotation in enumerate(symbolic_notation_out):
                    if sym_b in subnotation:
                        idx: int = x_symbolic_notation_out[k].index(sym_b)
                        x_symbolic_notation_out[k][idx] = sym_a
                        subnotation.remove(sym_b)
        for sym in syms_to_remove:
            symbolic_notation_in[0].remove(sym)
        # insert identities for batch distribution
        syms_to_distribute: list[SymIndex] = [sym for sym in symbolic_independent]
        for subnotation in symbolic_notation_in:
            for sym in subnotation:
                sym_in_output: bool = all(sym in sub for sub in symbolic_notation_out)
                sym_not_distributed: bool = sym not in syms_to_distribute
                require_distribution: bool = len(symbolic_notation_out) > 1
                if sym_in_output and sym_not_distributed and require_distribution:
                    syms_to_distribute.append(sym)
                    size: int = sym.size
                    symbolic_batch: list[SymIndex] = [sym]
                    for subsubnotation in symbolic_notation_out:
                        new_sym: SymIndex = LinkedSymIndex(sym_index=sym)
                        subsubnotation[subsubnotation.index(sym)] = new_sym
                        symbolic_batch.append(new_sym)
                    symbolic_internal.append(symbolic_batch)
                    identity: Tensor = construct_nd_identity(
                        n=size, ndim=len(symbolic_batch), dtype=dtype, device=device
                    )
                    internal_tensors.append(identity)
        # insert indices for symbolic output and extended symbolic output
        symbolic_output.extend(symbolic_notation_out)
        x_symbolic_output.extend(x_symbolic_notation_out)

    ### Reorder output partialities as indicated by variables
    reordered_symbolic_output: list[list[SymIndex]] = list()
    for i in variable_indices:
        reordered_symbolic_output.append(symbolic_output[i])
    symbolic_output = reordered_symbolic_output

    ### Contract
    XX_sym_index: SymIndex() = SymIndex()
    XX_sym_index.assert_size(size=XX)
    symbolic_start: list[list[SymIndex]] = [[XX_sym_index], symbolic_independent]
    nested_indices: list[list[list[SymIndex]]] = [
        symbolic_start,
        symbolic_external,
        symbolic_internal,
        symbolic_output,
    ]
    numerize_indices(nested_indices=nested_indices)
    composed_differential: Tensor = _contract(
        external_tensor=external_differential,
        internal_tensors=internal_tensors,
        nested_indices=nested_indices,
    )

    ### Calculate new shapes and indeps
    numerated_differential_positions: list[int] = list()
    for i, notation in enumerate(einstein_notations):
        numerated_differential_positions.extend([i for _ in notation[1]])
    new_shapes: list[list[int]] = [list() for _ in variable_indices]
    new_indeps: list[list[int]] = [list() for _ in variable_indices]
    # check that independent dims are always in all outputs; this must always pass
    #   otherwise some indep dim is int and should be None
    for i, out in enumerate(x_symbolic_output):
        position: int = numerated_differential_positions[i]
        for j, sym in enumerate(symbolic_independent):
            assert sym in out if indeps[position][j] else True
    # reconstruct new shapes and indeps
    for i, idx in enumerate(variable_indices):
        subnotation: list[SymIndex] = x_symbolic_output[idx]
        for sym in subnotation:
            new_shapes[i].append(sym.size)
        position: int = numerated_differential_positions[i]
        for dim, sym in zip(indeps[position], symbolic_independent):
            if dim is None:
                new_indeps[i].append(None)
            else:
                new_indeps[i].append(subnotation.index(sym))

    composed_shapes: Tuple[Tuple[int, ...], ...]
    composed_shapes = tuple(tuple(shape) for shape in new_shapes)
    composed_indeps: Tuple[Tuple[int, ...], ...]
    composed_indeps = tuple(tuple(indep) for indep in new_indeps)

    return (composed_differential, composed_shapes, composed_indeps)


def compose_differentials(
    variables: Tuple[int, int, Tuple[int, ...]],
    external_differentials: dict[Tuple[int, ...], Union[None, Tensor]],
    external_shapes: dict[Tuple[int, ...], Shape],
    external_indeps: dict[Tuple[int, ...], Indep],
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]],
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation],
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = torch.device("cpu"),
) -> Tuple[Tensor, dict[int, Shape], dict[int, Indep]]:

    ### Run argument checks
    check_variables(variables=variables)
    check_external_differentials(
        variables=variables,
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    check_internal_differentials(
        variables=variables,
        internal_differentials=internal_differentials,
        einstein_notations=einstein_notations,
    )

    ### Precalculations & Definitions
    order: int = len(variables[2])
    expression: SumGroup = assemble_symbolic_composition(order=order)

    ### Compute Compostitions
    # ---
    composed_differential: Union[None, Tensor] = None
    composed_shapes: Union[None, Tuple[Shape, ...]] = None
    composed_indeps: Union[None, Tuple[Indep, ...]] = None
    # iterate over contractions
    for contraction in expression.products:
        # ---
        sum_variations: Union[None, Tensor] = None
        sum_shapes: Union[None, Tuple[Shape, ...]] = None
        sum_indeps: Union[None, Tuple[Indep, ...]] = None
        # get external differential order & patialities
        ct_order: int = contraction.partials[0].order
        ext_partialities: Tuple[int, ...] = contraction.partials[0].tpdims
        # iterate over external variable variations
        external_variations: list[list[int]]
        external_variations = produce_variations(
            elements=range(variables[0]), size=ct_order
        )
        for external_variation in external_variations:

            ### Retrieve differentials & permutations
            external_tensor: Union[None, Tensor]
            external_tensor = external_differentials.get(external_variation, None)
            shapes: list[Tuple[int, ...]] = list()
            indeps: list[Tuple[Union[None, int], ...]] = list()
            var_indices: list[Tuple[int, ...]] = list()
            internal_tensors: list[Tensor] = list()
            variation_notations: list[Tuple[Tuple[Tuple[int]], Tuple[int]]] = list()
            zipped: zip = zip(ext_partialities, contraction.partials[1:])
            for ext_var_idx, int_partial in zipped:
                # build key
                ext_var: int = external_variation[ext_var_idx]
                dims: Tuple[int, ...] = int_partial.tpdims
                var_indices.extend(dims)
                indexed_vars: Tuple[int, ...] = tuple([variables[2][d] for d in dims])
                key: Tuple[int, Tuple[int, ...]] = (ext_var, indexed_vars)
                # retrieve external shapes and indeps
                shapes.append(external_shapes[ext_var])
                indeps.append(external_indeps[ext_var])
                # retrieve internal differential & einsum_notations
                internal_tensors.append(internal_differentials[key])
                variation_notations.append(einstein_notations[key])
            ### Compute contraction
            null: bool = external_tensor is None or None in internal_tensors
            if not null:  # and matched_partialities:
                contracted_tensor: Tensor
                contracted_shapes: Tuple[Shape, ...]
                contracted_indeps: Tuple[Shape, ...]
                (contracted_tensor, contracted_shapes, contracted_indeps) = (
                    _contract_differentials(
                        variable_indices=var_indices,
                        external_differential=external_tensor,
                        shapes=tuple(shapes),
                        indeps=tuple(indeps),
                        internal_differentials=internal_tensors,
                        einstein_notations=variation_notations,
                        dtype=dtype,
                        device=device,
                    )
                )

                ### Sum over variations (of each contraction)
                if sum_variations is None:
                    sum_variations = contracted_tensor
                    sum_shapes = contracted_shapes
                    sum_indeps = contracted_indeps
                else:
                    sum_variations += contracted_tensor
                    assert sum_shapes == contracted_shapes
                    assert sum_indeps == contracted_indeps

        ### Sum over contranctions (= composition)
        if composed_differential is None and sum_variations is not None:
            composed_differential = sum_variations
            composed_shapes = sum_shapes
            composed_indeps = sum_indeps
        elif sum_variations is not None:
            composed_differential += sum_variations
            assert composed_shapes == sum_shapes
            assert composed_indeps == sum_indeps

    ### Adapt shapes and indeps
    shapes_dict: Union[None, dict[int, Shape]] = None
    indeps_dict: Union[None, dict[int, Indep]] = None
    if composed_differential is not None:
        shapes_dict = {v: shape for v, shape in zip(variables[2], composed_shapes)}
        indeps_dict = {v: indep for v, indep in zip(variables[2], composed_indeps)}

    return (composed_differential, shapes_dict, indeps_dict)


class Loader:

    def __init__(
        self,
        external_size: int,
        internal_size: int,
        max_order: int,
        external_differentials: dict[Tuple[int, ...], Union[None, Tensor]],
        external_shapes: dict[int, Shape],
        external_indeps: dict[int, Indep],
        internal_differentials: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]],
        einstein_notations: Optional[
            dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]]
        ] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        self._external_size: int = external_size
        self._internal_size: int = internal_size
        self._max_order: int = max_order
        self._external_differentials: dict[Tuple[int, ...], Union[None, Tensor]]
        self._external_differentials = external_differentials
        self._external_shapes: dict[int, Tuple[int, ...]] = external_shapes
        self._external_indeps: dict[int, Tuple[int | None, ...]] = external_indeps
        self._internal_differentials: dict[
            Tuple[int, Tuple[int, ...]], Union[None, Tensor]
        ]
        self._internal_differentials = internal_differentials
        self._fill_differentials()
        # set calculated atributes
        self._independent_permutation: Union[None, Tuple[int, ...]] = None
        self._einstein_notations: dict[
            Tuple[int, Tuple[int, ...]], Union[None, Notation]
        ]
        if einstein_notations is None:
            self._einstein_notations = dict()
        else:
            self._einstein_notations = einstein_notations
        for key in self._internal_differentials.keys():
            if key not in self._einstein_notations:
                self._einstein_notations[key] = None
        for v in range(external_size):
            if v not in self._external_shapes:
                self._external_shapes[v] = None
            if v not in self._external_indeps or self._external_indeps[v] is None:
                indep_size: int = self._extract_indep_size()
                self._external_indeps[v] = indep_size * (None,)
        # set optional attributes
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device

    def _extract_indep_size(self) -> int:
        return len(set(self._external_indeps.values()).pop())

    def _fill_differentials(self) -> None:
        # fill differential dictionaries with missing keys
        variations: list[Tuple[int, ...]] = list()
        for order in range(1, self._external_size + 1):
            order_variations: list[Tuple[int, ...]] = produce_variations(
                elements=range(order), size=self._external_size
            )
            variations.extend(order_variations)
        for external_key in variations:
            if external_key not in self._external_differentials:
                self._external_differentials[external_key] = None
        internal_keys: Tuple[int, Tuple[int, ...]] = generate_permutation_keys(
            external_size=self._external_size,
            internal_size=self._internal_size,
            max_order=self._max_order,
        )
        for internal_key in internal_keys:
            if internal_key not in self._internal_differentials:
                self._internal_differentials[internal_key] = None
        return None

    @property
    def variables(self) -> Tuple[int, int, Tuple[int, ...]]:
        return self._variables

    @property
    def external_differentials(self) -> dict[Tuple[int, ...], Union[None, Tensor]]:
        return self._external_differentials

    @property
    def internal_differentials(
        self,
    ) -> dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]:
        return self._internal_differentials

    @property
    def einstein_notations(
        self,
    ) -> dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]]:
        return self._einstein_notations

    def register_einstein_notation(
        self,
        key: Tuple[int, Tuple[int, ...]],
        val: Notation,
    ) -> None:
        self._einstein_notations[key] = val
        return None

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def compose(self, variables: Tuple[int, ...]) -> None:
        assert all([v in range(self._internal_size) for v in variables])
        composed_differential: Tensor = compose_differentials(
            variables=(self._external_size, self._internal_size, variables),
            external_differentials=self._external_differentials,
            external_shapes=self._external_shapes,
            external_indeps=self._external_indeps,
            internal_differentials=self._internal_differentials,
            einstein_notations=self._einstein_notations,
            dtype=self._dtype,
            device=self._device,
        )
        return composed_differential
