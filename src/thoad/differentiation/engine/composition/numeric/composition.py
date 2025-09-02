# python 3.12

# Standard Library dependencies
import collections
import itertools
from typing import Iterator, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import thoad.config as config
from thoad.differentiation.engine.broadcasting.alignment import (
    semialign_derivative,
    unify_indeps,
)
from thoad.differentiation.engine.broadcasting.figuration import calculate_shapes
from thoad.differentiation.engine.composition.numeric.combination import (
    produce_variations,
)
from thoad.differentiation.engine.composition.numeric.contraction import (
    contract_derivatives,
)
from thoad.differentiation.engine.composition.numeric.validation import (
    check_external_derivatives,
    check_internal_derivatives,
    check_variables,
)
from thoad.differentiation.engine.composition.symbolic.structure import (
    Partial,
    SumGroup,
)
from thoad.differentiation.engine.composition.symbolic.construction import (
    assemble_symbolic_composition,
)
from thoad.differentiation.engine.control.symmetry import reverse_permutation
from thoad.typing import Shape, Indep, Notation, VPerm


type AlignmentKey = Tuple[Tuple[int, ...], Tuple[Shape, ...], Tuple[Indep, ...]]
type InternalKey = Tuple[int, Tuple[int, ...]]


class AlignmentCache:

    def __init__(self) -> None:
        self._cache: dict[
            AlignmentKey,
            Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]],
        ] = dict()
        return None

    def obtain_alignment(
        self,
        variables: Tuple[int, ...],  # external variables
        derivative: Tensor,
        baseline_shapes: Tuple[Shape, ...],
        baseline_indeps: Tuple[Indep, ...],
        expected_shapes: Tuple[Shape, ...],
    ) -> Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]:
        # checks
        assert isinstance(variables, Tuple)
        assert all(isinstance(v, int) for v in variables)
        assert isinstance(baseline_shapes, Tuple)
        assert len(baseline_shapes) == len(variables)
        assert all(isinstance(shape, Tuple) for shape in baseline_shapes)
        assert all(isinstance(sz, int) for shape in baseline_shapes for sz in shape)
        assert isinstance(baseline_indeps, Tuple)
        assert len(baseline_indeps) == len(variables)
        assert len(set(len(indep) for indep in baseline_indeps)) == 1
        assert all(isinstance(indep, Tuple) for indep in baseline_indeps)
        assert all(isinstance(d, (type(None), int)) for i in baseline_indeps for d in i)
        assert isinstance(expected_shapes, Tuple)
        assert len(expected_shapes) == len(variables)
        assert all(isinstance(shape, Tuple) for shape in expected_shapes)
        assert all(isinstance(sz, int) for shape in baseline_shapes for sz in shape)
        # check if expected derivative has already been computed
        alignment_key: AlignmentKey = (variables, baseline_shapes, baseline_indeps)
        alignement_cached: bool = alignment_key in self._cache
        # fetch / compute the expected derivative
        ED_data: Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]
        if alignement_cached:
            ED_data = self._cache[alignment_key]
        else:
            aligned_derivative: Union[None, Tensor]
            aligned_shapes: Union[None, Tuple[Shape, ...]]
            aligned_indeps: Union[None, Tuple[Indep, ...]]
            (aligned_derivative, aligned_shapes, aligned_indeps) = semialign_derivative(
                derivative=derivative,
                variables=range(len(variables)),
                shapes=baseline_shapes,
                indeps=baseline_indeps,
                expected_shapes=expected_shapes,
                keepdim=True,
            )
            assert aligned_derivative is not None
            assert aligned_shapes is not None
            assert aligned_shapes == expected_shapes
            assert aligned_indeps is not None
            ED_data = (aligned_derivative, aligned_shapes, aligned_indeps)
            self._cache[alignment_key] = ED_data

        return ED_data


class Harmonizer:

    def __init__(self) -> None:
        self._computed: bool = False
        self._internal_variables: set[int] = set()
        self._iv_shapes_map: dict[int, list[Shape]] = collections.defaultdict(list)
        self._iv_indeps_map: dict[int, list[Indep]] = collections.defaultdict(list)
        return None

    def update(
        self,
        flat_var_transitions: Tuple[Tuple[int, int], ...],
        variable_shape_map: dict[int, Shape],
        variable_indep_map: dict[Tuple[int, int], Indep],
    ) -> None:
        self._computed = False
        for _, iv in flat_var_transitions:
            self._internal_variables.add(iv)
        for (ev, iv), indep in variable_indep_map.items():
            self._iv_shapes_map[iv].append(variable_shape_map[ev])
            self._iv_indeps_map[iv].append(indep)
        return None

    def obtain_harmonized_indeps(self) -> dict[int, Indep]:
        iv_Hindep_map: dict[int, Indep] = dict()
        for iv in dict.fromkeys(self._internal_variables):
            iv_Hindep_map[iv] = unify_indeps(
                indeps=self._iv_indeps_map[iv],
                shapes_ndims=tuple(len(shape) for shape in self._iv_shapes_map[iv]),
                inclusive=False,
            )
        return iv_Hindep_map


class Variation:

    def __init__(
        self,
        composed_permutation: Sequence[int],
        variable_transitions: Sequence[Sequence[Tuple[int, int]]],
        aligned_external_derivative: Tensor,
        aligned_external_shapes: Sequence[Shape],
        aligned_external_indeps: Sequence[Indep],
        internal_derivatives: Sequence[Tensor],
        einstein_notations: Sequence[Notation],
    ) -> None:

        self._composed_permutation: Sequence[int] = composed_permutation
        self._variable_transitions: Sequence[Sequence[Tuple[int, int]]]
        self._variable_transitions = variable_transitions
        self._aligned_external_derivative: Tensor = aligned_external_derivative
        self._aligned_external_shapes: Sequence[Shape] = aligned_external_shapes
        self._aligned_external_indeps: Sequence[Indep] = aligned_external_indeps
        self._internal_derivatives: Sequence[Tensor] = internal_derivatives
        self._einstein_notations: Sequence[Notation] = einstein_notations

        return None

    def contract(
        self,
        harmonized_indeps: dict[int, Indep],
        effective_order: int,
        flexible_indeps: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]:
        expected_indeps: Tuple[Tuple[Indep, ...], ...] = tuple(
            tuple(harmonized_indeps[iv] for (_, iv) in sub_transitions)
            for sub_transitions in self._variable_transitions
        )
        # contract derivatives
        contracted_tensor: Tensor
        contracted_shapes: Tuple[Shape, ...]
        contracted_indeps: Tuple[Indep, ...]
        (contracted_tensor, contracted_shapes, contracted_indeps) = (
            contract_derivatives(
                composed_permutation=tuple(self._composed_permutation),
                external_derivative=self._aligned_external_derivative,
                external_shapes=tuple(self._aligned_external_shapes),
                external_indeps=tuple(self._aligned_external_indeps),
                expected_indeps=expected_indeps,
                internal_derivatives=tuple(self._internal_derivatives),
                einstein_notations=self._einstein_notations,
                effective_order=effective_order,
                flexible_indeps=flexible_indeps,
                dtype=dtype,
                device=device,
            )
        )
        # check coherence of composed derivative shape with shapes and indeps
        expected_view: Tuple[int, ...]
        expected_view, _ = calculate_shapes(
            first_size=contracted_tensor.shape[0],
            variables=tuple(range(len(self._composed_permutation))),
            shapes=contracted_shapes,
            indeps=contracted_indeps,
            indeps_squeezed=False,
        )
        assert contracted_tensor.shape == expected_view

        return (contracted_tensor, contracted_shapes, contracted_indeps)


class VariationGroup:

    def __init__(self) -> None:
        self._variations: list[Variation] = list()
        return None

    @property
    def empty(self) -> bool:
        return len(self._variations) == 0

    def add_variation(
        self,
        composed_permutation: Sequence[int],
        variable_transitions: Sequence[Sequence[Tuple[int, int]]],
        aligned_external_derivative: Tensor,
        aligned_external_shapes: Sequence[Shape],
        aligned_external_indeps: Sequence[Indep],
        internal_derivatives: Sequence[Tensor],
        einstein_notations: Sequence[Notation],
    ) -> None:
        variation: Variation = Variation(
            composed_permutation=composed_permutation,
            variable_transitions=variable_transitions,
            aligned_external_derivative=aligned_external_derivative,
            aligned_external_shapes=aligned_external_shapes,
            aligned_external_indeps=aligned_external_indeps,
            internal_derivatives=internal_derivatives,
            einstein_notations=einstein_notations,
        )
        self._variations.append(variation)
        return None

    def compose(
        self,
        harmonized_indeps: dict[int, Indep],
        effective_order: int,
        flexible_indeps: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[
        Union[None, Tensor],
        Union[None, Tuple[Shape, ...]],
        Union[None, Tuple[Indep, ...]],
    ]:
        ### Sum over variations (of each contraction)
        variation_tensor: Union[None, Tensor] = None
        variation_shapes: Union[None, Tuple[Shape, ...]] = None
        variation_indeps: Union[None, Tuple[Indep, ...]] = None
        for V in self._variations:
            contracted_tensor: Tensor
            contracted_shapes: Tuple[Shape, ...]
            contracted_indeps: Tuple[Indep, ...]
            (contracted_tensor, contracted_shapes, contracted_indeps) = V.contract(
                harmonized_indeps=harmonized_indeps,
                effective_order=effective_order,
                flexible_indeps=flexible_indeps,
                dtype=dtype,
                device=device,
            )
            if variation_tensor is None:
                variation_tensor = contracted_tensor
                variation_shapes = contracted_shapes
                variation_indeps = contracted_indeps
            else:
                variation_tensor += contracted_tensor
                assert variation_shapes == contracted_shapes
                assert variation_indeps == contracted_indeps

        return (variation_tensor, variation_shapes, variation_indeps)


class ContractionGroup:

    def __init__(
        self,
        effective_order: int,
        flexible_indeps: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._variation_groups: list[VariationGroup] = list()
        self._effective_order: int = effective_order
        self._flexible_indeps: bool = flexible_indeps
        self._harmonizer: Harmonizer = Harmonizer()
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device
        return None

    @property
    def flexible_indeps(self) -> bool:
        return self._flexible_indeps

    def update_harmonizer(
        self,
        transitions: Sequence[Sequence[Tuple[int, int]]],
        shapes: Sequence[Shape],
        expected_indeps: Sequence[Sequence[Indep]],
    ) -> None:
        evs: Tuple[int, ...] = tuple(Ts[0][0] for Ts in transitions)
        v_shape_map: dict[int, Shape] = {ev: S for ev, S in zip(evs, shapes)}
        flat_trans: Tuple[Tuple[int, int], ...]
        flat_trans = tuple(T for Ts in transitions for T in Ts)
        T_indep_map: dict[Tuple[int, int], Indep] = {
            T: I for T, I in zip(flat_trans, (I for Is in expected_indeps for I in Is))
        }
        self._harmonizer.update(
            flat_var_transitions=flat_trans,
            variable_shape_map=v_shape_map,
            variable_indep_map=T_indep_map,
        )
        return None

    def add_variation_group(self, variation_group: VariationGroup) -> None:
        self._variation_groups.append(variation_group)
        return None

    def compose(self) -> Tuple[
        Union[None, Tensor],
        Union[None, Tuple[Shape, ...]],
        Union[None, Tuple[Indep, ...]],
    ]:
        ### Sum over variations (of each contraction)
        composed_tensor: Union[None, Tensor] = None
        composed_shapes: Union[None, Tuple[Shape, ...]] = None
        composed_indeps: Union[None, Tuple[Indep, ...]] = None
        iv_Hindep_map: dict[int, Indep] = self._harmonizer.obtain_harmonized_indeps()
        for VG in self._variation_groups:
            if not VG.empty:
                variation_tensor: Union[None, Tensor]
                variation_shapes: Union[None, Tuple[Shape, ...]]
                variation_indeps: Union[None, Tuple[Indep, ...]]
                variation_tensor, variation_shapes, variation_indeps = VG.compose(
                    harmonized_indeps=iv_Hindep_map,
                    effective_order=self._effective_order,
                    flexible_indeps=self._flexible_indeps,
                    dtype=self._dtype,
                    device=self._device,
                )
                if composed_tensor is None:
                    composed_tensor = variation_tensor
                    composed_shapes = variation_shapes
                    composed_indeps = variation_indeps
                else:
                    composed_tensor += variation_tensor
                    assert composed_shapes == variation_shapes
                    assert composed_indeps == variation_indeps

        return (composed_tensor, composed_shapes, composed_indeps)


def compose_derivatives(
    variables: Tuple[int, int, Tuple[int, ...]],
    effective_order: int,
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]],
    external_shapes: dict[int, Union[None, Shape]],
    external_indeps: dict[int, Union[None, Indep]],
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]],
    expected_shapes: dict[int, Union[None, Shape]],
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]],
    internal_derivatives: dict[InternalKey, Union[None, Tensor]],
    flexible_indeps: bool,
    einstein_notations: dict[InternalKey, Union[None, Notation]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[
    Union[None, Tensor],
    Union[None, dict[int, Shape]],
    Union[None, dict[int, Indep]],
]:

    ### Run argument checks
    if bool(getattr(config, "DEBUG", False)):
        check_variables(variables=variables)
        check_external_derivatives(
            variables=variables,
            external_derivatives=external_derivatives,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
            external_vperms=external_vperms,
        )
        check_internal_derivatives(
            variables=variables,
            internal_derivatives=internal_derivatives,
            einstein_notations=einstein_notations,
        )

    ### Precalculations & Definitions
    order: int = len(variables[2])
    expression: SumGroup = assemble_symbolic_composition(order=order)

    ### Initialize cache of aligned external derivatives
    cache: AlignmentCache = AlignmentCache()

    ### Compute Compostitions
    # ---
    CG: ContractionGroup = ContractionGroup(
        effective_order=effective_order,
        flexible_indeps=flexible_indeps,
        dtype=dtype,
        device=device,
    )
    # iterate over contractions
    for product in expression.products:
        # ---
        VG: VariationGroup = VariationGroup()
        # get external derivative order & patialities
        ct_order: int = product.partials[0].order
        ext_partialities: Tuple[int, ...] = tuple(product.partials[0].dims)
        # iterate over external variable variations
        external_variations: list[Tuple[int, ...]]
        external_variations = produce_variations(
            elements=range(variables[0]), size=ct_order
        )
        external_trios: Iterator[Tuple[Tuple[int, ...], Tensor, Union[None, VPerm]]] = (
            (evs, ED, external_vperms[evs])
            for evs in external_variations
            if (ED := external_derivatives.get(evs, None)) is not None
        )
        for ext_variation, ext_tensor, ext_permutation in external_trios:

            ### Retrieve derivatives & related info (shapes indeps)
            # initialize sequences related variable monitoring
            ext_variables: list[int] = list()
            var_transitions: list[list[Tuple[int, int]]] = list()
            cmp_permutation: list[int] = list()
            # initialize sequences to store current and expected shapes and indeps
            variation_external_shapes: list[Shape] = list()
            variation_external_indeps: list[Indep] = list()
            variation_expected_shapes: list[Shape] = list()
            variation_expected_indeps: list[list[Indep]] = list()
            # initialize sequences to store variation internal derivatives
            internal_tensors: list[Tensor] = list()
            variation_notations: list[Notation] = list()
            # ---
            ext_depermutation: Tuple[int, ...]
            assert ext_permutation is not None
            ext_depermutation = reverse_permutation(permutation=ext_permutation)
            # ---
            int_partialities: Iterator[Partial]
            int_partialities = itertools.islice(product.partials, 1, None)
            internal_pairs: list[Tuple[int, Partial]]
            internal_pairs = list(zip(ext_partialities, int_partialities))
            internal_pairs = [internal_pairs[d] for d in ext_depermutation]
            # instrumental variables to check if there is any null tensor involved
            null: bool = ext_tensor is None
            # continue with execution if external tensor is not null
            if not null:
                # variable to avoid repeated data extraction
                tmp: list[
                    Tuple[
                        int,
                        Tuple[int, ...],
                        Tuple[int, ...],
                        Union[None, Tensor],
                        Union[None, Notation],
                    ]
                ] = list()
                # save info if no involved internal tensor is null
                for ext_var_idx, int_partial in internal_pairs:
                    ext_var: int = ext_variation[ext_var_idx]
                    dims: Tuple[int, ...] = tuple(int_partial.dims)
                    indexed_vars: Tuple[int, ...] = tuple(variables[2][d] for d in dims)
                    key: Tuple[int, Tuple[int, ...]] = (ext_var, indexed_vars)
                    internal_tensor: Union[None, Tensor]
                    internal_tensor = internal_derivatives.get(key)
                    if internal_tensor is None:
                        null = True
                        break
                    notation: Union[None, Notation] = einstein_notations.get(key)
                    tmp.append((ext_var, dims, indexed_vars, internal_tensor, notation))
                # complete information if no null tensor
                if not null:
                    for ext_var, dims, indexed_vars, int_tensor, notation in tmp:
                        # save external variable
                        ext_variables.append(ext_var)
                        # permutation / transitions
                        cmp_permutation.extend(dims)
                        var_transitions.append([(ext_var, iv) for iv in indexed_vars])
                        # external metadata
                        external_shape: Union[None, Shape] = external_shapes[ext_var]
                        assert external_shape is not None
                        variation_external_shapes.append(external_shape)
                        external_indep: Union[None, Indep] = external_indeps[ext_var]
                        assert external_indep is not None
                        variation_external_indeps.append(external_indep)
                        # expected metadata
                        expected_shape: Union[None, Shape] = expected_shapes[ext_var]
                        assert expected_shape is not None
                        variation_expected_shapes.append(expected_shape)
                        partial_indeps: list[Indep] = list()
                        for iv in indexed_vars:
                            expected_indep: Union[None, Indep]
                            expected_indep = expected_indeps[(ext_var, iv)]
                            assert expected_indep is not None
                            partial_indeps.append(expected_indep)
                        variation_expected_indeps.append(partial_indeps)
                        # internal derivative & notation
                        assert int_tensor is not None
                        internal_tensors.append(int_tensor)
                        assert notation is not None
                        variation_notations.append(notation)
            # preprocess tensors if no involved tensor is null
            if not null:
                fix_permutation: Tuple[int, ...] = tuple(cmp_permutation)
                fix_permutation = reverse_permutation(permutation=fix_permutation)
                # align shapes and external derivative
                aligned_derivative: Tensor
                aligned_shapes: Tuple[Shape, ...]
                aligned_indeps: Tuple[Indep, ...]
                assert all(S is not None for S in variation_external_shapes)
                assert all(S is not None for S in variation_expected_shapes)
                (aligned_derivative, aligned_shapes, aligned_indeps) = (
                    cache.obtain_alignment(
                        variables=tuple(ext_variables),
                        derivative=ext_tensor,
                        baseline_shapes=tuple(variation_external_shapes),
                        baseline_indeps=tuple(variation_external_indeps),
                        expected_shapes=tuple(variation_expected_shapes),
                    )
                )
                # harmonize indeps transitioning to the same internal variable
                CG.update_harmonizer(
                    transitions=var_transitions,
                    shapes=aligned_shapes,
                    expected_indeps=variation_expected_indeps,
                )
                # add variation to de variation group
                VG.add_variation(
                    composed_permutation=fix_permutation,
                    variable_transitions=var_transitions,
                    aligned_external_derivative=aligned_derivative,
                    aligned_external_shapes=aligned_shapes,
                    aligned_external_indeps=aligned_indeps,
                    internal_derivatives=internal_tensors,
                    einstein_notations=variation_notations,
                )
        CG.add_variation_group(variation_group=VG)

    ### Compose differnetials
    composed_derivative: Union[None, Tensor]
    composed_shapes: Union[None, Tuple[Shape, ...]]
    composed_indeps: Union[None, Tuple[Indep, ...]]
    composed_derivative, composed_shapes, composed_indeps = CG.compose()

    ### Adapt shapes and indeps
    shapes_dict: Union[None, dict[int, Shape]] = None
    indeps_dict: Union[None, dict[int, Indep]] = None
    if composed_derivative is not None:
        assert composed_shapes is not None
        assert composed_indeps is not None
        shapes_dict = {v: S for v, S in zip(variables[2], composed_shapes)}
        indeps_dict = {v: I for v, I in zip(variables[2], composed_indeps)}

    return (composed_derivative, shapes_dict, indeps_dict)


def determine_indeps_flexibility(
    external_variables: Sequence[int],
    external_shapes: Sequence[Shape],
    internal_flexibilities: Sequence[bool],
    einstein_notations: Sequence[Notation],
) -> bool:

    ### Flexible indeps
    # Conditions for flexible indeps are:
    # 1. all external variables must be the same one
    # 2. all internal variables must have the same shape ->
    #   -> all composed shapes accross all notations must be equal
    # 3. all internal derivatives (across all orders and external partialities) must
    #   be mutually diagonal
    #       (accross same notation)
    #       -> all composed indices must be equal
    #       (accross different notations)
    #       -> all composed indices must have the same len
    #       -> all internal shapes must be reciprocally broadcastable
    #       -> all external shapes must be equal

    flexible: bool = True

    ### Requirement: different external variables
    flexible &= len(set(external_variables)) == 1

    ### Requirement: not all internal variables flexible
    flexible &= all(flex for flex in internal_flexibilities)

    ### Requirements associated to notations
    # all external and composed shapes must be equal
    flexible &= len(set(external_shapes)) == 1
    flexible &= len(set((len(nt[2][0]) for nt in einstein_notations))) == 1
    # all internal shapes must be mutuallly broadcastable
    if flexible:
        internal_shape_len: int = set(len(nt[2][0]) for nt in einstein_notations).pop()
        internal_shape_aux: list[int] = [1 for _ in range(internal_shape_len)]
        for notation in einstein_notations:
            flexible &= len(set((tuple(sub) for sub in notation[1]))) == 1
            for d, size in enumerate(notation[2][0]):
                flexible &= internal_shape_aux[d] in (1, size)
                internal_shape_aux[d] = max(internal_shape_aux[d], size)
    # all dual dimensions must be included in indeps occupy the same
    #   relative positions accross all composed indices
    internal_dual_markers: list[Tuple[bool, ...]] = list()
    composed_dual_markers: list[Tuple[bool, ...]] = list()
    for notation in einstein_notations:
        marker: Tuple[bool, ...]
        marker = tuple(
            bool(idx not in notation[0][0] and notation[2][1][i])
            for i, idx in enumerate(notation[0][1])
        )
        internal_dual_markers.append(marker)
        for composed_sub_indices in notation[1]:
            marker = tuple(i not in notation[0][0] for i in composed_sub_indices)
            composed_dual_markers.append(marker)
    flexible &= len(set(internal_dual_markers)) == 1
    flexible &= len(set(composed_dual_markers)) == 1

    return flexible


class Loader:

    def __init__(
        self,
        external_size: int,
        internal_size: int,
        max_order: int,
        external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]],
        external_shapes: dict[int, Union[None, Shape]],
        external_indeps: dict[int, Union[None, Indep]],
        external_vperms: dict[Tuple[int, ...], Union[None, VPerm]],
        expected_shapes: dict[int, Union[None, Shape]],
        expected_indeps: dict[Tuple[int, int], Union[None, Indep]],
        internal_derivatives: dict[InternalKey, Union[None, Tensor]],
        internal_flexibilities: dict[int, bool],
        einstein_notations: dict[InternalKey, Union[None, Notation]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:

        ### Predefined attributes
        self._external_size: int = external_size
        self._internal_size: int = internal_size
        self._max_order: int = max_order
        self._external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
        self._external_derivatives = external_derivatives
        self._external_vperms: dict[Tuple[int, ...], Union[None, VPerm]]
        self._external_vperms = external_vperms
        self._effective_order: int = self._determine_effective_order()
        self._internal_derivatives: dict[InternalKey, Union[None, Tensor]]
        self._internal_derivatives = internal_derivatives
        self._internal_flexibilities: dict[int, bool] = internal_flexibilities
        self._fill_derivatives()
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device

        ### Processed attributes
        # einstein notations
        self._einstein_notations: dict[InternalKey, Union[None, Notation]]
        self._einstein_notations = dict()
        for key, value in einstein_notations.items():
            self._einstein_notations[key] = value
        for key in self._internal_derivatives.keys():
            if key not in self._einstein_notations:
                self._einstein_notations[key] = None
        # shapes & indeps
        self._external_shapes: dict[int, Union[None, Shape]] = external_shapes
        self._external_indeps: dict[int, Union[None, Indep]] = external_indeps
        self._expected_shapes: dict[int, Union[None, Shape]] = expected_shapes
        self._expected_indeps: dict[Tuple[int, int], Union[None, Indep]]
        self._expected_indeps = expected_indeps
        indep_size: int = self._extract_indep_size()
        for ev in range(external_size):
            # fill shapes
            if ev not in self._external_shapes:
                self._external_shapes[ev] = None
                self._expected_shapes[ev] = None
            # fill external indeps
            external_null: bool = False
            external_undefined: bool = False
            if ev in self._external_indeps:
                external_null = self._external_indeps[ev] is None
            else:
                external_undefined = True
            if external_undefined or external_null:
                self._external_indeps[ev] = indep_size * (None,)
            # fill expected indeps
            for iv in range(internal_size):
                pair: Tuple[int, int] = (ev, iv)
                expected_null: bool = False
                expected_undefined: bool = False
                if pair in self._expected_indeps:
                    expected_null = self._expected_indeps[pair] is None
                else:
                    expected_undefined = True
                if expected_undefined or expected_null:
                    if external_undefined or external_null:
                        self._expected_indeps[pair] = indep_size * (None,)
                    else:
                        self._expected_indeps[pair] = self._external_indeps[ev]

        return None

    def _extract_indep_size(self) -> int:
        assert len(self._external_indeps) > 0
        indep_size: Union[None, int] = None
        for indep in self._external_indeps.values():
            if indep is not None:
                if indep_size is None:
                    indep_size = len(indep)
                else:
                    assert indep_size == len(indep)
        assert indep_size is not None
        return indep_size

    def _fill_derivatives(self) -> None:
        # lazy fill external derivatives (None on demand)
        self._external_derivatives = collections.defaultdict(
            lambda: None, self._external_derivatives
        )
        # lazy fill external variable permutations (None on demand)
        self._external_vperms = collections.defaultdict(
            lambda: None, self._external_vperms
        )
        # lazy fill internal derivatives (None on demand)
        self._internal_derivatives = collections.defaultdict(
            lambda: None, self._internal_derivatives
        )
        return None

    def _determine_indeps_flexibility(
        self,
        internal_variables: Tuple[int, ...],
    ) -> bool:
        # gather data for the determine_indeps_flexibility function
        internal_flexibilities: Tuple[bool, ...] = tuple(
            self._internal_flexibilities[iv] for iv in internal_variables
        )
        expected_shapes: list[Shape] = []
        einstein_notations: list[Notation] = []
        _external: set[int] = set()  # ensure O(1) membership checks
        _internal: set[int] = set(internal_variables)  # ensure O(1) membership checks
        for (ev, ivs), notation in self._einstein_notations.items():
            if not _internal.issuperset(ivs):
                continue
            if ev not in _external:
                _external.add(ev)
                ev_expected_shape: Union[None, Shape] = self._expected_shapes[ev]
                assert ev_expected_shape is not None
                expected_shapes.append(ev_expected_shape)
            if notation is not None:
                einstein_notations.append(notation)
        # determine indeps flexibility
        flexible_indeps: bool = determine_indeps_flexibility(
            external_variables=tuple(_external),
            external_shapes=expected_shapes,
            internal_flexibilities=internal_flexibilities,
            einstein_notations=einstein_notations,
        )

        return flexible_indeps

    def _determine_effective_order(self) -> int:
        effective_order: int = 1
        for evs, diff in self._external_derivatives.items():
            if len(evs) > effective_order and diff is not None:
                effective_order = len(evs)
        return effective_order

    @property
    def external_derivatives(self) -> dict[Tuple[int, ...], Union[None, Tensor]]:
        return self._external_derivatives

    @property
    def internal_derivatives(
        self,
    ) -> dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]:
        return self._internal_derivatives

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

    def compose(self, variables: Tuple[int, ...]) -> Tuple[
        Union[None, Tensor],
        Union[None, dict[int, Shape]],
        Union[None, dict[int, Indep]],
    ]:
        assert all(v in range(self._internal_size) for v in variables)
        flexible_indeps: bool = self._determine_indeps_flexibility(
            internal_variables=variables
        )
        composed_derivative: Tuple[
            Union[None, Tensor],
            Union[None, dict[int, Shape]],
            Union[None, dict[int, Indep]],
        ] = compose_derivatives(
            variables=(self._external_size, self._internal_size, variables),
            effective_order=self._effective_order,
            external_derivatives=self._external_derivatives,
            external_shapes=self._external_shapes,
            external_indeps=self._external_indeps,
            external_vperms=self._external_vperms,
            expected_shapes=self._expected_shapes,
            expected_indeps=self._expected_indeps,
            internal_derivatives=self._internal_derivatives,
            flexible_indeps=flexible_indeps,
            einstein_notations=self._einstein_notations,
            dtype=self._dtype,
            device=self._device,
        )

        return composed_derivative
