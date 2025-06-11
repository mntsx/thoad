# Standard Library Dependencies
import itertools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction
from thoad.autodifferentiation.engine.backprop.composition.composition import Loader
from thoad.autodifferentiation.engine.broadcasting.alignment import (
    align_differential,
)
from thoad.autodifferentiation.engine.broadcasting.figuration import (
    compact_differential,
    denull_differential,
)
from thoad.autodifferentiation.engine.control.gradients import (
    DifferentialGrid,
    initialize_differential,
)
from thoad.graph.structures import Node
from thoad.typing.data import EDData, IDData, Notation, Indep, Shape


class IdxMapper:

    def __init__(self, objects: Iterable[Any]) -> None:
        self._idx2obj: dict[int, Any] = {i: o for i, o in enumerate(objects)}
        self._obj2idx: dict[Any, int] = {o: i for i, o in self._idx2obj.items()}

    def obj_to_int(self, obj: Any) -> int:
        assert obj in self._obj2idx
        return self._obj2idx[obj]

    def int_to_obj(self, idx: int) -> Any:
        assert idx in self._idx2obj
        return self._idx2obj[idx]

    def array_to_int(self, objects: Iterable[Any]) -> Tuple[int]:
        assert all(obj in self._obj2idx for obj in objects)
        return tuple([self._obj2idx[obj] for obj in objects])

    def array_to_obj(self, indices: Iterable[int]) -> Tuple[Any]:
        assert all(idx in self._idx2obj for idx in indices)
        return tuple([self._idx2obj[idx] for idx in indices])


class VariableOperator:

    def __init__(
        self,
        fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]],
        grid: DifferentialGrid,
    ) -> None:
        # instrumental attributes for extraction
        self._grid: DifferentialGrid = grid
        self._fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]]
        self._fns = fns
        # variable groups
        self._external_variables: list[Node] = self._extract_external_variables()
        self._internal_variables: list[Node] = self._extract_internal_variables()
        self._off_variables: list[Node]
        self._off_variables = self._extract_off_variables(
            external_variables=self._external_variables
        )
        ev_set: set[Node] = set(self._external_variables)
        iv_set: set[Node] = set(self._internal_variables)
        assert len(ev_set.intersection(iv_set)) == 0

        return None

    def _extract_external_variables(self) -> list[Node]:
        return list({ev for evs, _ in self._fns.values() for ev in evs})

    def _extract_internal_variables(self) -> list[Node]:
        return list({iv for _, ivs in self._fns.values() for iv in ivs})

    def _extract_off_variables(self, external_variables) -> list[Node]:
        off_external_variables: list[Node] = list()
        for ev in self._grid.variables:
            if ev not in external_variables:
                off_external_variables.append(ev)
        return off_external_variables

    @property
    def terminal(self) -> bool:
        pass

    @property
    def evs(self) -> list[Node]:
        return self._external_variables

    @property
    def ivs(self) -> list[Node]:
        return self._internal_variables

    @property
    def ovs(self) -> list[Node]:
        return self._off_variables

    @property
    def all_evs(self) -> list[Node]:
        return [*self._external_variables, *self._off_variables]

    @property
    def all_ivs(self) -> list[Node]:
        return [*self._internal_variables, *self._off_variables]


class GradOperator:

    def __init__(self) -> None:
        # control variable
        self._initialized: bool = False
        # AD initialization attributes
        self._order: int
        self._GO_tensor: Tensor
        # AD configuration attributes
        self._cross_terminals: bool = False
        self._keepbatch: bool = False
        self._terminals: dict[Node, Tensor] = dict()
        self._retentions: set[Tuple[Node, ...]] = set()
        self._hooks: dict[
            Tuple[Node, ...],
            Callable[[EDData, list[dict[str, Any]]], EDData],
        ] = dict()
        # technical requirements
        self._dtype: torch.dtype
        self._device: torch.device
        # gradient management
        self._grid: DifferentialGrid
        self._target_gradients: dict[Tuple[Node, ...], Union[None, EDData]] = dict()

        return None

    @property
    def cross_terminals(self) -> bool:
        return self._cross_terminals

    @cross_terminals.setter
    def cross_terminals(self, value: bool) -> None:
        self._cross_terminals = value

    @property
    def keepbatch(self) -> bool:
        return self._keepbatch

    @keepbatch.setter
    def keepbatch(self, value: bool) -> None:
        self._cross_terminals = value

    @property
    def terminals(self) -> dict[Node, Tensor]:
        return self._terminals

    @terminals.setter
    def terminals(self, value: dict[Node, Tensor]) -> None:
        self._terminals = value

    def add_gradient_retention(
        self,
        key: Tuple[Node, ...],
    ) -> None:
        self._retentions.add(key)
        return None

    def drop_gradient_retention(
        self,
        key: Tuple[Node, ...],
    ) -> None:
        if key in self._retentions:
            self._retentions.pop(key)
        return None

    def add_backward_hook(
        self,
        key: Tuple[Node, ...],
        hook: Callable[[EDData, list[dict[str, Any]]], EDData],
    ) -> None:
        self._hooks[key] = hook
        return None

    def drop_backward_hook(
        self,
        key: Tuple[Node, ...],
    ) -> None:
        if key in self._hooks:
            self._hooks.pop(key)
        return None

    def initialize_gradients(
        self,
        order: int,
        node: Node,
        tensor: Tensor,
    ) -> None:

        # control
        self._initialized = True

        ### Save class attributes relevant for differentiation
        self._order = order
        self._GO_tensor = tensor
        self._dtype = tensor.dtype
        self._device = tensor.device

        ### Initialize differentials
        self._grid = DifferentialGrid()
        for o in range(1, 1 + order):
            self._grid[o * (node,)] = initialize_differential(
                order=o,
                tensor=self._GO_tensor,
                dtype=self._dtype,
                device=self._device,
            )

        return None

    def initialize_retentions(self, order: int, groups: list[set[Node]]) -> None:
        # reset target gradients and redefine expected keys
        self._clear_gradients()
        # save keys of new required retentions
        self._target_gradients = dict()
        for o in range(1, 1 + order):
            for key in itertools.product(self._terminals.keys(), repeat=o):
                if self._cross_terminals:
                    self._target_gradients[key] = None
                else:
                    key_set: set[Node] = set(key)
                    if len(key_set) == 1:
                        self._target_gradients[key] = None
                    for G in groups:
                        if G.intersection(key_set) == key_set:
                            self._target_gradients[key] = None
        for key in self._retentions:
            if len(key) < order:
                self._target_gradients[key] = None

        return None

    def _clear_gradients(self) -> None:
        for tensor in self._terminals.values():
            if "hgrad" in dir(tensor):
                delattr(tensor, "hgrad")
        return None

    def _distribute_batched_gradient(
        self,
        key: Tuple[Node, ...],
        data: EDData,
    ) -> EDData:
        grad: Tensor
        shapes: Tuple[Shape, ...]
        indeps: Tuple[Indep, ...]
        (grad, shapes, indeps) = data
        new_indeps: Tuple[Indep, ...] = tuple(len(indep) * (None,) for indep in indeps)
        variables: Tuple[int, ...] = tuple(
            {v: i for i, v in enumerate(dict.fromkeys(key))}[n] for n in key
        )
        new_grad: Tensor = align_differential(
            differential=grad,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=shapes,
            expected_indeps=new_indeps,
            keepdim=False,
        )
        new_data: EDData = (new_grad, shapes, new_indeps)
        return new_data

    def fetch_hgrad(self, key: Tuple[Node, ...], keepbatch: bool) -> EDData:
        if key not in self._target_gradients:
            raise KeyError("No gradient saved for given key")
        ED_data: EDData = self._target_gradients[key]
        assert ED_data is not None, key
        explicit_ED_data: EDData = (None, None, None)
        if all(d is not None for d in ED_data):
            if not keepbatch:
                differential: Tensor
                shapes: Tuple[Shape]
                indeps: Tuple[Indep]
                (differential, shapes, indeps) = self._distribute_batched_gradient(
                    key=key,
                    data=ED_data,
                )
                variables: Tuple[int, ...]
                variables: Tuple[int, ...] = tuple(
                    {v: i for i, v in enumerate(dict.fromkeys(key))}[n] for n in key
                )
                differential = compact_differential(
                    differential=differential,
                    variables=variables,
                    shapes=shapes,
                    indeps=indeps,
                    indeps_squeezed=True,
                )
                ED_data = (differential, shapes, indeps)
            explicit_shapes: Tuple[Shape, ...]
            explicit_indeps: Tuple[Indep, ...]
            explicit_shapes = tuple(ED_data[1][tuple(set(key)).index(v)] for v in key)
            explicit_indeps = tuple(ED_data[2][tuple(set(key)).index(v)] for v in key)
            explicit_ED_data = (ED_data[0], explicit_shapes, explicit_indeps)
        return explicit_ED_data

    def _save_gradients(
        self,
        gradients: dict[Tuple[Node, ...], EDData],
    ) -> None:
        for key, data in gradients.items():
            if key in self._target_gradients:
                self._target_gradients[key] = data
        return None

    def attach_gradients(self) -> None:
        self._initialized = False
        for node, tensor in self._terminals.items():
            node_gradients: list[Tensor] = list()
            for o in range(1, 1 + self._order):
                key: Tuple[Node, ...] = tuple(node for _ in range(o))
                ED_data: EDData = self._grid[key]
                assert ED_data[0] is not None
                if not self._keepbatch:
                    differential: Tensor
                    shapes: Tuple[Shape]
                    indeps: Tuple[Indep]
                    (differential, shapes, indeps) = self._distribute_batched_gradient(
                        key=key,
                        data=ED_data,
                    )
                    variables: Tuple[int, ...] = tuple(
                        {v: i for i, v in enumerate(dict.fromkeys(key))}[n] for n in key
                    )
                    differential = compact_differential(
                        differential=differential,
                        variables=variables,
                        shapes=shapes,
                        indeps=indeps,
                        indeps_squeezed=True,
                    )
                ED_data = (differential, shapes, indeps)
                node_gradients.append(ED_data[0])
            setattr(tensor, "hgrad", tuple(node_gradients))
        return None

    def _denull_differentials(
        self,
        differentials: dict[Tuple[Node, ...], EDData],
    ) -> dict[Tuple[Node, ...], EDData]:

        denulled_differentials: dict[Tuple[Node, ...], EDData] = dict()
        for key, (tensor, shapes, indeps) in differentials.items():
            ED_data: EDData = (tensor, shapes, indeps)
            if tensor is not None and any(0 in s for s in shapes):
                variables: Tuple[int, ...] = tuple(
                    {v: i for i, v in enumerate(dict.fromkeys(key))}[n] for n in key
                )
                dtype: torch.dtype = tensor.dtype
                device: torch.device = tensor.device
                denull_tensor: Tensor
                denull_shapes: Tuple[Shape, ...]
                denull_indeps: Tuple[Indep, ...]
                (denull_tensor, denull_shapes, denull_indeps) = denull_differential(
                    differential=tensor,
                    variables=variables,
                    shapes=shapes,
                    indeps=indeps,
                    dtype=dtype,
                    device=device,
                )
                ED_data = (denull_tensor, denull_shapes, denull_indeps)
            denulled_differentials[key] = ED_data
        return denulled_differentials

    def _apply_hooks(
        self,
        fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]],
        differentials: dict[Tuple[Node, ...], EDData],
    ) -> None:
        modified_differentials: dict[Tuple[Node, ...], EDData] = dict()
        for key, ED_data in differentials.items():
            if key in self._hooks:
                G: set[Node] = set(key)
                hook: callable[[EDData, list[dict["str", Any]]], EDData]
                hook = self._hooks[key]
                context: list[dict["str", Any]] = list()
                for fn, (fn_evs, _) in fns.items():
                    if len(G.intersection(set(fn_evs))) == len(set(fn_evs)):
                        context.append(fn.context)
                mod_ED_data: EDData = hook(grad_data=ED_data, context=context)
                # checks
                if not isinstance(mod_ED_data, Sequence):
                    raise TypeError(
                        f"Hook for key {key} must return a sequence, "
                        f"but returned {type(mod_ED_data).__name__}."
                    )
                if len(mod_ED_data) != 3:
                    raise ValueError(
                        f"Hook for key {key} must return a sequence of "
                        f"length 3, but length was {len(mod_ED_data)}."
                    )
                if not isinstance(mod_ED_data[0], Tensor):
                    raise TypeError(
                        f"First element of returned ED_data must be a Tensor, "
                        f"but got {type(mod_ED_data[0]).__name__} for key {key}."
                    )
                if ED_data[0].shape != mod_ED_data[0].shape:
                    raise ValueError(
                        f"Shape mismatch in first element for key {key}: original "
                        f"shape {ED_data[0].shape}, modified shape "
                        f"{mod_ED_data[0].shape}."
                    )
                if not isinstance(mod_ED_data[1], Tuple):
                    raise TypeError(
                        f"Second element of returned ED_data must be a Tuple, but "
                        f"got {type(mod_ED_data[1]).__name__} for key {key}."
                    )
                if ED_data[1] != mod_ED_data[1]:
                    raise ValueError(
                        f"Second element of returned ED_data must match original "
                        f"inner nodes for key {key}. Original: {ED_data[1]}, "
                        f"modified: {mod_ED_data[1]}."
                    )
                if not isinstance(mod_ED_data[2], Tuple):
                    raise TypeError(
                        f"Third element of returned ED_data must be a Tuple, but got "
                        f"{type(mod_ED_data[2]).__name__} for key {key}."
                    )
                if not all(isinstance(i, (int, type(None))) for i in mod_ED_data[2]):
                    raise TypeError(
                        f"All elements in the third element of returned ED_data must "
                        f"be int or None for key {key}."
                    )
                for dim in mod_ED_data[2]:
                    if dim is not None and dim not in range(ED_data[1]):
                        raise IndexError(
                            f"Index {dim} in hook result for key {key} is out of "
                            f"valid range 0 to {ED_data[1] - 1}."
                        )
                    if dim is not None and mod_ED_data.count(dim) != 1:
                        raise ValueError(
                            f"Duplicated index {dim} found in third element of "
                            f"returned ED_data for key {key}."
                        )
                # save modified differential data
                modified_differentials[key] = tuple(mod_ED_data)
            else:
                modified_differentials[key] = ED_data

        return modified_differentials

    def _acquire_external_differentials(
        self, V: VariableOperator
    ) -> dict[Tuple[Node, ...], Tensor]:
        """
        Acquire external differentials for the given VariableOperator V.

        Iterates orders from 1 to self._order and builds a mapping of
        differential keys.

        - If at least one involved external variable is in the key,
          queries from the grid all differentials with respect to those
          external variables.

        - Otherwise, fakes differentials with respect to non-involved
          external variables as null tensors. Internal differentials from
          these variables to involved internal variables will be all null,
          so these fake entries have no effect on the final computed
          composed differentials, since in this contractive step only
          differentials with respect to involved internal variables are
          used.
        """
        external_differentials: dict[Tuple[Node, ...], EDData] = dict()
        for o in range(1, self._order + 1):
            for key in itertools.product(V.all_evs, repeat=o):  # ???
                if not all(N not in V.evs for N in key):
                    data = self._grid[key]
                    external_differentials[key] = data[0]
        return external_differentials

    def _acquire_external_shapes(
        self,
        V: VariableOperator,
        fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]],
    ) -> Tuple[dict[Node, Shape], dict[Node, Indep]]:

        ### Collect external assumption of shapes
        external_tensors: dict[Node, Shape] = dict()  # save just 1st order tensors
        external_shapes: dict[Node, Shape] = dict()
        external_indeps: dict[Node, Indep] = dict()
        for N in V.all_evs:
            assert N not in external_shapes
            assert N not in external_indeps
            data: EDData = self._grid[(N,)]
            external_tensors[N] = data[0]
            external_shapes[N] = data[1][0]
            external_indeps[N] = data[2][0]

        ### Resolve shapes & independencies
        expected_shapes: dict[Node, Shape] = dict()
        expected_indeps: dict[Node, Indep] = dict()
        for fn, (fn_evs, _) in fns.items():
            for ev in fn_evs:
                assert ev not in expected_shapes
                tensor_null: bool = external_tensors[ev] is None
                if not tensor_null:
                    shape: Shape = external_shapes[ev]
                    indep: Shape = external_indeps[ev]
                    projection: Tuple[Shape, Indep]
                    projection = fn.check_shape(shape=shape, indep=indep)
                    expected_shapes[ev] = projection[0]
                    expected_indeps[ev] = projection[1]
                else:
                    expected_shapes[ev] = None
                    expected_indeps[ev] = None
        for ov in V.ovs:
            # if ov not in grid: data <- (None, None, None)
            data = self._grid[(ov,)]
            expected_shapes[ov] = data[1][0]
            expected_indeps[ov] = data[2][0]

        return (external_shapes, external_indeps, expected_shapes, expected_indeps)

    def _compute_internal_differentials(
        self,
        V: VariableOperator,
        fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]],
    ) -> Tuple[
        dict[Tuple[Node, Tuple[Node, ...]], Tensor],
        dict[Tuple[Node, Tuple[Node, ...]], Notation],
    ]:
        """
        Compute all internal differentials up to the specified order.

        Steps:
        1. Resolve shapes and independencies for each external variable.
           - Shape projections must converge to one unique shape.
           - Independency projections can be multiple; final independency is the
             logical AND across projections.

        2. Initialize all internal differentials as null tensors.

        3. Substitute non-null tensors for internal-external pairs where a connecting
           function exists.

        4. Handle non-involved external variables:
           - For first-order and matching internal variable, use the identity tensor.
           - Otherwise, leave as null tensor.
        """

        ### Initialize internal differentials
        internal_diffs: dict[Tuple[Node, Tuple[Node, ...]], Tensor] = dict()
        eins_notations: dict[Tuple[Node, Tuple[Node, ...]], Notation] = dict()
        for ev in V.evs:
            for o in range(1, self._order + 1):
                for ivs in itertools.product(V.ivs, repeat=o):
                    internal_diffs[(ev, ivs)] = None
                    eins_notations[(ev, ivs)] = None

        ### Populate with actual differentials from fns
        for fn, (fn_evs, fn_ivs) in fns.items():
            for fn_ev in fn_evs:
                for o in range(1, self._order + 1):
                    for ivs in itertools.product(fn_ivs, repeat=o):
                        int_ev: int = fn_evs.index(fn_ev)
                        int_ivs: Tuple[int, ...] = tuple(fn_ivs.index(v) for v in ivs)
                        ID_data: IDData = fn[(int_ev, int_ivs)]
                        internal_diffs[(fn_ev, ivs)] = ID_data[0]
                        eins_notations[(fn_ev, ivs)] = ID_data[1]

        ### Handle non-involved external variables
        for oev in V.ovs:
            for o in range(1, self._order + 1):
                key: Tuple[Node, Tuple[Node, ...]] = (oev, o * (oev,))
                internal_diffs[key] = None
                eins_notations[key] = None
                if o == 1:
                    data: EDData = self._grid[(oev,)]
                    external_diff: Tensor = data[0]
                    external_shape: Shape = data[1][0]
                    if external_diff is not None:
                        int_diff: Tensor = torch.ones(
                            size=external_shape, dtype=self._dtype, device=self._device
                        )
                        internal_diffs[key] = int_diff
                        external_range: Tuple[int, ...]
                        external_range = tuple(range(len(external_shape)))
                        inputs_indices: list[list[int]]
                        inputs_indices = [external_range, external_range]
                        output_indices: list[list[int]] = [external_range]
                        eins_notations[key] = list()
                        eins_notations[key].append(inputs_indices)
                        eins_notations[key].append(output_indices)
                        eins_notations[key].append([external_shape])

        return (internal_diffs, eins_notations)

    def _align_external_differentials(
        self,
        external_differentials: dict[Tuple[Node, ...], Tensor],
        external_shapes: dict[Node, Shape],
        external_indeps: dict[Node, Indep],
        expected_shapes: dict[Node, Shape],
        expected_indeps: dict[Node, Indep],
    ) -> dict[Tuple[Node, ...], Tensor]:

        aligned_external_differentials: dict[Tuple[Node, ...], Tensor] = dict()
        aligned_shapes: dict[Node, Tuple[Shape, ...]] = dict()
        aligned_indeps: dict[Node, Tuple[Indep, ...]] = dict()
        for variables, tensor in external_differentials.items():
            diff_vars: Tuple[int, ...] = tuple(dict.fromkeys(variables))
            if tensor is not None:
                diff_shapes: Tuple[Shape, ...]
                diff_shapes = tuple([external_shapes[N] for N in diff_vars])
                diff_indeps: Tuple[Indep, ...]
                diff_indeps = tuple([external_indeps[N] for N in diff_vars])
                diff_expected_shapes: Tuple[Shape, ...]
                diff_expected_shapes = tuple([expected_shapes[N] for N in diff_vars])
                diff_expected_indeps: Tuple[Indep, ...]
                diff_expected_indeps = tuple([expected_indeps[N] for N in diff_vars])
                equal_shapes: bool = diff_shapes == diff_expected_shapes
                equal_indeps: bool = diff_indeps == diff_expected_indeps
                data_shift: bool = not equal_shapes or not equal_indeps
                if data_shift:
                    assert len(set(variables) & set(self._terminals.keys())) == 0
                    int_variables: Tuple[int, ...]
                    int_variables = tuple(diff_vars.index(v) for v in variables)
                    aligned_external_differentials[variables] = align_differential(
                        differential=tensor,
                        variables=int_variables,
                        shapes=diff_shapes,
                        indeps=diff_indeps,
                        expected_shapes=diff_expected_shapes,
                        expected_indeps=diff_expected_indeps,
                        keepdim=True,
                    )
                else:
                    aligned_external_differentials[variables] = tensor
                for v in variables:
                    v_idx: int = diff_vars.index(v)
                    aligned_shapes[v] = diff_expected_shapes[v_idx]
                    aligned_indeps[v] = diff_expected_indeps[v_idx]
        # ensure the appearance of all original shapes
        for v in external_shapes:
            if v not in aligned_shapes:
                aligned_shapes[v] = external_shapes[v]
        for v in external_indeps:
            if v not in aligned_indeps:
                aligned_indeps[v] = external_indeps[v]

        return (aligned_external_differentials, aligned_shapes, aligned_indeps)

    def _plan_differentiations(
        self,
        V: VariableOperator,
        groups: list[set[Node]],
    ) -> dict[Tuple[int, ...], bool]:
        ### Resolve which composed differentials need to be computed
        #   (~ which are necesary for considered crossings?)
        #   (in a general sense; ie. including self crossings)
        internal_keys: dict[Tuple[int, ...], bool] = dict()
        for o in range(1, self._order + 1):
            for key in itertools.product(V.all_ivs, repeat=o):
                key_set: set[Node] = set(key)
                require_internal: bool = self._cross_terminals
                if not self._cross_terminals:
                    # compute crossing if all variables are found together in one group
                    for G in groups:
                        if G.intersection(key_set) == key_set:
                            require_internal = True
                all_off: bool = all(N not in V.ivs for N in key)
                if not all_off:
                    internal_keys[key] = require_internal

        return internal_keys

    def _compose_differentials(
        self,
        V: VariableOperator,
        required_differentiations: dict[Tuple[Node, ...], bool],
        mappers: Tuple[IdxMapper, IdxMapper],
        external_differentials: dict[Tuple[Node, ...], Tensor],
        external_shapes: dict[Node, Shape],
        external_indeps: dict[Node, Indep],
        internal_differentials: dict[Tuple[Node, Tuple[Node, ...]], Tensor],
        einstein_notations: dict[Tuple[Node, Tuple[Node, ...]], Optional[Notation]],
    ) -> dict[Tuple[Node, ...], EDData]:

        # modify keys swapping Nodes by integers
        ED: dict[Tuple[int, ...], Union[None, Tensor]] = dict()
        ID: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]] = dict()
        NT: dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]] = dict()
        ES: dict[int, Shape] = dict()
        EI: dict[int, Shape] = dict()
        for E_key, E_diff in external_differentials.items():
            int_E_key: Tuple[int, ...] = mappers[0].array_to_int(objects=E_key)
            ED[int_E_key] = E_diff
        for (pre_I_key, pos_I_key), I_diff in internal_differentials.items():
            int_pre_I_key: int = mappers[0].obj_to_int(obj=pre_I_key)
            int_pos_I_key: Tuple[int, ...] = mappers[1].array_to_int(objects=pos_I_key)
            int_I_key: Tuple[int, Tuple[int, ...]] = (int_pre_I_key, int_pos_I_key)
            ID[int_I_key] = I_diff
            NT[int_I_key] = einstein_notations[(pre_I_key, pos_I_key)]
        for ev in V.all_evs:  # ???
            ev_int_key: int = mappers[0].obj_to_int(obj=ev)
            ES[ev_int_key] = external_shapes[ev]
            EI[ev_int_key] = external_indeps[ev]

        # initialize composition loader
        E_size: int = len(V.all_evs)
        I_size: int = len(V.all_ivs)
        loader: Loader = Loader(
            external_size=E_size,
            internal_size=I_size,
            max_order=self._order,
            external_differentials=ED,
            external_shapes=ES,
            external_indeps=EI,
            internal_differentials=ID,
            einstein_notations=NT,
            dtype=self._dtype,
            device=self._device,
        )
        # compute required differentials
        composed_differentials: dict[Tuple[int, ...], EDData] = dict()
        for C_key, compute in required_differentiations.items():
            C_int_key: Tuple[int, ...] = mappers[1].array_to_int(objects=C_key)
            CD_tensor: Union[None, Tensor] = None
            composed_shapes: Union[None, Tuple[Shape, ...]] = None
            composed_indeps: Union[None, Tuple[Indep, ...]] = None
            if compute:
                dict_shapes: dict[int, Shape]
                dict_indeps: dict[int, Indep]
                CD_tensor, dict_shapes, dict_indeps = loader.compose(
                    variables=C_int_key,
                )
                if CD_tensor is None:
                    continue
                unique_int_key: Tuple[int, ...] = tuple(dict.fromkeys(C_int_key))
                composed_shapes = tuple(dict_shapes[v] for v in unique_int_key)
                composed_indeps = tuple(dict_indeps[v] for v in unique_int_key)
            CD_data: EDData = (CD_tensor, composed_shapes, composed_indeps)
            composed_differentials[C_key] = CD_data

        return composed_differentials

    def _update_grid(
        self,
        V: VariableOperator,
        composed_differentials: dict[Tuple[Node, ...], EDData],
    ) -> None:
        # remove external variable diffs from differential_grid
        for o in range(1, 1 + self._order):
            non_terminal_evs: list[Node]
            non_terminal_evs = [ev for ev in V.evs if ev not in self._terminals]
            self._grid.remove(variables=non_terminal_evs)
        # add new composed differentials to differential_grid
        for key, ED_data in composed_differentials.items():
            o: int = len(key)
            assert o > 0 and o <= self._order
            self._grid[key] = ED_data
        return None

    def contractive_update(
        self,
        fns: dict[ExtendedAutogradFunction, Tuple[Tuple[Node], Tuple[Node]]],
        groups: list[set[Node]],
    ) -> None:

        assert self._initialized

        ### Gather (external & internal) variables
        V: VariableOperator = VariableOperator(fns=fns, grid=self._grid)

        # instantiate mappers
        ev_mapper: IdxMapper = IdxMapper(objects=V.all_evs)
        iv_mapper: IdxMapper = IdxMapper(objects=V.all_ivs)

        ### Acquire external differentials
        external_differentials: dict[Tuple[Node, ...], Tensor]
        external_differentials = self._acquire_external_differentials(V=V)

        ### Aquire external variable shapes
        external_shapes: dict[Tuple[Node], Shape]
        external_indeps: dict[Tuple[Node], Indep]
        expected_shapes: dict[Tuple[Node], Shape]
        expected_indeps: dict[Tuple[Node], Indep]
        (external_shapes, external_indeps, expected_shapes, expected_indeps) = (
            self._acquire_external_shapes(V=V, fns=fns)
        )

        ### Debroadcast (& debatch) uncompatible external differentials
        aligned_external_differentials: dict[Tuple[Node, ...], Tensor]
        aligned_external_shapes: dict[Tuple[Node, ...], Tuple[Shape, ...]]
        aligned_external_indeps: dict[Tuple[Node, ...], Tuple[Indep, ...]]
        (
            aligned_external_differentials,
            aligned_external_shapes,
            aligned_external_indeps,
        ) = self._align_external_differentials(
            external_differentials=external_differentials,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
        )

        ### Compute internal differentials
        internal_differentials: dict[Tuple[Node, Tuple[Node, ...]], Tensor]
        einstein_notations: dict[Tuple[Node, Tuple[Node, ...]], Notation]
        (internal_differentials, einstein_notations) = (
            self._compute_internal_differentials(
                V=V,
                fns=fns,
            )
        )

        ### Determine which composed differentials need to be computed
        required_differentiations: dict[Tuple[Node, ...], bool]
        required_differentiations = self._plan_differentiations(V=V, groups=groups)

        ### Compose differentials
        composed_differentials: dict[Tuple[Node, ...], EDData]
        composed_differentials = self._compose_differentials(
            V=V,
            required_differentiations=required_differentiations,
            mappers=(ev_mapper, iv_mapper),
            external_differentials=aligned_external_differentials,
            external_shapes=aligned_external_shapes,
            external_indeps=aligned_external_indeps,
            internal_differentials=internal_differentials,
            einstein_notations=einstein_notations,
        )

        ### Post-comoposition tasks
        # apply registered backward hooks
        composed_differentials = self._apply_hooks(
            fns=fns,
            differentials=composed_differentials,
        )
        # save gradients if node is terminal or marked by user
        self._save_gradients(gradients=composed_differentials)
        # remove null dimensions from partials for further propagation
        composed_differentials = self._denull_differentials(
            differentials=composed_differentials,
        )

        ### Update grid
        self._update_grid(V=V, composed_differentials=composed_differentials)

        return None

    def direct_update(
        self, fn: ExtendedAutogradFunction, source: Node, target: Node
    ) -> None:
        raise NotImplementedError("direct update is still under development")
        return None
