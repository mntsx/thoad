# Standard Library Dependencies
import itertools
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import (
    ContractiveFunction,
    DirectFunction,
    ExtendedAutogradFunction,
)
from thoad.differentiation.engine.composition.numeric.composition import Loader
from thoad.differentiation.engine.broadcasting.alignment import (
    align_derivative,
    unify_indeps,
)
from thoad.differentiation.engine.broadcasting.figuration import (
    calculate_shapes,
    denull_derivative,
)
from thoad.differentiation.engine.control.symmetry import (
    depermute_metadata,
    find_variable_permutation,
    reverse_permutation,
)
from thoad.graph.structures import Node, MultiEdge
from thoad.typing import (
    AutogradFunction,
    EDData,
    Hook,
    IDData,
    Indep,
    Notation,
    PopulatedEDData,
    Shape,
    VPerm,
)
import thoad.config as config


def _populated(obj: Union[None, Any]) -> Any:
    assert obj is not None
    return obj


class DerivativeGrid:
    def __init__(self) -> None:
        self._external_derivatives: dict[Tuple[Node, ...], EDData] = {}

    @property
    def keys(self) -> set[Tuple[Node, ...]]:
        return set(self._external_derivatives.keys())

    def contains(self, key: Tuple[Node, ...]) -> bool:
        return key in self._external_derivatives

    @property
    def variables(self) -> set[Node]:
        variables: set[Node] = set()
        for key in self._external_derivatives.keys():
            variables = variables.union(key)
        return variables

    def get(self, key: Tuple[Node, ...]) -> EDData:
        substitute_data: EDData = (None, None, None, None)
        return self._external_derivatives.get(key, substitute_data)

    def __getitem__(self, key: Tuple[Node, ...]) -> EDData:
        return self.get(key)

    def set(
        self,
        key: Tuple[Node, ...],
        data: EDData,
    ) -> None:

        ###  Key checks
        assert isinstance(key, Tuple)
        assert all(isinstance(N, Node) for N in key)
        # unique-valued key
        unique_key: Tuple[Node, ...] = tuple(dict.fromkeys(key))

        null: bool = data[0] is None

        ### Data checks
        assert isinstance(data, Tuple)
        assert len(data) == 4
        assert not null or all(d is None for d in data), data
        if not null:
            # data[0] (Tensor)
            assert isinstance(data[0], Tensor)
            # data[1] (Shapes)
            assert isinstance(data[1], Tuple)
            assert len(unique_key) == len(data[1])
            assert all(isinstance(T, Tuple) for T in data[1])
            assert all(isinstance(i, int) for T in data[1] for i in T)
            # data[2] (Indeps)
            assert isinstance(data[2], Tuple)
            assert len(unique_key) == len(data[2])
            assert all(isinstance(T, Tuple) for T in data[2])
            assert all(isinstance(i, (type(None), int)) for T in data[2] for i in T)
            for i, indep in enumerate(data[2]):
                for dim in indep:
                    assert dim is None or dim in range(len(data[1][i]))
            assert len(set(len(i) for i in data[2])) == 1
            # data[3] (VPerm)
            assert isinstance(data[3], Tuple)
            assert all(isinstance(v, int) for v in data[3])
            assert set(data[3]) == set(range(len(key)))

        # Save data
        self._external_derivatives[key] = data
        return None

    def __setitem__(self, key: Tuple[Node, ...], data: EDData) -> None:
        self.set(key, data)

    def remove(self, variables: Iterable[Node]) -> None:
        keys_to_remove = [
            key
            for key in self._external_derivatives.keys()
            if len(set(variables).intersection(set(key))) > 0
        ]
        for key in keys_to_remove:
            self._external_derivatives.pop(key)
        return None


class IdxMapper:

    def __init__(self, objects: Iterable["Any"]) -> None:
        self._obj2idx: dict[int, "Any"] = {o: i for i, o in enumerate(set(objects))}
        self._idx2obj: dict["Any", int] = {i: o for o, i in self._obj2idx.items()}

    def obj_to_int(self, obj: "Any") -> int:
        assert obj in self._obj2idx
        return self._obj2idx[obj]

    def int_to_obj(self, idx: int) -> "Any":
        assert idx in self._idx2obj
        return self._idx2obj[idx]

    def array_to_int(self, objects: Iterable["Any"]) -> Tuple[int]:
        assert all(obj in self._obj2idx for obj in objects)
        return tuple([self._obj2idx[obj] for obj in objects])

    def array_to_obj(self, indices: Iterable[int]) -> Tuple["Any"]:
        assert all(idx in self._idx2obj for idx in indices)
        return tuple([self._idx2obj[idx] for idx in indices])


class VariableOperator:

    def __init__(
        self,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
        grid: DerivativeGrid,
    ) -> None:
        # instrumental attributes for extraction
        self._grid: DerivativeGrid = grid
        self._fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ]
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
        return list(
            {ev for evs, _ in self._fns.values() for ev in evs if ev is not None}
        )

    def _extract_internal_variables(self) -> list[Node]:
        return list(
            {iv for _, ivs in self._fns.values() for iv in ivs if iv is not None}
        )

    def _extract_off_variables(self, external_variables: list[Node]) -> list[Node]:
        off_external_variables: list[Node] = list()
        for ev in self._grid.variables:
            if ev not in external_variables:
                off_external_variables.append(ev)
        return off_external_variables

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


def _initialize_derivative(
    order: int,
    tensor: Tensor,
    gradient: Union[None, Tensor],
    batch: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> EDData:

    numel: int = tensor.numel()
    ndim: int = max(1, tensor.ndim)
    shape: Tuple[int, ...] = tuple(tensor.shape)
    if numel in (0, 1):
        if tensor.ndim == 0:
            shape = (1,)
        else:
            shape = tuple(max(1, sz) for sz in shape)
    grad_shape: Tuple[int, ...] = (numel, *shape)

    # create derivative data (Tensor, shapes, indeps & vperm)
    derivative: Union[None, Tensor] = None
    shapes: Union[None, Tuple[Shape, ...]] = None
    indeps: Union[None, Tuple[Indep, ...]] = None
    vperm: Union[None, VPerm] = None
    assert order >= 1
    if order == 1:
        derivative = torch.eye(numel, dtype=dtype, device=device)
        derivative = derivative.reshape(shape=grad_shape)
        shapes = (tuple(shape),)
        indeps = (tuple(range(ndim)),) if batch else (tuple(),)
        vperm = (0,)
        # apply gradient
        if gradient is not None:
            assert tensor.shape == gradient.shape
            flat_gradient: Tensor = gradient.flatten()
            flat_gradient = flat_gradient.unsqueeze(dim=0)
            derivative = flat_gradient @ derivative

    return (derivative, shapes, indeps, vperm)


def _check_hook_data(
    ED_data: EDData,
    mod_ED_data: EDData,
) -> None:
    if not isinstance(mod_ED_data, Sequence):
        raise TypeError(
            "Hook must return a sequence, "
            f"but returned {type(mod_ED_data).__name__}."
        )
    if len(mod_ED_data) != 4:
        raise ValueError(
            "Hook return-Tuple must be " f"length 4, but got length {len(mod_ED_data)}."
        )
    if not isinstance(mod_ED_data[0], Tensor):
        raise TypeError(
            "First element of hook return-Tuple must be a Tensor, "
            f"but got {type(mod_ED_data[0]).__name__}."
        )
    assert ED_data[0] is not None
    if ED_data[0].shape != mod_ED_data[0].shape:
        raise ValueError(
            "Shape mismatch hook return-Tuple first element: original "
            f"Tensor shape {ED_data[0].shape}, modified Tensor shape "
            f"{mod_ED_data[0].shape}."
        )
    if not isinstance(mod_ED_data[1], Tuple):
        raise TypeError(
            "Second element of hook return-Tuple must be a Tuple[Shape, ...] "
            f"(Shape=Tuple[int, ...]), but got {type(mod_ED_data[1]).__name__}."
        )
    if not isinstance(mod_ED_data[1], Tuple):
        raise TypeError(
            "Second element of hook return-Tuple must be a Tuple[Shape, ...], "
            f"but got{type(mod_ED_data[1]).__name__}"
        )

    for sequence in mod_ED_data[1]:
        for sz in sequence:
            if not isinstance(sequence, Sequence):
                raise TypeError(
                    "All indeps in the third element hook return-Tuple must "
                    "be type Indep (Tuple[int, ...]), "
                    f"but got {type(sequence).__name__}."
                )
            if not isinstance(sz, int):
                raise TypeError(
                    "All indeps in the third element hook return-Tuple must "
                    "be type Indep (Tuple[int, ...]), but got "
                    f"{type(sequence).__name__}"
                    f"[Union[{type(sz).__name__}, ...], ...], ...]"
                )
    if ED_data[1] != mod_ED_data[1]:
        raise ValueError(
            "Second element of hook return-Tuple must match original "
            f"shapes. Original shapes: {ED_data[1]}, "
            f"modified shapes: {mod_ED_data[1]}."
        )
    if not isinstance(mod_ED_data[2], Tuple):
        raise TypeError(
            "Third element of hook return-Tuple must be a Tuple[Indep, ...], "
            f"but got{type(mod_ED_data[2]).__name__}"
        )
    for sequence in mod_ED_data[2]:
        for dim in sequence:
            if not isinstance(sequence, Sequence):
                raise TypeError(
                    "All indeps in the third element hook return-Tuple must "
                    "be type Indep (Tuple[Union[None, int], ...]), "
                    f"but got {type(sequence).__name__}."
                )
            if not isinstance(dim, (type(None), int)):
                raise TypeError(
                    "All indeps in the third element hook return-Tuple must "
                    "be type Indep (Tuple[Union[None, int], ...]), but got "
                    f"{type(sequence).__name__}"
                    f"[Union[{type(dim).__name__}, ...], ...], ...]"
                )
    sufix: dict[int, str] = {1: "st", 2: "nd", 3: "rd"}
    for i, indep in enumerate(mod_ED_data[2]):
        for dim in indep:
            assert isinstance(ED_data[1], Tuple)
            if dim is not None and dim not in range(len(ED_data[1][i])):
                raise IndexError(
                    f"Index {dim} in hook {i}-{sufix.get(i, "th")} "
                    "return-Indep is out of range."
                )
            if dim is not None and mod_ED_data[2][i].count(dim) != 1:
                raise ValueError(
                    f"Index {dim} in hook {i}-{sufix.get(i, "th")} "
                    "return-Indep is duplicated."
                )
    if ED_data[2] != mod_ED_data[2]:
        raise ValueError(
            "Third element of hook return-Tuple must match original "
            f"shapes. Original shapes: {ED_data[2]}, "
            f"modified shapes: {mod_ED_data[2]}."
        )
    if not isinstance(mod_ED_data[3], Tuple):
        raise TypeError(
            "Forth element of hook return Tuple must be a Tuple[int, ...], "
            f"but got{type(mod_ED_data[3]).__name__}"
        )
    for i in mod_ED_data[3]:
        if not isinstance(i, int):
            raise TypeError(
                "Forth element hook return-Tuple must "
                "be type Tuple[int, ...], but got "
                f"Tuple[Union[{type(i).__name__}, ...], ...], ...]"
            )
    if ED_data[3] != mod_ED_data[3]:
        raise ValueError(
            "Forth element of hook return-Tuple must match original "
            f"vperm. Original vperm: {ED_data[3]}, "
            f"modified vperm: {mod_ED_data[3]}."
        )

    return None


def _regularize_derivative(
    key: Tuple[Node, ...],
    data: EDData,
    keep_batch: bool,
    keep_vperm: bool,
) -> PopulatedEDData:
    """
    distribute all independent dimensions in passed derivative
    """
    derivative: Union[None, Tensor]
    shapes: Union[None, Tuple[Shape, ...]]
    indeps: Union[None, Tuple[Indep, ...]]
    vperm: Union[None, VPerm]
    (derivative, shapes, indeps, vperm) = _extract_EDData(data)
    new_indeps: Tuple[Indep, ...] = tuple(len(indep) * (None,) for indep in indeps)
    depermuted_key: Tuple[Node, ...]
    depermuted_key = tuple(key[d] for d in reverse_permutation(vperm))
    var_map: dict[Node, int]
    var_map = {v: i for i, v in enumerate(dict.fromkeys(depermuted_key))}
    depermuted_variables: Tuple[int, ...]
    depermuted_variables = tuple(var_map[v] for v in depermuted_key)
    new_vperm: VPerm = tuple(range(len(key))) if keep_vperm else vperm
    new_derivative: Tensor = align_derivative(
        derivative=derivative,
        variables=depermuted_variables,
        variable_perm=new_vperm,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=shapes,
        expected_indeps=new_indeps,
        keepdim=keep_batch,
    )
    new_vperm: VPerm = tuple(range(len(key)))
    new_data: EDData = (new_derivative, shapes, new_indeps, new_vperm)
    return new_data


def _extract_EDData(data: EDData) -> PopulatedEDData:
    derivative: Union[None, Tensor]
    shapes: Union[None, Tuple[Shape, ...]]
    indeps: Union[None, Tuple[Indep, ...]]
    vperm: Union[None, VPerm]
    (derivative, shapes, indeps, vperm) = data
    assert derivative is not None
    assert shapes is not None
    assert indeps is not None
    assert vperm is not None
    return (derivative, shapes, indeps, vperm)


def _reconstruct_null_derivative(
    registry: dict[Tuple[Node, ...], EDData],
    key: Tuple[Node, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> PopulatedEDData:
    assert all((v,) in registry for v in key)
    var_num_map: dict[Node, int]
    var_num_map = {v: i for i, v in enumerate(dict.fromkeys(key))}
    num_variables: Tuple[int, ...] = tuple(var_num_map[v] for v in key)
    shapes: list[Shape] = list()
    indeps: list[Indep] = list()
    primal_size: Union[None, int] = None
    for v in var_num_map.keys():
        reference_data: PopulatedEDData = _extract_EDData(registry[(v,)])
        assert primal_size is None or primal_size == reference_data[0].shape[0]
        primal_size = reference_data[0].shape[0]
        # primal_size = reference_data[0].shape[0]
        shapes.append(reference_data[1][0])
        if len(set(key)) == 1:
            indeps.append(tuple(range(len(reference_data[1][0]))))
        else:
            indeps.append(reference_data[2][0])
    assert primal_size is not None
    for indepT in zip(*indeps):
        indepT_sizes: set[int] = set()
        for i, d in enumerate(indepT):
            if d is not None:
                indepT_sizes.add(shapes[i][d])
        assert len(indepT_sizes) == 1
    # calculate differnetial expected view
    view: Tuple[int, ...]
    view, _ = calculate_shapes(
        first_size=primal_size,
        variables=num_variables,
        shapes=tuple(shapes),
        indeps=tuple(indeps),
        indeps_squeezed=False,
    )
    derivative: Tensor = torch.zeros(size=view, dtype=dtype, device=device)
    vperm: VPerm = tuple(range(len(key)))

    return (derivative, tuple(shapes), tuple(indeps), vperm)


class GradOperator:

    def __init__(self) -> None:
        # control variable
        self._initialized: bool = False
        # AD initialization attributes
        self._order: int
        self._GO_tensor: Tensor
        # AD configuration static attributes
        self._batch_optimization: bool = bool(
            getattr(config, "BATCH_OPTIMIZATION", False)
        )
        self._schwarz_optimization: bool = bool(
            getattr(config, "SCHWARZ_OPTIMIZATION", False)
        )
        # AD configuration dynamic attributes
        self._cross_terminals: bool = False
        self._keep_batch: bool = False
        self._keep_vperm: bool = False
        # AD mods
        self._terminals: dict[Node, Tensor] = dict()
        self._retentions: set[Tuple[Node, ...]] = set()
        self._hooks: dict[
            Tuple[Node, ...], Tuple[Hook, dict[MultiEdge, set[Tensor]]]
        ] = dict()
        self._xfn_hooks: dict[
            # node combination where to apply the hook
            Tuple[Node, ...],
            # 1. hook function
            # 2. child nodes per function leading to some node in the combination
            # 3. boolean indicating whether the hook has already been applied
            Tuple[Hook, dict[ExtendedAutogradFunction, set[Tensor]], list[bool]],
        ]
        # technical requirements
        self._dtype: torch.dtype
        self._device: torch.device
        # instrumental attributes
        self._grid: DerivativeGrid
        self._past_fns: set[ExtendedAutogradFunction]
        self._past_schwarz: dict[Node, bool]
        self._target_gradients: dict[Tuple[Node, ...], EDData] = dict()

        return None

    @property
    def cross_terminals(self) -> bool:
        return self._cross_terminals

    @cross_terminals.setter
    def cross_terminals(self, value: bool) -> None:
        self._cross_terminals = value

    @property
    def keep_batch(self) -> bool:
        return self._keep_batch

    @keep_batch.setter
    def keep_batch(self, value: bool) -> None:
        self._keep_batch = value

    @property
    def keep_vperm(self) -> bool:
        return self._keep_vperm

    @keep_vperm.setter
    def keep_vperm(self, value: bool) -> None:
        self._keep_vperm = value

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
            self._retentions.remove(key)
        return None

    def add_backward_hook(
        self,
        key: Tuple[Node, ...],
        connecting_functions: dict[MultiEdge, set[Tensor]],
        hook: Hook,
    ) -> None:
        self._hooks[key] = (hook, connecting_functions)
        return None

    def drop_backward_hook(
        self,
        key: Tuple[Node, ...],
    ) -> None:
        if key in self._hooks:
            self._hooks.pop(key)
        return None

    def initialize_gradients(
        self, order: int, node: Node, tensor: Tensor, gradient: Union[None, Tensor]
    ) -> None:

        # control
        self._initialized = True
        self._batch_optimization &= order > 1
        self._past_fns = set()
        self._past_schwarz = {node: self._schwarz_optimization}

        ### Save class attributes relevant for differentiation
        self._order = order
        self._GO_tensor = tensor
        self._dtype = tensor.dtype
        self._device = tensor.device

        ### Initialize derivatives
        self._grid = DerivativeGrid()
        for o in range(1, 1 + order):
            self._grid[o * (node,)] = _initialize_derivative(
                order=o,
                tensor=self._GO_tensor,
                gradient=gradient,
                batch=self._batch_optimization,
                dtype=self._dtype,
                device=self._device,
            )

        return None

    def initialize_retentions(self, order: int, groups: list[set[Node]]) -> None:
        # reset target gradients and redefine expected keys
        self.clear_gradients()
        # save keys of new required retentions
        self._target_gradients = dict()
        for o in range(1, 1 + order):
            for key in itertools.product(self._terminals.keys(), repeat=o):
                if self._cross_terminals:
                    self._target_gradients[key] = (None, None, None, None)
                else:
                    key_set: set[Node] = set(key)
                    if len(key_set) == 1:
                        self._target_gradients[key] = (None, None, None, None)
                    for G in groups:
                        if key_set.issubset(G):
                            self._target_gradients[key] = (None, None, None, None)
        for key in self._retentions:
            if len(key) <= order:
                self._target_gradients[key] = (None, None, None, None)

        return None

    def initialize_hooks(self) -> None:
        self._xfn_hooks = dict()
        for key, (hook, D) in self._hooks.items():
            self._xfn_hooks[key] = (hook, {E.xfn: T for (E, T) in D.items()}, [False])
        return None

    def clear_gradients(self) -> None:
        for tensor in self._terminals.values():
            if "hgrad" in dir(tensor):
                delattr(tensor, "hgrad")
            if "hdata" in dir(tensor):
                delattr(tensor, "hdata")
        return None

    def fetch_hgrad(
        self,
        key: Tuple[Node, ...],
        keep_batch: bool,
        keep_vperm: bool,
    ) -> PopulatedEDData:
        if key not in self._target_gradients:
            raise KeyError("No gradient saved for given key")
        ED_data: EDData = self._target_gradients[key]
        explicit_ED_data: EDData = (None, None, None, None)
        derivative: Tensor
        shapes: Tuple[Shape, ...]
        indeps: Tuple[Indep, ...]
        vperm: VPerm
        if None not in ED_data:
            (derivative, shapes, indeps, vperm) = _extract_EDData(ED_data)
            if not keep_batch:
                (derivative, shapes, indeps, vperm) = _regularize_derivative(
                    key=key,
                    data=(derivative, shapes, indeps, vperm),
                    keep_batch=keep_batch,
                    keep_vperm=keep_vperm,
                )
            ED_data = (derivative, shapes, indeps, vperm)
            explicit_shapes: Tuple[Shape, ...]
            explicit_indeps: Tuple[Indep, ...]
            explicit_shapes = tuple(shapes[tuple(set(key)).index(v)] for v in key)
            explicit_indeps = tuple(indeps[tuple(set(key)).index(v)] for v in key)
            explicit_ED_data = (derivative, explicit_shapes, explicit_indeps, vperm)
        else:
            assert all(d is None for d in ED_data)
            (derivative, shapes, indeps, vperm) = _reconstruct_null_derivative(
                registry=self._target_gradients,
                key=key,
                dtype=self._dtype,
                device=self._device,
            )
            if not keep_batch:
                (derivative, shapes, indeps, vperm) = _regularize_derivative(
                    key=key,
                    data=(derivative, shapes, indeps, vperm),
                    keep_batch=keep_batch,
                    keep_vperm=keep_vperm,
                )
            assert vperm == tuple(range(len(vperm)))
            explicit_ED_data = (derivative, shapes, indeps, vperm)
        return explicit_ED_data

    def _save_gradients(
        self,
        gradients: dict[Tuple[Node, ...], EDData],
    ) -> None:
        data: EDData
        for key, data in gradients.items():
            if key in self._target_gradients:
                self._target_gradients[key] = data
        return None

    def attach_gradients(self) -> None:
        # Note. does not attach hgrad to intermediate retentions as they are not
        #   necesarily related to one single tensor
        self._initialized = False
        for node, tensor in self._terminals.items():
            node_gradients: list[Tensor] = list()
            node_metadatas: list[Tuple[Tuple[Shape, ...], Tuple[Indep, ...], VPerm]] = (
                list()
            )
            for o in range(1, 1 + self._order):
                key: Tuple[Node, ...] = tuple(node for _ in range(o))
                ED_data: EDData = self.fetch_hgrad(
                    key=key,
                    keep_batch=self._keep_batch,
                    keep_vperm=self._keep_vperm,
                )
                derivate: Tensor
                shapes: Tuple[Shape, ...]
                indeps: Tuple[Indep, ...]
                vperm: VPerm
                (derivate, shapes, indeps, vperm) = _extract_EDData(ED_data)
                node_gradients.append(derivate)
                node_metadatas.append((shapes, indeps, vperm))
            setattr(tensor, "hgrad", tuple(node_gradients))
            setattr(tensor, "hdata", tuple(node_metadatas))
        return None

    def _denull_internals(
        self,
        external_indeps: dict[Node, Union[None, Indep]],
        expected_shapes: dict[Node, Union[None, Shape]],
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]],
        einstein_notations: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Notation]],
    ) -> Tuple[
        dict[Node, Union[None, Shape]],
        dict[Tuple[Node, Node], Union[None, Indep]],
        dict[Tuple[Node, Tuple[Node, ...]], Union[None, Notation]],
    ]:
        null_variables: set[Node] = set()
        denulled_shapes: dict[Node, Union[None, Shape]] = dict()
        denulled_indeps: dict[Tuple[Node, Node], Union[None, Indep]] = dict()
        denulled_notations: dict[
            Tuple[Node, Tuple[Node, ...]], Union[None, Notation]
        ] = dict()
        # denull shapes
        for ev, shape in expected_shapes.items():
            if shape == tuple():
                null_variables.add(ev)
                denulled_shapes[ev] = (1,)
            else:
                denulled_shapes[ev] = shape
        # denull indeps
        for (ev, iv), indep in expected_indeps.items():
            if ev in null_variables:
                denulled_indep: Union[None, Indep] = external_indeps[ev]
                assert denulled_indep is not None
                denulled_indeps[(ev, iv)] = (None,) * len(denulled_indep)
            else:
                denulled_indeps[(ev, iv)] = indep
        # denull notations
        for (ev, ivs), notation in einstein_notations.items():
            if ev in null_variables and notation is not None:
                assert len(notation[0][0]) == 0
                max_index: int = max((0, *notation[0][1]))
                denulled_notations[(ev, ivs)] = [
                    ((max_index + 1,), notation[0][1]),
                    *notation[1:],
                ]
            else:
                denulled_notations[(ev, ivs)] = notation

        return (denulled_shapes, denulled_indeps, denulled_notations)

    def _denull_derivatives(
        self,
        derivatives: dict[Tuple[Node, ...], EDData],
    ) -> dict[Tuple[Node, ...], EDData]:

        denulled_derivatives: dict[Tuple[Node, ...], EDData] = dict()
        for key, (tensor, shapes, indeps, vperm) in derivatives.items():
            ED_data: EDData = (tensor, shapes, indeps, vperm)
            null_tensor: bool = tensor is None
            empty_shape: bool = shapes is not None
            if shapes is not None:
                empty_shape &= any(0 in s or len(s) > 0 for s in shapes)
            # if tensor is not None or any(0 in s or len(s) > 0 for s in shapes):
            if not null_tensor or empty_shape:
                assert tensor is not None
                assert shapes is not None
                assert indeps is not None
                assert vperm is not None
                depermuted_key: Tuple[Node, ...]
                depermuted_key = tuple(key[d] for d in reverse_permutation(vperm))
                var_map: dict[Node, int]
                var_map = {v: i for i, v in enumerate(dict.fromkeys(depermuted_key))}
                depermuted_variables: Tuple[int, ...]
                depermuted_variables = tuple(var_map[v] for v in depermuted_key)
                denulled_tensor: Tensor
                denulled_shapes: Tuple[Shape, ...]
                denulled_indeps: Tuple[Indep, ...]
                (denulled_tensor, denulled_shapes, denulled_indeps) = denull_derivative(
                    derivative=tensor,
                    variables=depermuted_variables,
                    shapes=shapes,
                    indeps=indeps,
                    dtype=self._dtype,
                    device=self._device,
                )
                ED_data = (
                    denulled_tensor,
                    denulled_shapes,
                    denulled_indeps,
                    vperm,
                )
            denulled_derivatives[key] = ED_data
        return denulled_derivatives

    def _apply_hooks(
        self,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
    ) -> None:
        # update viewed ExtendedAtutogradFunctions
        self._past_fns.update(fns.keys())

        ### Apply hooks in specified variable (node) combinations
        for key, hook_data in self._xfn_hooks.items():
            apply_hook: bool = self._grid.contains(key)
            apply_hook &= not hook_data[2][0]
            if apply_hook and set(hook_data[1].keys()).issubset(self._past_fns):
                hook_data[2][0] = True
                ED_data: EDData = self._grid[key]
                # reconstruct ED_data (with 0s) if derivative is null
                if ED_data == (None, None, None, None):
                    registry: dict[Tuple[Node, ...], EDData] = dict()
                    registry = {key: self._grid[key] for key in self._grid.keys}
                    ED_data: EDData = _reconstruct_null_derivative(
                        registry=registry,
                        key=key,
                        dtype=self._dtype,
                        device=self._device,
                    )
                # apply hook
                hook: Hook = self._xfn_hooks[key][0]
                context: dict[AutogradFunction, set[Tensor]] = dict()
                for fn, (fn_evs, fn_ivs) in fns.items():
                    if len(set(fn_ivs).intersection(set(key))) > 0:
                        context[fn.grad_fn] = self._xfn_hooks[key][1][fn]
                assert len(context) > 0
                mod_ED_data: EDData = hook(ED_data, context)
                # checks
                if bool(getattr(config, "DEGUB", False)):
                    _check_hook_data(ED_data=ED_data, mod_ED_data=mod_ED_data)
                # save modified derivative data
                self._grid[key] = mod_ED_data

        return None

    def _plan_differentiations(
        self,
        V: VariableOperator,
        groups: list[set[Node]],
    ) -> dict[Tuple[Node, ...], bool]:
        ### Determine which composed derivatives need to be computed
        #   (~ which are necesary for considered crossings?)
        #   (in a general sense; ie. including self crossings)
        internal_keys: dict[Tuple[Node, ...], bool] = dict()
        for o in range(1, self._order + 1):
            for key in itertools.product(V.all_ivs, repeat=o):
                key_set: set[Node] = set(key)
                require_internal: bool = self._cross_terminals
                if not self._cross_terminals:
                    # compute crossing if all variables are found together in one group
                    require_internal |= any(key_set.issubset(G) for G in groups)
                if any(N in V.ivs for N in key):
                    internal_keys[key] = require_internal
                else:
                    internal_keys[key] = False
        return internal_keys

    def _determine_flexibilities(
        self,
        V: VariableOperator,
        groups: list[set[Node]],
    ) -> dict[Node, bool]:
        internal_flexibilities: dict[Node, bool] = dict()
        for iv in V.all_ivs:
            if iv in V.ivs:
                internal_flexibilities[iv] = (
                    all(len(G.intersection(V.all_ivs) - {iv}) == 0 for G in groups)
                    and self._batch_optimization
                )
            else:
                internal_flexibilities[iv] = False
        return internal_flexibilities

    def _determine_schwarz_condition(
        self,
        V: VariableOperator,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
    ) -> dict[Node, bool]:
        internal_schwarz: dict[Node, bool] = {iv: True for iv in V.all_ivs}
        for fn, (evs, ivs) in fns.items():
            external_schwarz: bool = all(self._past_schwarz[ev] for ev in evs)
            for iv in (iv for iv in ivs if iv is not None):
                internal_schwarz[iv] &= external_schwarz
                internal_schwarz[iv] &= fn.schwarz
                internal_schwarz[iv] &= self._schwarz_optimization
        for iv in V.all_ivs:
            internal_schwarz[iv] &= self._past_schwarz.get(iv, True)
        self._past_schwarz.update(internal_schwarz)
        return internal_schwarz

    def _acquire_external_derivatives(self, V: VariableOperator) -> Tuple[
        dict[Tuple[Node, ...], Union[None, Tensor]],
        dict[Tuple[Node, ...], Union[None, VPerm]],
    ]:
        """
        Acquire external derivatives for the given VariableOperator V.

        Iterates orders from 1 to self._order and builds a mapping of
        derivative keys.

        - If at least one involved external variable is in the key,
          queries from the grid all derivatives with respect to those
          external variables.

        - Otherwise, fakes derivatives with respect to non-involved
          external variables as null tensors. Internal derivatives from
          these variables to involved internal variables will be all null,
          so these fake entries have no effect on the final computed
          composed derivatives, since in this contractive step only
          derivatives with respect to involved internal variables are
          used.
        """
        external_derivatives: dict[Tuple[Node, ...], Union[None, Tensor]] = dict()
        external_vperms: dict[Tuple[Node, ...], Union[None, VPerm]] = dict()
        for o in range(1, self._order + 1):
            for key in itertools.product(V.all_evs, repeat=o):
                if not all(N not in V.evs for N in key):
                    data = self._grid[key]
                    external_derivatives[key] = data[0]
                    external_vperms[key] = data[3]

        return (external_derivatives, external_vperms)

    def _acquire_derivative_metadata(
        self,
        V: VariableOperator,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
        required_differentiations: dict[Tuple[Node, ...], bool],
    ) -> Tuple[
        dict[Node, Union[None, Shape]],
        dict[Node, Union[None, Indep]],
        dict[Node, Union[None, Shape]],
        dict[Tuple[Node, Node], Union[None, Indep]],
    ]:
        """
        *TODO: Fix the way external indeps are collected
            Right now external indeps are asociated to an external variable.
            This simplifies the process of their retrieval, given that the alternative
            [retrieving them by patiality] would require to link them to the patiality
            position of an external variable [(Node) -> ((Node, ...), int)].
            However, this forces all partialities associated to the same variable to
            have the same Indep [even across different external derivatives]. This
            premise is susceptible of being violated in graph joins, due to usage of
            hooks.
            How to fix?
            -> Pass to fn.check_shape the non inclusive unification of shapes and indeps
        """

        ### Collect externally assumed shapes
        external_tensors: dict[Node, Union[None, Tensor]] = dict()
        external_shapes: dict[Node, Union[None, Shape]] = dict()
        external_indeps: dict[Node, Union[None, Indep]] = dict()
        for ev in V.all_evs:
            assert ev not in external_shapes
            assert ev not in external_indeps
            data = self._grid[(ev,)]
            assert None not in data
            # save tensor
            tensor: Union[None, Tensor] = data[0]
            assert tensor is not None
            external_tensors[ev] = tensor
            # save shape
            shapes: Union[None, Tuple[Shape, ...]] = data[1]
            assert shapes is not None
            external_shapes[ev] = shapes[0]
            # save indep
            indeps: Union[None, Tuple[Indep, ...]] = data[2]
            assert indeps is not None
            external_indeps[ev] = indeps[0]  # *TODO

        ### Resolve shapes & independencies
        # determine which variables are crossed with others
        crossed_variables: dict[Node, bool] = {iv: False for iv in V.all_ivs}
        for ivs, required in required_differentiations.items():
            if len(set(ivs)) > 1:
                for iv in ivs:
                    crossed_variables[iv] = crossed_variables[iv] or required
        # collect indep and shape projections
        expected_shapes: dict[Node, Union[None, Shape]] = dict()
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]] = dict()
        for fn, (fn_evs, fn_ivs) in fns.items():
            for i, ev in enumerate(fn_evs):
                assert ev not in expected_shapes
                tensor_null: bool = external_tensors[ev] is None
                if not tensor_null:
                    shape: Union[None, Shape] = external_shapes[ev]
                    indep: Union[None, Indep] = external_indeps[ev]
                    assert shape is not None
                    assert indep is not None
                    expected_shape: Union[None, Shape] = None
                    for j, iv in enumerate(fn_ivs):
                        if iv is not None:
                            projection: Tuple[Shape, Indep]
                            projection = fn.check_shape(
                                out_id=i,
                                inp_id=j,
                                shape=shape,
                                indep=indep,
                                crossed=crossed_variables[iv],
                            )
                            expected_indeps[(ev, iv)] = projection[1]
                            expected_shape = projection[0]
                    assert expected_shape is not None
                    expected_shapes[ev] = expected_shape
                else:
                    expected_shapes[ev] = None
                    for j, iv in enumerate(fn_ivs):
                        if iv is not None:
                            expected_indeps[(ev, iv)] = None
        for ov in V.ovs:
            # if ov not in grid: data <- (None, None, None, None)
            data = self._grid[(ov,)]
            # save shape
            shapes: Union[None, Tuple[Shape, ...]] = data[1]
            assert shapes is not None
            expected_shapes[ov] = shapes[0]
            # save indep
            indeps: Union[None, Tuple[Indep, ...]] = data[2]
            assert indeps is not None
            expected_indeps[(ov, ov)] = indeps[0]

        return (external_shapes, external_indeps, expected_shapes, expected_indeps)

    def _compute_internal_derivatives(
        self,
        V: VariableOperator,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
        required_differentiations: dict[Tuple[Node, ...], bool],
    ) -> Tuple[
        dict[Tuple[Node, Tuple[Node, ...]], Union[None, Tensor]],
        dict[Tuple[Node, Tuple[Node, ...]], Union[None, Notation]],
    ]:
        """
        Compute all internal derivatives up to the specified order.

        Steps:
        1. Determine shapes and independencies for each external variable.
           - Shape projections must converge to one unique shape.
           - Independency projections can be multiple; final independency is the logical
             AND across projections.

        2. Initialize all internal derivatives as null tensors.

        3. Substitute non-null tensors for internal-external pairs where a connecting
           function exists.

        4. Handle non-involved external variables:
           - For first-order and matching internal variable, use the identity tensor.
           - Otherwise, leave as null tensor.
        """

        ### Initialize internal derivatives
        internal_derivs: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Tensor]]
        internal_derivs = dict()
        eins_notations: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Notation]]
        eins_notations = dict()
        for ev in V.evs:
            for o in range(1, self._order + 1):
                for ivs in itertools.product(V.ivs, repeat=o):
                    internal_derivs[(ev, ivs)] = None
                    eins_notations[(ev, ivs)] = None

        ### Populate with actual derivatives from fns
        for fn, (fn_evs, fn_ivs) in fns.items():
            for fn_ev in fn_evs:
                for o in range(1, self._order + 1):
                    for ivs in itertools.product(fn_ivs, repeat=o):
                        if None not in ivs:
                            _ivs: Tuple[Node] = tuple(_populated(iv) for iv in ivs)
                            if required_differentiations[_ivs]:
                                assert isinstance(fn, ContractiveFunction)
                                int_ev: int = fn_evs.index(fn_ev)
                                int_ivs: Tuple[int, ...]
                                int_ivs = tuple(fn_ivs.index(v) for v in _ivs)
                                ID_data: IDData = fn[(int_ev, int_ivs)]
                                internal_derivs[(fn_ev, _ivs)] = ID_data[0]
                                eins_notations[(fn_ev, _ivs)] = ID_data[1]

        ### Handle non-involved external variables
        for oev in V.ovs:
            for o in range(1, self._order + 1):
                key: Tuple[Node, Tuple[Node, ...]] = (oev, o * (oev,))
                internal_derivs[key] = None
                eins_notations[key] = None
                if o == 1:
                    data: EDData = self._grid[(oev,)]
                    external_deriv: Union[None, Tensor] = data[0]
                    if external_deriv is not None:
                        external_shapes: Union[None, Tuple[Shape, ...]] = data[1]
                        assert external_shapes is not None
                        external_shape: Shape = external_shapes[0]
                        int_diff: Tensor = torch.ones(
                            size=external_shape, dtype=self._dtype, device=self._device
                        )
                        internal_derivs[key] = int_diff
                        external_range: Tuple[int, ...]
                        external_range = tuple(range(len(external_shape)))
                        inputs_indices: Tuple[Tuple[int, ...], Tuple[int, ...]]
                        inputs_indices = (external_range, external_range)
                        output_indices: Tuple[Tuple[int, ...]] = (external_range,)
                        einstein_notation = list()
                        einstein_notation.append(inputs_indices)
                        einstein_notation.append(output_indices)
                        einstein_notation.append(
                            (tuple(external_shape), tuple(True for _ in external_shape))
                        )
                        eins_notations[key] = einstein_notation

        return (internal_derivs, eins_notations)

    def _compose_derivatives(
        self,
        V: VariableOperator,
        required_differentiations: dict[Tuple[Node, ...], bool],
        mappers: Tuple[IdxMapper, IdxMapper],
        external_derivatives: dict[Tuple[Node, ...], Union[None, Tensor]],
        external_shapes: dict[Node, Union[None, Shape]],
        external_indeps: dict[Node, Union[None, Indep]],
        external_vperms: dict[Tuple[Node, ...], Union[None, VPerm]],
        expected_shapes: dict[Node, Union[None, Shape]],
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]],
        internal_derivatives: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Tensor]],
        internal_flexibilities: dict[Node, bool],
        internal_schwarz: dict[Node, bool],
        einstein_notations: dict[Tuple[Node, Tuple[Node, ...]], Optional[Notation]],
    ) -> dict[Tuple[Node, ...], EDData]:

        # modify keys swapping Nodes by integers
        ED: dict[Tuple[int, ...], Union[None, Tensor]] = dict()
        ID: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]] = dict()
        NT: dict[Tuple[int, Tuple[int, ...]], Union[None, Notation]] = dict()
        ES: dict[int, Union[None, Shape]] = dict()
        EI: dict[int, Union[None, Indep]] = dict()
        EP: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
        XS: dict[int, Union[None, Shape]] = dict()
        XI: dict[Tuple[int, int], Union[None, Indep]] = dict()
        IF: dict[int, bool] = dict()
        IS: dict[int, bool] = dict()

        for evs, diff in external_derivatives.items():
            num_evs: Tuple[int, ...] = mappers[0].array_to_int(objects=evs)
            ED[num_evs] = diff
            EP[num_evs] = external_vperms[evs]
        for (ev, ivs), diff in internal_derivatives.items():
            num_ev = mappers[0].obj_to_int(obj=ev)
            num_ivs: Tuple[int] = mappers[1].array_to_int(objects=ivs)
            num_full: Tuple[int, Tuple[int, ...]] = (num_ev, num_ivs)
            ID[num_full] = diff
            NT[num_full] = einstein_notations[(ev, ivs)]
        for ev in V.all_evs:
            num_ev: int = mappers[0].obj_to_int(obj=ev)
            ES[num_ev] = external_shapes[ev]
            EI[num_ev] = external_indeps[ev]
            XS[num_ev] = expected_shapes[ev]
            for iv in V.all_ivs:
                num_iv: int = mappers[1].obj_to_int(obj=iv)
                num_pair: Tuple[int, int] = (num_ev, num_iv)
                XI[num_pair] = expected_indeps.get((ev, iv), None)
        for iv in V.all_ivs:
            num_iv: int = mappers[1].obj_to_int(obj=iv)
            IF[num_iv] = internal_flexibilities[iv]
            IS[num_iv] = internal_schwarz[iv]

        # initialize composition loader
        E_size: int = len(V.all_evs)
        I_size: int = len(V.all_ivs)
        loader: Loader = Loader(
            external_size=E_size,
            internal_size=I_size,
            max_order=self._order,
            external_derivatives=ED,
            external_shapes=ES,
            external_indeps=EI,
            external_vperms=EP,
            expected_shapes=XS,
            expected_indeps=XI,
            internal_derivatives=ID,
            internal_flexibilities=IF,
            einstein_notations=NT,
            dtype=self._dtype,
            device=self._device,
        )

        ### Compute required composed derivatives
        composed_derivatives: dict[Tuple[Node, ...], EDData] = dict()
        # determine which composed derivatives will be faked with schwarz
        schwarz_bases: list[Tuple[Tuple[Node, ...], bool]] = list()
        schwarz_fakes: list[Tuple[Tuple[Node, ...], bool]] = list()
        schwarz_map: dict[Tuple[int, ...], Tuple[int, ...]] = dict()
        for cmp_key, compute in required_differentiations.items():
            num_cmp: Tuple[int, ...] = mappers[1].array_to_int(objects=cmp_key)
            is_base: bool = all(IS[num_iv] for num_iv in num_cmp)
            is_base &= compute
            is_base &= tuple(sorted(num_cmp)) not in schwarz_map
            if is_base or not self._schwarz_optimization:
                schwarz_bases.append((cmp_key, compute))
                schwarz_map[tuple(sorted(num_cmp))] = num_cmp
            else:
                schwarz_fakes.append((cmp_key, compute))
        # compute bases
        for cmp_key, compute in schwarz_bases:
            num_cmp: Tuple[int, ...] = mappers[1].array_to_int(objects=cmp_key)
            composed_tensor: Union[None, Tensor] = None
            composed_shapes: Union[None, Tuple[Shape, ...]] = None
            composed_indeps: Union[None, Tuple[Indep, ...]] = None
            composed_vperm: Union[None, VPerm] = None
            if compute:
                dict_shapes: Union[None, dict[int, Shape]]
                dict_indeps: Union[None, dict[int, Indep]]
                composed_tensor, dict_shapes, dict_indeps = loader.compose(
                    variables=num_cmp,
                )
                if composed_tensor is not None:
                    num_unique_key: Tuple[int, ...] = tuple(dict.fromkeys(num_cmp))
                    assert dict_shapes is not None
                    assert dict_indeps is not None
                    composed_shapes = tuple(dict_shapes[v] for v in num_unique_key)
                    composed_indeps = tuple(dict_indeps[v] for v in num_unique_key)
                    composed_vperm = tuple(range(len(num_cmp)))
            composed_derivatives[cmp_key] = (
                composed_tensor,
                composed_shapes,
                composed_indeps,
                composed_vperm,
            )
        # aggregate fakes
        for fake_cmp_key, compute in schwarz_fakes:
            composed_data: EDData = (None, None, None, None)
            if compute:
                fake_num_cmp_key: Tuple[int, ...]
                fake_num_cmp_key = mappers[1].array_to_int(objects=fake_cmp_key)
                base_num_cmp_key: Tuple[int, ...]
                base_num_cmp_key = schwarz_map[tuple(sorted(fake_num_cmp_key))]
                base_cmp_key: Tuple[Node, ...]
                base_cmp_key = mappers[1].array_to_obj(indices=base_num_cmp_key)
                base_data: EDData = composed_derivatives[base_cmp_key]
                if None not in base_data:
                    composed_data = (
                        *base_data[0:3],
                        find_variable_permutation(
                            init=base_num_cmp_key, target=fake_num_cmp_key
                        ),
                    )
            composed_derivatives[fake_cmp_key] = composed_data

        return composed_derivatives

    def _transform_derivatives(
        self,
        fn: ExtendedAutogradFunction,
        sources: Tuple[Node, ...],
        targets: Tuple[Union[None, Node], ...],
        V: VariableOperator,
        required_differentiations: dict[Tuple[Node, ...], bool],
        mappers: Tuple[IdxMapper, IdxMapper],
        external_derivatives: dict[Tuple[Node, ...], Union[None, Tensor]],
        external_shapes: dict[Node, Union[None, Shape]],
        external_indeps: dict[Node, Union[None, Indep]],
        external_vperms: dict[Tuple[Node, ...], Union[None, VPerm]],
        expected_shapes: dict[Node, Union[None, Shape]],
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]],
        internal_schwarz: dict[Node, bool],
    ) -> dict[Tuple[Node, ...], EDData]:

        # modify keys swapping Nodes by integers
        SS: Tuple[int, ...] = mappers[0].array_to_int(objects=sources)  # sources
        TT: Tuple[Union[None, int], ...] = tuple(
            None if iv is None else mappers[1].obj_to_int(obj=iv) for iv in targets
        )  # targets
        ED: dict[Tuple[int, ...], Union[None, Tensor]] = dict()
        ES: dict[int, Union[None, Shape]] = dict()  # external shapes
        EI: dict[int, Union[None, Indep]] = dict()  # external indeps
        EP: dict[Tuple[int, ...], Union[None, VPerm]] = dict()  # external vperms
        XS: dict[int, Union[None, Shape]] = dict()  # expected shapes
        XI: dict[Tuple[int, int], Union[None, Indep]] = dict()  # expected_indeps
        IS: dict[int, bool] = dict()  # internal schwarz

        for evs, diff in external_derivatives.items():
            num_evs: Tuple[int, ...] = mappers[0].array_to_int(objects=evs)
            ED[num_evs] = diff
            EP[num_evs] = external_vperms[evs]
        for ev in V.all_evs:
            num_ev: int = mappers[0].obj_to_int(obj=ev)
            ES[num_ev] = external_shapes[ev]
            EI[num_ev] = external_indeps[ev]
            XS[num_ev] = expected_shapes[ev]
            for iv in V.all_ivs:
                num_iv: int = mappers[1].obj_to_int(obj=iv)
                num_pair: Tuple[int, int] = (num_ev, num_iv)
                XI[num_pair] = expected_indeps.get((ev, iv), None)
        for iv in V.all_ivs:
            num_iv: int = mappers[1].obj_to_int(obj=iv)
            IS[num_iv] = internal_schwarz[iv]

        def _spot_active_variables(
            sequence: Sequence[Any],
            variables: Sequence[Any],
        ) -> Tuple[bool, ...]:
            return tuple(item in variables for item in sequence)

        def _index(
            sequence: Sequence[Union[None, int]], value: int
        ) -> Union[None, int]:
            index: Union[None, int] = None
            if value in sequence:
                index = sequence.index(value)
            return index

        ### Compute Required derivatives
        composed_derivatives: dict[Tuple[Node, ...], EDData] = dict()
        for cmp_key, compute in required_differentiations.items():
            num_cmp: Tuple[int, ...] = mappers[1].array_to_int(objects=cmp_key)
            cmp_spots: Tuple[bool, ...] = _spot_active_variables(num_cmp, TT)
            if compute:
                # build input key
                assert all(TT.count(iv) <= 1 for iv in num_cmp)
                inp_id: Tuple[Union[None, int], ...]
                inp_id = tuple(_index(TT, iv) for iv in num_cmp)
                # filter external keys
                # 1. they must share size with composed key
                # 2. all external partialities
                #     if corresponding composed partiality is among targets:
                #         must be among sources
                #     else:
                #         mustnt be among sources
                # Note. Second requirement assumes that opetors handled with direct
                #       backward direct operators only have (<=)1st order derivatives.
                filtered_ED: dict[Tuple[int, ...], Union[None, Tensor]] = dict()
                for num_evs, v in ED.items():
                    if len(num_evs) == len(num_cmp):  # avoid unnecesary iterations
                        ext_spots: Tuple[bool, ...]
                        ext_spots = _spot_active_variables(num_evs, SS)
                        if len(num_evs) == len(num_cmp) and cmp_spots == ext_spots:
                            filtered_ED[num_evs] = v
                # ...
                for num_evs, external_derivative in filtered_ED.items():
                    # set default composed derivative to None (full zero)
                    composed_derivative: Union[None, Tensor] = None
                    composed_shapes: Union[None, Tuple[Shape, ...]] = None
                    composed_indeps: Union[None, Tuple[Indep, ...]] = None
                    composed_vperm: Union[None, VPerm] = EP[num_evs]
                    # ---
                    if external_derivative is not None:
                        # obtain reversed vperm
                        depermutation: Tuple[int, ...]
                        permutation: Union[None, Tuple[int, ...]] = EP[num_evs]
                        assert permutation is not None
                        depermutation = reverse_permutation(permutation=permutation)
                        # build output key
                        assert all(SS.count(ev) <= 1 for ev in num_evs)
                        out_id: Tuple[Union[None, int], ...]
                        out_id = tuple(_index(SS, ev) for ev in num_evs)
                        # retrieve [current & expected] [shapes & indeps] of the
                        #     external derivative
                        explicit_external_shapes: list[Shape] = list()
                        explicit_external_indeps: list[Indep] = list()
                        explicit_expected_shapes: list[Shape] = list()
                        explicit_expected_indeps: list[Indep] = list()
                        assert len(num_evs) == len(num_cmp)
                        pairs: list[Tuple[int, int]]
                        pairs = list(zip(num_evs, num_cmp))
                        pairs = [pairs[d] for d in depermutation]
                        for ev, iv in pairs:
                            explicit_external_shapes.append(_populated(ES[ev]))
                            explicit_external_indeps.append(_populated(EI[ev]))
                            explicit_expected_shapes.append(_populated(XS[ev]))
                            expected_indep: Union[None, Indep] = XI[(ev, iv)]
                            assert expected_indep is not None
                            explicit_expected_indeps.append(expected_indep)
                        # align external derivative
                        int_variables: Tuple[int, ...] = tuple(range(len(cmp_key)))
                        aligned_external_derivative: Tensor = align_derivative(
                            derivative=external_derivative,
                            variables=int_variables,
                            variable_perm=tuple(range(len(int_variables))),
                            shapes=tuple(explicit_external_shapes),
                            indeps=tuple(explicit_external_indeps),
                            expected_shapes=tuple(explicit_expected_shapes),
                            expected_indeps=tuple(explicit_expected_indeps),
                            keepdim=True,
                        )
                        # transform external derivative into composed derivative
                        explicit_internal_shapes: Union[None, Tuple[Shape, ...]] = None
                        explicit_internal_indeps: Union[None, Tuple[Indep, ...]] = None
                        assert isinstance(fn, DirectFunction)
                        (
                            composed_derivative,
                            explicit_internal_shapes,
                            explicit_internal_indeps,
                        ) = fn.transform(
                            derivative=aligned_external_derivative,
                            shapes=tuple(explicit_expected_shapes),
                            indeps=tuple(explicit_expected_indeps),
                            out_id=tuple(out_id[d] for d in depermutation),
                            inp_id=tuple(inp_id[d] for d in depermutation),
                        )
                        # restructure shapes and indeps to implicit format
                        if composed_derivative is None:
                            continue
                        depermuted_num_cmp: Tuple[int, ...]
                        depermuted_num_cmp = tuple(num_cmp[d] for d in depermutation)
                        unique_num_cmp: Tuple[int, ...]
                        unique_num_cmp = tuple(dict.fromkeys(depermuted_num_cmp))
                        aux_shapes: list[Shape] = list()
                        aux_indeps: list[Indep] = list()
                        for iv in unique_num_cmp:
                            assert explicit_internal_shapes is not None
                            assert explicit_internal_indeps is not None
                            cv: int = depermuted_num_cmp.index(iv)
                            aux_shapes.append(explicit_internal_shapes[cv])
                            aux_indeps.append(explicit_internal_indeps[cv])
                        composed_shapes = tuple(aux_shapes)
                        composed_indeps = tuple(aux_indeps)
                    # save composed derivative data
                    present: bool = cmp_key in composed_derivatives
                    assert not present or None in composed_derivatives[cmp_key]
                    CD_data: EDData = (
                        composed_derivative,
                        composed_shapes,
                        composed_indeps,
                        composed_vperm,
                    )
                    composed_derivatives[cmp_key] = CD_data
            else:
                composed_derivatives[cmp_key] = (None, None, None, None)

        return composed_derivatives

    def _update_grid(
        self,
        V: VariableOperator,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
        composed_derivatives: dict[Tuple[Node, ...], EDData],
    ) -> None:

        # remove external variable diffs from derivative_grid
        for o in range(1, 1 + self._order):
            non_terminal_evs: list[Node]
            non_terminal_evs = [ev for ev in V.evs if ev not in self._terminals]
            self._grid.remove(variables=non_terminal_evs)
        # save composed derivatives corresponding to new variables
        keys: list[Tuple[Node, ...]] = list()
        while composed_derivatives:
            key: Tuple[Node, ...]
            new_ED_data: EDData
            key, new_ED_data = composed_derivatives.popitem()
            keys.append(key)
            o: int = len(key)
            assert o > 0 and o <= self._order
            acc_ED_data: EDData = self._grid[key]
            null_update: bool = None in new_ED_data
            null_stored: bool = None in acc_ED_data
            exists_null: bool = null_update or null_stored
            if self._grid.contains(key) and not exists_null:
                # depermute varaibles and shapes of new ED_data
                dep_key_new: Tuple[Node, ...]
                dep_key_new = tuple(
                    key[d] for d in reverse_permutation(_populated(new_ED_data[3]))
                )
                var_map_new: dict[Node, int]
                var_map_new = {v: i for i, v in enumerate(dict.fromkeys(dep_key_new))}
                dep_variables_new: Tuple[int, ...]
                dep_variables_new = tuple(var_map_new[v] for v in dep_key_new)
                depermuted_new_shapes: Tuple[Indep, ...]
                depermuted_new_shapes, _ = depermute_metadata(
                    variables=dep_variables_new,
                    shapes=_populated(new_ED_data[1]),
                    indeps=_populated(new_ED_data[2]),
                    permutation=_populated(new_ED_data[3]),
                )
                # depermute varaibles and shapes of accumulated ED_data
                dep_key_acc: Tuple[Node, ...]
                dep_key_acc = tuple(
                    key[d] for d in reverse_permutation(_populated(acc_ED_data[3]))
                )
                var_map_acc: dict[Node, int]
                var_map_acc = {v: i for i, v in enumerate(dict.fromkeys(dep_key_acc))}
                dep_variables_acc: Tuple[int, ...]
                dep_variables_acc = tuple(var_map_acc[v] for v in dep_key_acc)
                depermuted_acc_shapes: Tuple[Indep, ...]
                depermuted_acc_shapes, _ = depermute_metadata(
                    variables=dep_variables_acc,
                    shapes=_populated(acc_ED_data[1]),
                    indeps=_populated(acc_ED_data[2]),
                    permutation=_populated(acc_ED_data[3]),
                )
                assert depermuted_new_shapes == depermuted_acc_shapes  # Note. ->
                # -> consider doing best effort unification instead
                new_derivative: Tensor = _populated(new_ED_data[0])
                acc_derivative: Tensor = _populated(acc_ED_data[0])
                unified_indeps: Tuple[Indep, ...] = _populated(new_ED_data[2])
                if new_ED_data[2] != acc_ED_data[2]:
                    # unify depermuted indeps
                    zip_shapes: zip = zip(
                        _populated(new_ED_data[1]), _populated(acc_ED_data[1])
                    )
                    zip_indeps: zip = zip(
                        _populated(new_ED_data[2]), _populated(acc_ED_data[2])
                    )
                    unified_indeps = tuple(
                        unify_indeps(
                            indeps=indep_pair,
                            shapes_ndims=[len(shape) for shape in shape_pair],
                            inclusive=False,
                        )
                        for shape_pair, indep_pair in zip(zip_shapes, zip_indeps)
                    )
                    # fix derivatives to match in indeps and vperm
                    var_map: dict[Node, int]
                    var_map = {v: i for i, v in enumerate(dict.fromkeys(key))}
                    variables: Tuple[int, ...] = tuple(var_map[v] for v in key)
                    # align new derivative
                    align_new: bool = False
                    align_new |= new_ED_data[2] != unified_indeps
                    align_new |= new_ED_data[3] != acc_ED_data[3]
                    if align_new:
                        variables_new: Tuple[int, ...] = dep_variables_new
                        shapes_new: Tuple[Shape, ...] = _populated(new_ED_data[1])
                        indeps_new: Tuple[Indep, ...] = _populated(new_ED_data[2])
                        expected_shapes_new: Tuple[Shape, ...]
                        expected_shapes_new = _populated(new_ED_data[1])
                        unified_indeps_new: Tuple[Indep, ...] = unified_indeps
                        permutation_new: VPerm = tuple(
                            range(len(_populated(new_ED_data[3])))
                        )
                        if new_ED_data[3] != acc_ED_data[3]:
                            permutation_new = _populated(new_ED_data[3])
                        new_derivative = align_derivative(
                            derivative=new_derivative,
                            variables=variables_new,
                            variable_perm=permutation_new,
                            shapes=shapes_new,
                            indeps=indeps_new,
                            expected_shapes=expected_shapes_new,
                            expected_indeps=unified_indeps_new,
                            keepdim=True,
                        )
                    # align accumulated derivative
                    align_acc: bool = False
                    align_acc |= acc_ED_data[2] != unified_indeps
                    align_acc |= new_ED_data[3] != acc_ED_data[3]
                    if align_acc:
                        variables_acc: Tuple[int, ...] = dep_variables_acc
                        shapes_acc: Tuple[Shape, ...] = _populated(acc_ED_data[1])
                        indeps_acc: Tuple[Indep, ...] = _populated(acc_ED_data[2])
                        expected_shapes_acc: Tuple[Shape, ...]
                        expected_shapes_acc = _populated(acc_ED_data[1])
                        unified_indeps_acc: Tuple[Indep, ...] = unified_indeps
                        permutation_acc: VPerm = tuple(
                            range(len(_populated(acc_ED_data[3])))
                        )
                        if new_ED_data[3] != acc_ED_data[3]:
                            permutation_acc = _populated(acc_ED_data[3])
                        acc_derivative = align_derivative(
                            derivative=acc_derivative,
                            variables=variables_acc,
                            variable_perm=permutation_acc,
                            shapes=shapes_acc,
                            indeps=indeps_acc,
                            expected_shapes=expected_shapes_acc,
                            expected_indeps=unified_indeps_acc,
                            keepdim=True,
                        )
                updated_ED_data: EDData = (None, None, None, None)
                if None not in (new_derivative, acc_derivative):
                    derivative_sum: Tensor = new_derivative + acc_derivative
                    updated_ED_data = (
                        derivative_sum,
                        new_ED_data[1],
                        unified_indeps,
                        tuple(range(len(_populated(new_ED_data[3])))),
                    )
                if acc_derivative is None and new_derivative is not None:
                    updated_ED_data = new_ED_data
                self._grid[key] = updated_ED_data

            elif self._grid.contains(key) and null_update and not null_stored:
                assert None not in acc_ED_data
                self._grid[key] = acc_ED_data
            else:  # standard case
                self._grid[key] = new_ED_data

        self._apply_hooks(fns=fns)
        self._save_gradients(
            gradients={key: self._grid[key] for key in keys},
        )
        return None

    def contractive_update(
        self,
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ],
        groups: list[set[Node]],
    ) -> None:

        assert self._initialized

        ### Gather (external & internal) variables
        # instantiate interface to manage variables
        V: VariableOperator = VariableOperator(fns=fns, grid=self._grid)
        # instantiate mappers
        ev_mapper: IdxMapper = IdxMapper(objects=V.all_evs)
        iv_mapper: IdxMapper = IdxMapper(objects=V.all_ivs)

        ### Determine control factors related whith the graph topology
        # determine which composed derivatives need to be computed
        required_differentiations: dict[Tuple[Node, ...], bool]
        required_differentiations = self._plan_differentiations(V=V, groups=groups)
        # determine the composed derivatives whose indeps size can be modified
        internal_flexibilities: dict[Node, bool] = self._determine_flexibilities(
            V=V,
            groups=groups,
        )
        # determine which internal variables satisfy schwarz hypothesis
        internal_schwarz: dict[Node, bool]
        internal_schwarz = self._determine_schwarz_condition(V=V, fns=fns)

        ### Acquire external derivatives
        external_derivatives: dict[Tuple[Node, ...], Union[None, Tensor]]
        external_vperms: dict[Tuple[Node, ...], Union[None, VPerm]]
        (
            external_derivatives,
            external_vperms,
        ) = self._acquire_external_derivatives(V=V)

        ### Aquire external variable shapes
        external_shapes: dict[Node, Union[None, Shape]]
        external_indeps: dict[Node, Union[None, Indep]]
        expected_shapes: dict[Node, Union[None, Shape]]
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]]
        (
            external_shapes,
            external_indeps,
            expected_shapes,
            expected_indeps,
        ) = self._acquire_derivative_metadata(
            V=V,
            fns=fns,
            required_differentiations=required_differentiations,
        )

        ### Compute internal derivatives
        internal_derivatives: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Tensor]]
        einstein_notations: dict[Tuple[Node, Tuple[Node, ...]], Union[None, Notation]]
        (internal_derivatives, einstein_notations) = self._compute_internal_derivatives(
            V=V,
            fns=fns,
            required_differentiations=required_differentiations,
        )
        # remove null dimensions expected shapes and expected indeps
        (expected_shapes, expected_indeps, einstein_notations) = self._denull_internals(
            external_indeps=external_indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            einstein_notations=einstein_notations,
        )

        ### Compose derivatives
        composed_derivatives: dict[Tuple[Node, ...], EDData]
        composed_derivatives = self._compose_derivatives(
            V=V,
            required_differentiations=required_differentiations,
            mappers=(ev_mapper, iv_mapper),
            external_derivatives=external_derivatives,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
            external_vperms=external_vperms,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            internal_derivatives=internal_derivatives,
            internal_flexibilities=internal_flexibilities,
            internal_schwarz=internal_schwarz,
            einstein_notations=einstein_notations,
        )
        # remove null dimensions from partials for further propagation
        composed_derivatives = self._denull_derivatives(
            derivatives=composed_derivatives,
        )

        ### Update grid
        self._update_grid(V=V, fns=fns, composed_derivatives=composed_derivatives)

        return None

    def direct_update(
        self,
        fn: ExtendedAutogradFunction,
        sources: Tuple[Node, ...],
        targets: Tuple[Union[None, Node], ...],
        groups: list[set[Node]],
    ) -> None:

        assert self._initialized

        ### Gather (external & internal) variables
        fns: dict[
            ExtendedAutogradFunction,
            Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
        ]
        fns = {fn: (sources, targets)}
        V: VariableOperator = VariableOperator(fns=fns, grid=self._grid)
        # instantiate mappers
        ev_mapper: IdxMapper = IdxMapper(objects=V.all_evs)
        iv_mapper: IdxMapper = IdxMapper(objects=V.all_ivs)

        ### Determine control factors related whith the graph topology
        # determine which composed derivatives need to be computed
        required_differentiations: dict[Tuple[Node, ...], bool]
        required_differentiations = self._plan_differentiations(V=V, groups=groups)
        # determine which internal variables satisfy schwarz hypothesis
        internal_schwarz: dict[Node, bool]
        internal_schwarz = self._determine_schwarz_condition(V=V, fns=fns)

        ### Acquire external derivatives
        external_derivatives: dict[Tuple[Node, ...], Union[None, Tensor]]
        external_vperms: dict[Tuple[Node, ...], Union[None, VPerm]]
        (
            external_derivatives,
            external_vperms,
        ) = self._acquire_external_derivatives(V=V)

        ### Aquire external variable shapes
        external_shapes: dict[Node, Union[None, Shape]]
        external_indeps: dict[Node, Union[None, Indep]]
        expected_shapes: dict[Node, Union[None, Shape]]
        expected_indeps: dict[Tuple[Node, Node], Union[None, Indep]]
        (
            external_shapes,
            external_indeps,
            expected_shapes,
            expected_indeps,
        ) = self._acquire_derivative_metadata(
            V=V,
            fns=fns,
            required_differentiations=required_differentiations,
        )

        ### Transform derivatives
        composed_derivatives: dict[Tuple[Node, ...], EDData]
        composed_derivatives = self._transform_derivatives(
            V=V,
            fn=fn,
            sources=sources,
            targets=targets,
            required_differentiations=required_differentiations,
            mappers=(ev_mapper, iv_mapper),
            external_derivatives=external_derivatives,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
            external_vperms=external_vperms,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            internal_schwarz=internal_schwarz,
        )
        # remove null dimensions from partials for further propagation
        composed_derivatives = self._denull_derivatives(
            derivatives=composed_derivatives,
        )

        ### Update grid
        self._update_grid(V=V, fns=fns, composed_derivatives=composed_derivatives)

        return None
