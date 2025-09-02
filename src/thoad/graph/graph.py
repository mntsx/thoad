# Standard Library Dependencies
from typing import cast, Protocol, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.graph.structures import Node, MultiEdge
from thoad.differentiation.initialization.assignment import FunctionTranscoder
from thoad.differentiation.internals.base import ExtendedAutogradFunction
from thoad.typing import AutogradFunction


class _HasVariable(Protocol):
    variable: Tensor


class Graph:
    def __init__(self, tensor: Tensor) -> None:
        self._source_tensor: Tensor = tensor
        self._nodes: dict[Tuple[Union[Tensor, AutogradFunction], int], Node] = dict()
        self._terminals: dict[Node, Tensor] = dict()
        self._edges: dict[AutogradFunction, MultiEdge] = dict()
        self._initial_node: Node = self._build()
        self._transcoder = FunctionTranscoder()
        return None

    def _build(self) -> Node:
        """
        Walks the .grad_fn.next_functions chain of self._source_tensor
        and builds one Node+MultiEdge for each (grad_fn, idx) pair.
        The direction is: for each current grad_fn, look at its next_functions
        (its "parents" in PyTorch's backward graph), create/lookup each parent
        as a Node, then create/lookup the MultiEdge for the current grad_fn,
        register parent _ current. Recurse on each parent node.
        """
        nodes: dict[Tuple[Union[Tensor, AutogradFunction], int], Node] = {}
        edges: dict[AutogradFunction, MultiEdge] = {}

        def build_node(curr_fn: AutogradFunction, idx: int) -> Node:
            # if there's no function (leaf), skip
            assert curr_fn is not None
            key: Tuple[Union[Tensor, AutogradFunction], int] = (curr_fn, idx)
            if key in nodes:
                return nodes[key]
            node: Node = Node()
            nodes[key] = node
            if curr_fn not in edges:
                edges[curr_fn] = MultiEdge(curr_fn)
            me: MultiEdge = edges[curr_fn]
            node.register_multiedge(multiedge=me, index=idx)
            me.register_source(output_idx=idx, node=node)
            # now link all parents _ this node
            for input_idx, (child_fn, child_idx) in enumerate(curr_fn.next_functions):
                if child_fn is None:
                    me.register_target(input_idx=input_idx, node=None)
                    continue
                child_node: Node = build_node(child_fn, child_idx)
                me.register_target(input_idx=input_idx, node=child_node)
            if len(curr_fn.next_functions) == 0:
                assert "variable" in dir(curr_fn)
                leaf: Tensor = cast(_HasVariable, curr_fn).variable
                key = (leaf, 0)
                terminal_node: Node = Node()
                if key in nodes:
                    terminal_node = nodes[key]
                nodes[key] = terminal_node
                self._terminals[terminal_node] = leaf
                terminal_node.link(edge=me, shape=tuple(leaf.shape))
                me.register_target(input_idx=0, node=terminal_node)
            return node

        # Take the very first grad_fn from `tensor.sum().grad_fn.next_functions[0]`
        aux1: Union[None, AutogradFunction] = self._source_tensor.sum().grad_fn
        assert aux1 is not None
        first_next_fn: Union[None, AutogradFunction]
        first_idx: int
        first_next_fn, first_idx = aux1.next_functions[0]
        assert first_next_fn is not None

        root_node: Node
        root_node = build_node(first_next_fn, first_idx)
        # store into self._nodes / self._edges
        self._nodes = nodes
        self._edges = edges

        return root_node

    def transcode_fns(
        self,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        for E in self._edges.values():
            E.transcode(
                transcoder=self._transcoder,
                order=order,
                dtype=dtype,
                device=device,
            )
        return None

    @property
    def source_tensor(self) -> Tensor:
        return self._source_tensor

    @property
    def nodes(self) -> set[Node]:
        return set(self._nodes.values())

    @property
    def terminals(self) -> dict[Node, Tensor]:
        return self._terminals

    @property
    def edges(self) -> set[MultiEdge]:
        return set(self._edges.values())

    def find_node(self, tensor: Tensor, raw: bool = False) -> Node:
        grad_fn: Union[None, AutogradFunction] = tensor.grad_fn
        target_node: Union[None, Node] = None
        for key, node in self._nodes.items():
            if isinstance(key[0], Tensor):
                assert node.multiedge is None
                if key[0] is tensor:
                    target_node = node
        if target_node is None and grad_fn is None:
            raise ValueError(
                "Cannot find node: the provided tensor has no grad_fn and is not "
                "part of the computational graph."
            )
        if target_node is not None and not raw:
            target_node = None
            for node in self._nodes.values():
                if node.multiedge is not None:
                    if "variable" in dir(node.multiedge.gfn):
                        gfn: Union[AutogradFunction, None] = node.multiedge.gfn
                        assert gfn is not None
                        if cast(_HasVariable, gfn).variable is tensor:
                            target_node = node
            assert target_node is not None
        if target_node is None:
            for node in self._nodes.values():
                if node.multiedge is not None:
                    if node.multiedge.gfn == grad_fn:
                        target_node = node
        if target_node is None:
            raise ValueError(
                "Cannot find node: the provided tensor is not "
                "part of the computational graph."
            )
        return target_node

    @property
    def index(
        self,
    ) -> dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]:
        return self._transcoder.index

    @property
    def transcoder(self) -> FunctionTranscoder:
        return self._transcoder

    @property
    def compatible(self) -> bool:
        def _compatible(
            grad_fn: AutogradFunction,
            transcoder: FunctionTranscoder,
        ) -> bool:
            compatible: bool = True
            for gfn, _ in grad_fn.next_functions:
                if gfn is not None:
                    compatible &= _compatible(
                        grad_fn=gfn,
                        transcoder=transcoder,
                    )
            compatible &= self._transcoder.supports(grad_fn=grad_fn)
            return compatible

        grad_fn: Union[None, AutogradFunction] = self._source_tensor.grad_fn
        assert grad_fn is not None
        compatible: bool = _compatible(
            grad_fn=grad_fn,
            transcoder=self._transcoder,
        )
        return compatible
