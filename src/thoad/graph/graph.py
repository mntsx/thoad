# Standard Library Dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.graph.structures import Node, MultiEdge
from thoad.autodifferentiation.initialization.assignment import FunctionTranscoder
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction


class Graph:
    def __init__(self, tensor: Tensor) -> None:
        self._source_tensor: Tensor = tensor
        self._nodes: set["Node"] = set()
        self._terminals: set["Node"] = dict()
        self._edges: dict[torch.autograd.Function, "MultiEdge"] = dict()
        self._initial_node: "Node" = self._build()
        self._transcoder: FunctionTranscoder = FunctionTranscoder()
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
        nodes: dict[Tuple[Union[Tensor, torch.autograd.Function], int], Node] = {}
        edges: dict[torch.autograd.Function, MultiEdge] = {}

        def build_node(curr_fn: torch.autograd.Function, idx: int) -> Node:
            # if there's no function (leaf), skip
            assert curr_fn is not None
            key = (curr_fn, idx)
            if key in nodes:
                return nodes[key]
            node: "Node" = Node()
            nodes[key] = node
            if curr_fn not in edges:
                edges[curr_fn] = MultiEdge(curr_fn)
            me: "MultiEdge" = edges[curr_fn]
            node.register_multiedge(multiedge=me, index=idx)
            me.register_source(output_idx=idx, node=node)
            # now link all parents _ this node
            for input_idx, (child_fn, child_idx) in enumerate(curr_fn.next_functions):
                if child_fn is None:
                    continue
                child_node = build_node(child_fn, child_idx)
                me.register_target(input_idx=input_idx, node=child_node)
            if len(curr_fn.next_functions) == 0:
                assert "variable" in dir(curr_fn)
                leaf: Tensor = curr_fn.variable
                key: Tuple[Tensor, int] = (leaf, 0)
                terminal_node: "Node" = Node()
                if key in nodes:
                    terminal_node = nodes[key]
                nodes[key] = terminal_node
                self._terminals[terminal_node] = leaf
                terminal_node.link(edge=me, size=tuple(leaf.shape))
                me.register_target(input_idx=0, node=terminal_node)
            return node

        # Take the very first grad_fn from `tensor.sum().grad_fn.next_functions[0]`
        first_next_fn: torch.autograd.Function
        first_idx: int
        first_next_fn, first_idx = self._source_tensor.sum().grad_fn.next_functions[0]
        root_node: Node = build_node(first_next_fn, first_idx)
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
    def nodes(self) -> set["Node"]:
        return set(self._nodes.values())

    @property
    def terminals(self) -> dict["Node", Tensor]:
        return self._terminals

    def find_node(self, tensor: Tensor) -> "Node":
        grad_fn: torch.autograd.Function = tensor.grad_fn
        target_node: Union[None, "Node"] = None
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
        if target_node is None:
            for node in self._nodes.values():
                if node.multiedge is not None:
                    if node.multiedge.gfn == grad_fn:
                        target_node = node
        assert target_node is not None
        return target_node

    @property
    def index(
        self,
    ) -> dict[torch.autograd.Function, ExtendedAutogradFunction]:
        return self._transcoder.index

    @property
    def transcoder(self) -> FunctionTranscoder:
        return self._transcoder

    @property
    def compatible(self) -> bool:
        def _compatible(
            grad_fn: torch.autograd.Function,
            transcoder: FunctionTranscoder,
        ) -> bool:
            compatible: bool = True
            for gfn, _ in grad_fn.next_functions:
                compatible *= _compatible(
                    grad_fn=gfn,
                    transcoder=transcoder,
                )
            compatible *= self._transcoder.supports(grad_fn=grad_fn)
            return compatible

        compatible: bool = _compatible(
            grad_fn=self._source_tensor.grad_fn,
            transcoder=self._transcoder,
        )
        return compatible
