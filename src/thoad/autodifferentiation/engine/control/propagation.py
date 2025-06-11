# Standard Library Dependencies
from typing import Any, Callable, Iterable, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.control.composition import GradOperator
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction
from thoad.graph.graph import Graph
from thoad.graph.structures import Node, MultiEdge
from thoad.typing.data import EDData


class Controller:

    def __init__(self) -> None:
        # control attributes
        self._propagated: bool = False
        # propagation attributes
        self._expanded_nodes: set[Node] = set()
        self._active_nodes: set[Node] = set()
        self._expansion_candidates: set[MultiEdge] = set()
        # configuration attibutes
        self._terminal_groups: list[set[Node]] = list()
        # instrumental interfaces
        self._grad_operator: GradOperator = GradOperator()
        self._graph: Union[None, Graph] = None

        return None

    def setup_graph(self, tensor: Tensor) -> None:
        self._graph = Graph(tensor=tensor)
        return None

    @property
    def graph(self) -> "Graph":
        return self._graph

    @property
    def cross_terminals(self) -> bool:
        return self._grad_operator.cross_terminals

    @cross_terminals.setter
    def cross_terminals(self, value: bool) -> None:
        self._grad_operator.cross_terminals = value
        return None

    @property
    def keepbatch(self) -> bool:
        return self._grad_operator.keepbatch

    @keepbatch.setter
    def keepbatch(self, value: bool) -> None:
        self._grad_operator.keepbatch = value
        return None

    def fetch_hgrad(self, key: Tuple[Tensor, ...], keepbatch: bool) -> EDData:
        assert self._graph is not None
        if not self._propagated:
            raise RuntimeError(
                "Fetch hgrad called before backpropagation. "
                "Call backward first to propagate gradients."
            )
        node_key: Tuple[Node] = tuple(self._graph.find_node(T) for T in key)
        ED_data: EDData = self._grad_operator.fetch_hgrad(
            key=node_key, keepbatch=keepbatch
        )
        return ED_data

    def add_gradient_retention(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        node_key: Tuple[Node] = tuple(self._graph.find_node(tensor=T) for T in key)
        self._grad_operator.add_gradient_retention(key=node_key, tensors=key)
        return None

    def drop_gradient_retention(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        node_key: Tuple[Node] = tuple(self._graph.find_node(tensor=T) for T in key)
        self._grad_operator.drop_gradient_retention(key=node_key)
        return None

    def add_backward_hook(
        self,
        key: Tuple[Tensor, ...],
        hook: Callable[[EDData, list[dict["str", Any]]], EDData],
    ) -> None:
        assert self._graph is not None
        node_key: Tuple[Node] = tuple(self._graph.find_node(tensor=T) for T in key)
        self._grad_operator.add_backward_hook(key=node_key, hook=hook)
        return None

    def drop_backward_hook(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        node_key: Tuple[Node] = tuple(self._graph.find_node(tensor=T) for T in key)
        self._grad_operator.drop_backward_hook(key=node_key)
        return None

    def _update_active_nodes(self) -> None:
        """
        This function collects expanded nodes with unexpanded childs
        """
        self._active_nodes = set()
        for N in self._expanded_nodes:
            exists_unexpanded_child: bool = False
            for NN in N.childs:
                child_unexpanded: bool = NN not in self._expanded_nodes
                exists_unexpanded_child = exists_unexpanded_child or child_unexpanded
            if exists_unexpanded_child:
                self._active_nodes.add(N)
        return None

    def _update_expansion_candidates(self) -> None:
        """
        This function collects multiedges that verify:
            1. all its sources are expanaded.
            2. all its targets are in frontier (one step from active nodes)
        Note. Every node has one single multiedge.
              Therefore, set of active nodes is set of active multiedges sources
        """
        # get active edges
        active_edges: set[ExtendedAutogradFunction] = set()
        for N in self._active_nodes:
            active_edges.add(N.multiedge)
        # prune active edges with non-reachable targets
        for E in active_edges:
            if all(N in self._active_nodes for N in E.sources):
                self._expansion_candidates.add(E)
        return None

    def _group_by_dependencies(self, frontier=Iterable[Node]) -> list[set[Node]]:
        """
        This function groups nodes by their linked dependencies.
        """
        # determine which nodes need to be crossed due to future joint
        groups: list[set[Node]] = [N.collect_dependencies() for N in frontier]
        converged: bool = False
        while not converged:
            converged = True
            new_groups: list[set[Node]] = []
            while len(groups) > 0:
                current: set[Node] = groups.pop(0)
                i: int = 0
                while i < len(groups):
                    if len(current.intersection(groups[i])) > 0:
                        current |= groups.pop(i)
                        converged = False
                    else:
                        i += 1
                new_groups.append(current)
            groups = new_groups
        # determine which frontier nodes need to be crossed in order for future
        #   user-required crossing of terminal nodes
        enforced_groups: list[set[Node]] = [set() for _ in self._terminal_groups]

        def _join_terminals(node: Node, path: set[Node]) -> None:
            childs: set[Node] = node.childs
            if len(childs) > 0:
                for child in childs:
                    new_path: set[Node] = set((*path, child))
                    _join_terminals(node=child, path=new_path)
            else:
                for i, group in enumerate(self._terminal_groups):
                    if node in group:
                        enforced_groups[i] |= path

        for start_node in frontier:
            _join_terminals(node=start_node, path={start_node})
        groups.extend(enforced_groups)
        return groups

    def _expand_nodes(self) -> None:
        # if exists direct function among candidates -> expand only direct functions
        candidates: list[MultiEdge] = list(self._expansion_candidates)
        direct_checks: list[bool] = ["direct" in E.xfn.method for E in candidates]
        if any(direct_checks):
            for E, check in zip(candidates, direct_checks):
                if check:
                    assert len(E.sources) == 1
                    assert len(E.targets) == 1
                    self._grad_operator.direct_update(
                        fn=E.xfn,
                        source=E.sources[0],  # set(...) ??? dont think so
                        target=E.targets[0],
                    )
                self._expanded_nodes.add(E.targets[0])
                self._expansion_candidates.pop(E)
        elif len(candidates) > 0:
            frontier: set[Node] = set()
            fns: dict[ExtendedAutogradFunction, Tuple[dict[int, Node], dict[int, Node]]]
            fns = dict()
            for E in candidates:
                sources: Tuple[Node] = E.sources
                targets: Tuple[Node] = E.targets
                frontier.update(targets)
                fns[E.xfn] = (sources, targets)
            groups: list[set[Node]] = self._group_by_dependencies(frontier=frontier)
            self._grad_operator.contractive_update(fns=fns, groups=groups)
            self._expanded_nodes = self._expanded_nodes.union(frontier)
            self._expansion_candidates.clear()
        return None

    def _step(self) -> None:
        self._update_active_nodes()
        self._update_expansion_candidates()
        self._expand_nodes()
        return None

    def propagate(
        self,
        order: int,
        groups: list[Iterable[Tensor]],
    ) -> None:
        # obtain graph terminal nodes
        assert self._graph is not None
        terminals: dict[Node, Tensor] = self._graph.terminals
        self._grad_operator.terminals = terminals

        ### Initialize
        # initialize propagation variables
        source_tensor: Tensor = self._graph.source_tensor
        source_node: Node = self._graph.find_node(tensor=source_tensor)
        self._grad_operator.initialize_gradients(
            order=order, node=source_node, tensor=source_tensor
        )
        self._expanded_nodes = {self._graph.find_node(self._graph.source_tensor)}
        self._active_nodes = set()
        self._expansion_candidates = set()
        # gather terminal node groups
        self._terminal_groups = list()
        for tensor_group in groups:
            node_group: set[Node] = set()
            for T in tensor_group:
                node: Node = self._graph.find_node(tensor=T)
                if node not in terminals:
                    raise ValueError(
                        "Cannot propagate: tensor corresponds to a node that is "
                        "not in the computational graph."
                    )
                node_group.add(node)
            self._terminal_groups.append(node_group)
        # initialize variable retentions
        self._grad_operator.initialize_retentions(
            order=order,
            groups=self._terminal_groups,
        )

        ### Progapage
        max_steps: int = 300
        counter: int = 0
        all_expanded: bool = False
        while counter < max_steps and not all_expanded:
            self._step()
            all_expanded: bool = len(self._graph.nodes - self._expanded_nodes) == 0
            counter += 1
        self._grad_operator.attach_gradients()
        self._propagated = True

        return None
