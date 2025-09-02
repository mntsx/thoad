# Standard Library Dependencies
from typing import Iterable, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.control.composition import GradOperator
from thoad.differentiation.internals.base import ExtendedAutogradFunction
from thoad.graph.graph import Graph
from thoad.graph.structures import Node, MultiEdge
from thoad.typing import Hook, PopulatedEDData


class BackpropOrchestrator:

    def __init__(self) -> None:
        # control attributes
        self._propagated: bool = False
        # propagation attributes
        self._expanded_nodes: set[Node] = set()
        self._expanded_edges: set[MultiEdge] = set()
        self._active_nodes: set[Node] = set()
        self._expansion_candidates: set[MultiEdge] = set()
        # configuration attibutes
        self._terminal_groups: list[set[Node]] = list()
        self._bloking_keys: set[Tuple[Node, ...]] = set()
        # instrumental interfaces
        self._grad_operator: GradOperator = GradOperator()
        self._graph: Union[None, Graph] = None

        return None

    def setup_graph(self, tensor: Tensor) -> None:
        self._graph = Graph(tensor=tensor)
        self._terminal_groups.clear()
        self._bloking_keys.clear()
        return None

    @property
    def graph(self) -> "Graph":
        assert self._graph is not None
        return self._graph

    @property
    def cross_terminals(self) -> bool:
        return self._grad_operator.cross_terminals

    @cross_terminals.setter
    def cross_terminals(self, value: bool) -> None:
        self._grad_operator.cross_terminals = value
        return None

    @property
    def keep_batch(self) -> bool:
        return self._grad_operator.keep_batch

    @keep_batch.setter
    def keep_batch(self, value: bool) -> None:
        self._grad_operator.keep_batch = value
        return None

    @property
    def keep_schwarz(self) -> bool:
        return self._grad_operator.keep_vperm

    @keep_schwarz.setter
    def keep_schwarz(self, value: bool) -> None:
        self._grad_operator.keep_vperm = value
        return None

    def fetch_hgrad(
        self,
        key: Tuple[Tensor, ...],
        keep_batch: bool,
        keep_schwarz: bool,
    ) -> PopulatedEDData:
        assert self._graph is not None
        if not self._propagated:
            raise RuntimeError(
                "fetch_hgrad called before backpropagation. "
                "Call backward first to propagate gradients."
            )
        key_nodes: Tuple[Node, ...] = tuple(
            self._graph.find_node(tensor=T, raw=True) for T in key
        )
        ED_data: PopulatedEDData = self._grad_operator.fetch_hgrad(
            key=key_nodes, keep_batch=keep_batch, keep_vperm=keep_schwarz
        )
        return ED_data

    def clear(self) -> None:
        if not self._propagated:
            raise RuntimeError(
                "clear called before backpropagation. "
                "Call backward first to propagate gradients."
            )
        self._grad_operator.clear_gradients()
        return None

    def _check_key_reachability(self, key: Tuple[Tensor, ...]) -> None:
        """
        This function determines if all required combinations of nodes are reacheable
        within backpropagation.
        """
        assert self._graph is not None
        key_nodes: Tuple[Node, ...]
        key_nodes = tuple(self._graph.find_node(tensor=T) for T in key)
        blocking_keys: list[Tuple[Node, ...]] = [key_nodes, *self._bloking_keys]
        # blocking_edges: set[Node] = set()
        self._expanded_nodes = {self._graph.find_node(self._graph.source_tensor)}
        self._expanded_edges = set()
        self._active_nodes = set()
        self._expansion_candidates = set()
        blocked: bool = False
        while not blocked:
            self._update_active_nodes()
            self._update_expansion_candidates()
            blocking_edges: set[MultiEdge] = set()
            frontier: set[Node] = set()
            for E in self._expansion_candidates:
                frontier.update(E.filtered_sources)
            blocking_keys = [k for k in blocking_keys if not set(k).issubset(frontier)]
            for N in (N for nodes in blocking_keys for N in nodes):
                if N.multiedge is not None:
                    blocking_edges.add(N.multiedge)
            pruned_candidates: set[MultiEdge]
            pruned_candidates = self._expansion_candidates - blocking_edges
            self._expanded_edges.update(pruned_candidates)
            self._expanded_nodes.update(
                N for E in pruned_candidates for N in E.filtered_targets
            )
            self._expansion_candidates.clear()
            blocked = len(pruned_candidates) == 0
        # restore global varaibles
        self._expanded_nodes = set()
        self._expanded_edges = set()
        self._active_nodes = set()
        self._expansion_candidates = set()
        # determine feasibility
        feasible: bool = len(blocking_keys) == 0
        if not feasible:
            raise ValueError(
                "Introduced variable combination is unreachable during backpropagation."
            )
        return None

    def add_gradient_retention(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        self._check_key_reachability(key=key)
        key_nodes: Tuple[Node, ...]
        key_nodes = tuple(self._graph.find_node(tensor=T, raw=False) for T in key)
        self._bloking_keys.add(key_nodes)
        self._grad_operator.add_gradient_retention(key=key_nodes)
        return None

    def drop_gradient_retention(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        key_nodes: Tuple[Node, ...]
        key_nodes = tuple(self._graph.find_node(tensor=T, raw=True) for T in key)
        self._grad_operator.drop_gradient_retention(key=key_nodes)
        return None

    def add_backward_hook(
        self,
        key: Tuple[Tensor, ...],
        hook: Hook,
    ) -> None:
        assert self._graph is not None
        self._check_key_reachability(key=key)
        key_nodes: Tuple[Node, ...]
        key_nodes = tuple(self._graph.find_node(tensor=T, raw=False) for T in key)
        self._bloking_keys.add(key_nodes)
        connecting_functions: dict[MultiEdge, set[Tensor]] = dict()
        for E in self._graph.edges:
            connects: bool = False
            tensors: set[Tensor] = set()
            for N in E.filtered_targets:
                if N in key_nodes:
                    connects = True
                    tensors.add(key[key_nodes.index(N)])
            if connects:
                connecting_functions[E] = tensors
        self._grad_operator.add_backward_hook(
            key=key_nodes,
            hook=hook,
            connecting_functions=connecting_functions,
        )
        return None

    def drop_backward_hook(self, key: Tuple[Tensor, ...]) -> None:
        assert self._graph is not None
        key_nodes: Tuple[Node, ...]
        key_nodes = tuple(self._graph.find_node(tensor=T, raw=True) for T in key)
        self._grad_operator.drop_backward_hook(key=key_nodes)
        return None

    def _update_active_nodes(self) -> None:
        """
        This function collects expanded nodes with unexpanded edges
        """
        self._active_nodes = set()
        dependencies: set[Node] = set()
        for N in self._expanded_nodes:
            if N.multiedge not in {None, *self._expanded_edges}:
                self._active_nodes.add(N)
                dependencies.update(N.collect_dependencies() - {N})
        self._active_nodes -= dependencies
        return None

    def _block_key_nodes(self) -> None:
        reached_nodes: set[Node] = set((*self._active_nodes, *self._expanded_nodes))
        for key_nodes in self._bloking_keys:
            key_reached: bool = all(k in reached_nodes for k in key_nodes)
            if not key_reached:
                self._active_nodes -= set(key_nodes)
        return None

    def _update_expansion_candidates(self) -> None:
        """
        This function collects multiedges that verify:
            1. all its sources are expanded.
            2. all its targets are in frontier (one step from active nodes)
        Note. Every node has one single multiedge.
              Therefore, set of active nodes is set of active multiedges sources
        """
        # get active edges
        active_edges: set[MultiEdge] = set()
        for N in self._active_nodes:
            multiedge: Union[None, MultiEdge] = N.multiedge
            if multiedge is not None:
                active_edges.add(multiedge)
        # prune active edges with non-reachable targets
        for E in active_edges:
            if all(N in self._active_nodes for N in E.sources):
                self._expansion_candidates.add(E)
        return None

    def _group_by_dependencies(self, frontier: Iterable[Node]) -> list[set[Node]]:
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

        assert self._graph is not None
        for start_node in self._graph.nodes:
            if any(c in frontier for c in start_node.childs):
                _join_terminals(node=start_node, path={start_node})
        groups.extend(enforced_groups)
        # remove redundant groups
        reduced_groups: list[set[Node]] = []
        seen: set[frozenset[Node]] = set()
        for G1 in groups:
            key: frozenset[Node] = frozenset(G1)
            if key in seen:
                continue
            # strict subset operator: True only if G1 ⊆ G2 and G1 != G2
            is_strict_subset: bool = any(G1 < G2 for G2 in groups)
            if not is_strict_subset:
                reduced_groups.append(G1)
                seen.add(key)
        groups = reduced_groups

        return groups

    def _expand_nodes(self) -> None:
        # if exists direct function among candidates -> expand only direct functions
        candidates: list[MultiEdge] = list(self._expansion_candidates)
        frontier: set[Node] = set()
        groups: list[set[Node]]
        direct_checks: list[bool] = ["direct" in E.xfn.method for E in candidates]
        if any(direct_checks):
            frontier.update(set((T for E in candidates for T in E.filtered_targets)))
            for E, check in zip(candidates, direct_checks):
                groups = self._group_by_dependencies(frontier=frontier)
                if check:
                    self._grad_operator.direct_update(
                        fn=E.xfn,
                        sources=E.filtered_sources,
                        targets=E.targets,
                        groups=groups,
                    )
                    self._expanded_nodes.update(E.filtered_targets)
                    self._expanded_edges.add(E)
                    self._expansion_candidates.remove(E)
        elif len(candidates) > 0:
            fns: dict[
                ExtendedAutogradFunction,
                Tuple[Tuple[Node, ...], Tuple[Union[None, Node], ...]],
            ]
            fns = dict()
            for E in candidates:
                sources: Tuple[Node, ...] = E.filtered_sources
                targets: Tuple[Union[None, Node], ...] = E.targets
                frontier.update(E.filtered_targets)
                fns[E.xfn] = (sources, targets)

            groups = self._group_by_dependencies(frontier=frontier)
            self._grad_operator.contractive_update(fns=fns, groups=groups)
            self._expanded_nodes.update(frontier)
            self._expanded_edges.update(candidates)
            self._expansion_candidates.clear()
        return None

    def _step(self) -> None:
        self._update_active_nodes()
        self._block_key_nodes()
        self._update_expansion_candidates()
        self._expand_nodes()
        return None

    def propagate(
        self,
        order: int,
        groups: list[Iterable[Tensor]],
        gradient: Union[None, Tensor],
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
            order=order,
            node=source_node,
            tensor=source_tensor,
            gradient=gradient,
        )
        self._expanded_nodes = {source_node}
        self._expanded_edges = set()
        self._active_nodes = set()
        self._expansion_candidates = set()
        # gather terminal node groups
        self._terminal_groups = list()
        for tensor_group in groups:
            node_group: set[Node] = set()
            for T in tensor_group:
                node: Node = self._graph.find_node(tensor=T, raw=True)
                if node not in terminals:
                    raise ValueError(
                        f"Cannot propagate: tensor corresponds to a node that is "
                        f"not in the computational graph."
                    )
                node_group.add(node)
            self._terminal_groups.append(node_group)
        # initialize variable retentions
        self._grad_operator.initialize_retentions(
            order=order,
            groups=self._terminal_groups,
        )
        # initialize hooks
        self._grad_operator.initialize_hooks()

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
