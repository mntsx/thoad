# Standard Library Dependencies
import random
from typing import Tuple, Union, Type

# PyTorch dependencies
import torch

# Internal dependencies
from thoad.differentiation.internals.base import ExtendedAutogradFunction
from thoad.differentiation.initialization.assignment import FunctionTranscoder
from thoad.typing import AutogradFunction


class Node:
    def __init__(self) -> None:
        # core attributes
        self._multiedge: Union[None, "MultiEdge"] = None
        self._index: Union[None, int] = None
        # only for representation
        self._link: Union[None, "MultiEdge"] = None  # only for terminal nodes
        self._leaf_shape: Union[None, Tuple[int, ...]] = None
        return None

    def __str__(self) -> str:
        string: str
        if self._multiedge is None:
            string = f"<Node[{self._leaf_shape}<-{self._link}]>"
        else:
            string = f"<Node[{self._multiedge}]({self._index})>"
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def register_multiedge(self, multiedge: "MultiEdge", index: int) -> None:
        assert isinstance(index, int)
        # assert isinstance(multiedge, MultiEdge)
        self._multiedge = multiedge
        self._index = index
        return None

    @property
    def multiedge(self) -> Union[None, "MultiEdge"]:
        return self._multiedge

    @property
    def index(self) -> Union[None, int]:
        return self._index

    def link(self, edge: "MultiEdge", shape: Tuple[int, ...]) -> None:
        self._link = edge
        self._leaf_shape = shape
        return None

    @property
    def childs(self) -> set["Node"]:
        childs: set["Node"] = set()
        if self._multiedge is not None:
            childs = set(self._multiedge.filtered_targets)
        return childs

    def collect_dependencies(self) -> set["Node"]:
        """
        This function collects set of all present node dependencies (nodes)
        """
        dependencies: set["Node"] = set()
        queue: list["Node"] = [self]
        while len(queue) > 0:
            N: "Node" = queue.pop(0)
            dependencies.add(N)
            if N.multiedge is not None:
                queue.extend(N.multiedge.filtered_targets)
        return dependencies


class MultiEdge:
    def __init__(self, grad_fn: AutogradFunction) -> None:
        # core attributes
        self._gfn: AutogradFunction = grad_fn
        self._xfn: ExtendedAutogradFunction
        self._source_index: dict[int, "Node"] = dict()
        self._target_index: dict[int, Union[None, "Node"]] = dict()
        # util attributes
        self._sources: Tuple[Union[None, "Node"], ...]
        self._targets: Tuple[Union[None, "Node"], ...]
        # only  for representation
        self._id: int = random.randint(0, 9999)
        return None

    def __str__(self) -> str:
        name: str = f"{self._gfn!r}".split(" ")[0].replace("<", "")
        return f"{name}|{f"{self._id:04d}"}"

    def __repr__(self) -> str:
        return self.__str__()

    def register_source(self, output_idx: int, node: "Node") -> None:
        # input integer here represents the numeration within outputs
        assert isinstance(output_idx, int)
        assert isinstance(node, Node)
        self._source_index[output_idx] = node
        self._collect_sources()
        return None

    def register_target(self, input_idx: int, node: Union[None, "Node"]) -> None:
        # input integer here represents the numeration within inputs
        assert isinstance(input_idx, int)
        assert isinstance(node, (type(None), Node))
        self._target_index[input_idx] = node
        self._collect_targets()
        return None

    def transcode(
        self,
        transcoder: FunctionTranscoder,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assert self._gfn is not None
        gfn_type: Type[AutogradFunction] = type(self._gfn)
        xfn_class: Type[ExtendedAutogradFunction] = transcoder.map(gfn_type=gfn_type)
        self._xfn = xfn_class(
            grad_fn=self._gfn,
            order=order,
            dtype=dtype,
            device=device,
        )
        return None

    @property
    def gfn(self) -> AutogradFunction:
        return self._gfn

    @property
    def xfn(self) -> ExtendedAutogradFunction:
        return self._xfn

    def _collect_sources(self) -> None:
        key_range: Tuple[int, ...] = tuple(range(max(self._source_index) + 1))
        sources: list["Node"] = list()
        for i in (i for i in key_range if i in self._source_index):
            sources.append(self._source_index[i])
        self._sources = tuple(sources)
        return None

    @property
    def sources(self) -> Tuple[Union[None, "Node"], ...]:
        assert self._sources is not None
        return self._sources

    @property
    def filtered_sources(self) -> Tuple["Node", ...]:
        # only filters when there are multi-output operators with heterogeneus
        # requires_grad in outputs
        assert self._sources is not None
        return tuple(N for N in self._sources if N is not None)

    def _collect_targets(self) -> None:
        key_range: Tuple[int, ...] = tuple(range(max(self._target_index) + 1))
        targets: list["Node"] = list()
        for i in self._target_index:
            targets.append(self._target_index.get(i, None))
        self._targets = tuple(targets)
        return None

    @property
    def targets(self) -> Tuple[Union[None, "Node"], ...]:
        assert self._targets is not None
        return self._targets

    @property
    def filtered_targets(self) -> Tuple["Node", ...]:
        assert self._targets is not None
        return tuple(N for N in self._targets if N is not None)
