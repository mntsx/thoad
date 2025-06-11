# Standard Library Dependencies
import random
from typing import Tuple, Union, Type

# PyTorch dependencies
import torch

# Internal dependencies
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction
from thoad.autodifferentiation.initialization.assignment import FunctionTranscoder


class Node:
    def __init__(self) -> None:
        self._multiedge: Union[None, "MultiEdge"] = None
        self._index: Union[None, int] = None
        # only for representation
        self._link: Union[None, "MultiEdge"] = None  # only for terminal nodes
        self._leaf_size: Union[None, Tuple[int, ...]] = None
        return None

    def __str__(self) -> str:
        string: str
        if self._multiedge is None:
            string = f"<Node[{self._leaf_size}<-{self._link}]>"
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

    def link(self, edge: "MultiEdge", size: torch.Size) -> None:
        self._link = edge
        self._leaf_size = size
        return None

    @property
    def childs(self) -> set["Node"]:
        childs: set["Node"] = set()
        if self._multiedge is not None:
            childs = set(self._multiedge.targets)
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
                queue.extend(N.multiedge.targets)
        return dependencies


class MultiEdge:
    def __init__(self, grad_fn: torch.autograd.Function) -> None:
        self._gfn: Union[None, torch.autograd.Function] = grad_fn
        self._xfn: Union[None, ExtendedAutogradFunction] = None
        self._sources: dict[int, "Node"] = dict()
        self._targets: dict[int, "Node"] = dict()
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
        self._sources[output_idx] = node
        return None

    def register_target(self, input_idx: int, node: "Node") -> None:
        # input integer here represents the numeration within inputs
        assert isinstance(input_idx, int)
        assert isinstance(node, Node)
        self._targets[input_idx] = node
        return None

    def transcode(
        self,
        transcoder: FunctionTranscoder,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        xfn_class: Type[ExtendedAutogradFunction] = transcoder.map(grad_fn=self._gfn)
        self._xfn = xfn_class(
            grad_fn=self._gfn,
            order=order,
            dtype=dtype,
            device=device,
        )
        return transcoder

    @property
    def gfn(self) -> torch.autograd.Function:
        return self._gfn

    @property
    def xfn(self) -> "ExtendedAutogradFunction":
        return self._xfn

    @property
    def sources(self) -> Tuple["Node"]:
        key_range: Tuple[int, ...] = range(len(self._sources))
        assert set(self._sources.keys()) == set(key_range)
        sources: list["Node"] = list()
        for i in key_range:
            sources.append(self._sources[i])
        return tuple(sources)

    @property
    def targets(self) -> Tuple["Node"]:
        # not when there are nodes with require_grad=False
        # key_range: Tuple[int, ...] = range(len(self._targets))
        # assert set(self._targets.keys()) == set(key_range)
        targets: list["Node"] = list()
        for i in self._targets.keys():
            targets.append(self._targets[i])
        return tuple(targets)
