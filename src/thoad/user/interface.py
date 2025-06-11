# Standard Library Dependencies
from typing import Iterable, Optional, Sequence, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.control.propagation import Controller
from thoad.autodifferentiation.internals.base import ExtendedAutogradFunction
from thoad.graph.graph import Graph
from thoad.typing.data import Indep, Shape
from thoad.typing.user import Hook
from thoad.user.display import display_tensor_subgraph


class Operator:
    def __init__(self, tensor: Tensor) -> None:
        # tensor checks
        self._tensor_checks(tensor=tensor)
        # control
        self._controller = Controller()
        self._controller.setup_graph(tensor=tensor)
        # data
        self._tensor: Tensor = tensor

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def _tensor_checks(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise ValueError(
                f"tensor must be a Tensor, but got type {type(tensor).__name__}"
            )
        if not tensor.requires_grad:
            raise ValueError("tensor tensor does not require grad")
        if tensor.grad_fn is None:
            raise ValueError("tensor tensor does not have grad_fn")
        return None

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self._tensor_checks(tensor=tensor)
        self._tensor = tensor
        self._graph = Graph(tensor=self._tensor)
        return None

    @property
    def compatible(self) -> bool:
        return self._controller.graph.compatible

    @property
    def index(
        self,
    ) -> dict[
        Type[torch.autograd.Function],
        Type[ExtendedAutogradFunction],
    ]:
        return self._controller.graph.transcoder.index

    @index.setter
    def index(
        self,
        index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]],
    ) -> None:
        if not isinstance(index, dict):
            raise ValueError(
                f"index must be a dict, but got type {type(index).__name__}"
            )
        for v in index.values():
            if not issubclass(v, ExtendedAutogradFunction):
                raise ValueError(
                    f"All values in index must be ExtendedAutogradFunction, "
                    f"but got type {type(v).__name__}"
                )
        self._controller.graph.transcoder.index = index
        return None

    def _backward_checks(
        self,
        order: int,
        crossings: Optional[bool] = False,
        groups: Optional[Iterable[Iterable[Tensor]]] = None,
        batch: Optional[bool] = False,
    ) -> None:

        if not isinstance(crossings, bool):
            raise TypeError(
                f"crossings must be a bool, but got type {type(crossings).__name__}"
            )
        if not isinstance(order, int):
            raise TypeError(
                f"order must be an int, but got type {type(order).__name__}"
            )
        if not order > 0:
            raise ValueError(f"order must be positive integer, but got {order}")
        if groups is not None:
            if crossings:
                raise ValueError(
                    "groups and crossings are mutually exclusive "
                    f"(received crossings={crossings!r}, groups={groups!r})"
                )
            if not isinstance(groups, Iterable):
                raise TypeError(
                    f"groups must be an Iterable, but got type "
                    f"{type(groups).__name__}"
                )
            for G in groups:
                if not isinstance(G, Sequence):
                    raise TypeError(
                        f"Each group must be a Sequence, but got type "
                        f"{type(G).__name__}"
                    )
                for T in G:
                    if not isinstance(T, Tensor):
                        raise TypeError(
                            f"All elements in groups must be Tensor, but got "
                            f"type {type(T).__name__}"
                        )
        if not isinstance(batch, bool):
            raise ValueError(
                f"batch must be a bool, but got type {type(batch).__name__}"
            )

        return None

    def backward(
        self,
        order: int,
        crossings: Optional[bool] = False,
        groups: Optional[Iterable[Iterable[Tensor]]] = None,
        batch: Optional[bool] = False,
    ) -> None:
        # checks
        self._backward_checks(
            order=order,
            crossings=crossings,
            groups=groups,
            batch=batch,
        )
        # backprop
        self._controller.keepbatch = batch
        self._controller.cross_terminals = crossings
        self._controller.graph.transcode_fns(
            order=order,
            dtype=self._tensor.dtype,
            device=self._tensor.device,
        )
        groups = list() if groups is None else list(groups)
        self._controller.propagate(order=order, groups=groups)

        return None

    def display_graph(self) -> None:
        display_tensor_subgraph(
            tensor=self._tensor,
            supports=self._controller.graph.transcoder.supports,
        )
        return None

    def _check_variables(self, variables: Sequence[Tensor]) -> None:
        if not isinstance(variables, Sequence):
            raise ValueError(
                f"variables must be a sequence, not {type(variables).__name__!r}"
            )
        for T in variables:
            if not isinstance(T, Tensor):
                raise ValueError(
                    "each element in variables must be a Tensor, "
                    f"but got {type(T).__name__!r}"
                )

    def register_backward_hook(
        self,
        variables: Sequence[Tensor],
        hook: Hook,
    ) -> None:
        self._check_variables(variables=variables)
        self._controller.add_backward_hook(key=variables, hook=hook)
        return None

    def require_grad_(self, variables: Sequence[Tensor]) -> None:
        self._check_variables(variables=variables)
        self._controller.add_gradient_retention(key=variables)
        return None

    def fetch_hgrad(
        self, variables: Sequence[Tensor], batch: Optional[bool] = False
    ) -> Tuple[Tensor, Tuple[Indep, ...], Tuple[Shape, ...]]:
        self._check_variables(variables=variables)
        data: Tuple[
            Tensor,
            Tuple[Tuple[int, ...]],
            Tuple[Tuple[Union[None, int], ...]],
        ]
        data = self._controller.fetch_hgrad(key=variables, keepbatch=batch)
        return data


def backward(
    tensor: Tensor,
    order: int,
    crossings: Optional[bool] = False,
    groups: Optional[Iterable[Iterable[Tensor]]] = None,
    batch: Optional[bool] = False,
) -> Operator:
    operator: Operator = Operator(tensor=tensor)
    operator.backward(order=order, crossings=crossings, groups=groups, batch=batch)
    return operator
