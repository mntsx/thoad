# Standard Library Dependencies
from typing import Iterable, Sequence, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import thoad.config as config
from thoad.differentiation.engine.control.propagation import BackpropOrchestrator
from thoad.differentiation.internals.base import ExtendedAutogradFunction
from thoad.graph.graph import Graph
from thoad.typing import Indep, Shape
from thoad.typing.functions import Hook
from thoad.user.display import display_tensor_subgraph
from thoad.typing import AutogradFunction, PopulatedEDData, VPerm


class Controller:
    def __init__(self, tensor: Tensor) -> None:
        # tensor checks
        self._tensor_checks(tensor=tensor)
        # control
        self._orchestrator = BackpropOrchestrator()
        self._orchestrator.setup_graph(tensor=tensor)
        # data
        self._tensor: Tensor = tensor

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def _tensor_checks(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise ValueError(
                f"arguemnt tensor expects Tensor, but got type {type(tensor).__name__}"
            )
        if not tensor.requires_grad:
            raise ValueError(f"received tensor has require_grad=False")
        if tensor.grad_fn is None:
            raise ValueError(f"received tensor does not have grad_fn")
        return None

    @tensor.setter
    def tensor(self, tensor: Tensor) -> None:
        self._tensor_checks(tensor=tensor)
        self._tensor = tensor
        self._graph = Graph(tensor=self._tensor)
        return None

    @property
    def index(
        self,
    ) -> dict[
        Type[AutogradFunction],
        Type[ExtendedAutogradFunction],
    ]:
        return self._orchestrator.graph.transcoder.index

    @index.setter
    def index(
        self,
        index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]],
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
        self._orchestrator.graph.transcoder.index = index
        return None

    @property
    def compatible(self) -> bool:
        return self._orchestrator.graph.compatible

    def display_graph(self) -> None:
        display_tensor_subgraph(
            tensor=self._tensor,
            supports=self._orchestrator.graph.transcoder.supports,
        )
        return None

    def _backward_checks(
        self,
        order: int,
        gradient: Union[None, Tensor] = None,
        crossings: bool = False,
        groups: Union[None, Iterable[Iterable[Tensor]]] = None,
        keep_batch: bool = False,
        keep_schwarz: bool = False,
    ) -> None:

        if not isinstance(order, int):
            raise TypeError(
                f"order must be type int, but got type {type(order).__name__}"
            )
        if not order > 0:
            raise ValueError(f"order must be a positive integer, but got {order}")

        if isinstance(gradient, Tensor) and gradient.shape != self._tensor.shape:
            raise ValueError(
                f"gradient Tensor must have same shape as tensor, "
                f"but got shapes {list(gradient.shape)} and {list(self._tensor.shape)}"
            )
        if not isinstance(gradient, (type(None), Tensor)):
            raise TypeError(
                f"gradient must be type Tensor, but got type {type(gradient).__name__}"
            )

        if not isinstance(crossings, bool):
            raise TypeError(
                f"crossings must be type bool, but got type {type(crossings).__name__}"
            )
        if groups is not None:
            if crossings:
                raise ValueError(
                    "groups and crossings are mutually exclusive "
                    f"(received crossings={crossings!r}, groups={groups!r})"
                )
            if not isinstance(groups, Iterable):
                raise TypeError(
                    f"groups must be type Iterable, but got type "
                    f"{type(groups).__name__}"
                )
            for G in groups:
                if not isinstance(G, Iterable):
                    raise TypeError(
                        f"All groups must be type Iterable, but got type "
                        f"{type(G).__name__}"
                    )
                for T in G:
                    if not isinstance(T, Tensor):
                        raise TypeError(
                            f"All elements in groups must be type Tensor, but got "
                            f"type {type(T).__name__}"
                        )
        if not isinstance(keep_batch, bool):
            raise ValueError(
                "keep_batch must be type bool, but got type "
                f"{type(keep_batch).__name__}"
            )
        if not isinstance(keep_schwarz, bool):
            raise ValueError(
                "keep_schwarz must be type bool, but got type "
                f"{type(keep_schwarz).__name__}"
            )

        return None

    def backward(
        self,
        order: int,
        gradient: Union[None, Tensor] = None,
        crossings: bool = False,
        groups: Union[None, Iterable[Iterable[Tensor]]] = None,
        keep_batch: bool = False,
        keep_schwarz: bool = False,
    ) -> None:
        # checks
        self._backward_checks(
            order=order,
            gradient=gradient,
            crossings=crossings,
            groups=groups,
            keep_batch=keep_batch,
            keep_schwarz=keep_schwarz,
        )
        # backprop
        self._orchestrator.keep_batch = keep_batch
        self._orchestrator.keep_schwarz = keep_schwarz
        self._orchestrator.cross_terminals = crossings
        self._orchestrator.graph.transcode_fns(
            order=order,
            dtype=self._tensor.dtype,
            device=self._tensor.device,
        )
        groups = list() if groups is None else list(groups)
        self._orchestrator.propagate(order=order, groups=groups, gradient=gradient)

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

    def _check_hook(self, hook: Hook) -> None:
        code = hook.__code__
        if set(code.co_varnames[: code.co_argcount]) != {"grad_data", "context"}:
            raise ValueError(
                "hook must be a Callable expecting arguments:"
                "\n    grad_data: "
                "Tuple[torch.Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]"
                "\n    context: dict[AutogradFunction, set[Tensor]]"
            )
        return None

    def register_backward_hook(
        self,
        variables: Sequence[Tensor],
        hook: Hook,
    ) -> None:
        if bool(getattr(config, "DEBUG", False)):
            self._check_variables(variables=variables)
            self._check_hook(hook=hook)
        self._orchestrator.add_backward_hook(key=tuple(variables), hook=hook)
        return None

    def require_grad_(self, variables: Sequence[Tensor]) -> None:
        self._check_variables(variables=variables)
        self._orchestrator.add_gradient_retention(key=tuple(variables))
        return None

    def fetch_hgrad(
        self,
        variables: Sequence[Tensor],
        keep_batch: bool = False,
        keep_schwarz: bool = False,
    ) -> Tuple[Tensor, Tuple[Tuple[Shape, ...], Tuple[Indep, ...], VPerm]]:
        if bool(getattr(config, "DEBUG", False)):
            self._check_variables(variables=variables)
        data: PopulatedEDData = self._orchestrator.fetch_hgrad(
            key=tuple(variables),
            keep_batch=keep_batch,
            keep_schwarz=keep_schwarz,
        )
        return (data[0], (data[1], data[2], data[3]))

    def clear(self) -> None:
        self._orchestrator.clear()
        return None


def backward(
    tensor: Tensor,
    order: int,
    gradient: Union[None, Tensor] = None,
    crossings: bool = False,
    groups: Union[None, Iterable[Iterable[Tensor]]] = None,
    keep_batch: bool = False,
    keep_schwarz: bool = False,
) -> Controller:
    controller: Controller = Controller(tensor=tensor)
    controller.backward(
        order=order,
        gradient=gradient,
        crossings=crossings,
        groups=groups,
        keep_batch=keep_batch,
        keep_schwarz=keep_schwarz,
    )
    return controller
