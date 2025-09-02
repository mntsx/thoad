# Standard Library Dependencies
import pytest
from typing import Callable, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import (
    ExtendedAutogradFunction,
)
from thoad.typing import AutogradFunction, EDData
from thoad.user.interface import Controller
from tests.graph.utils import (
    TestUnivariableXBackward1,
    acquire_test0_gfn_map,
    acquire_test1_gfn_map,
)


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=[acquire_test0_gfn_map, acquire_test1_gfn_map],
                ids=["contractive", "direct"])
def gfn_map(request) -> dict[Type, Type]:
    return request.param


def hook_model(
    grad_data: EDData,
    context: dict[AutogradFunction, set[Tensor]],
) -> EDData:
    # check grad_data
    assert isinstance(grad_data, Tuple)
    assert len(grad_data) == 4
    assert isinstance(grad_data[0], Tensor)
    assert isinstance(grad_data[1], Tuple)
    assert all(isinstance(shape, Tuple) for shape in grad_data[1])
    assert all(isinstance(d, int) for shape in grad_data[1] for d in shape)
    assert isinstance(grad_data[2], Tuple)
    assert all(isinstance(indep, Tuple) for indep in grad_data[2])
    Nint: Tuple[Type, Type] = (type(None), int)
    assert all(isinstance(d, Nint) for indep in grad_data[2] for d in indep)
    assert isinstance(grad_data[3], Tuple)
    assert all(isinstance(i, int) for i in grad_data[3])
    # check context
    assert isinstance(context, dict)
    assert all(isinstance(tensors, set) for tensors in context.values())
    assert all(isinstance(T, Tensor) for Ts in context.values() for T in Ts)
    # return untouched grad data
    return grad_data

def test_hooks_01(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. single variable hook
    # 2. linear graph
    # 3. order 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T0, T0), hook=_hook)
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 1, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_02(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. single variable hook
    # 2. linear graph
    # 3. order 3
    # 4. unreachable variable combination

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Register hook and run backward
    try:
        controller.register_backward_hook(variables=(T0, T1, T0), hook=hook_model)
        assert False
    except ValueError:
        pass

    return None

def test_hooks_03(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. multiple variable hook
    # 2. tree graph
    # 3. order 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T1, T0), hook=_hook)
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 1, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_04(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. multiple variable hook
    # 2. tree graph
    # 3. order 3
    # 4. double reachable hook

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T1, T0), hook=_hook)
    controller.register_backward_hook(variables=(T2, T3, T2), hook=_hook)
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 2, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_05(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. multiple variable hook
    # 2. tree graph
    # 3. order 3
    # 4. double reachable hook (with repeated variables)

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T1, T0), hook=_hook)
    controller.register_backward_hook(variables=(T1, T0, T0), hook=_hook)
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 2, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_06(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. multiple variable hook
    # 2. tree graph
    # 3. order 3
    # 4. double reachable hook

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T3, T0), hook=hook_model)
    try:
        controller.register_backward_hook(variables=(T1, T2, T1), hook=hook_model)
        assert False
    except ValueError:
        pass

    return None

def test_hooks_07(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. multiple variable hook
    # 2. tree graph
    # 3. order 3
    # 4. double reachable hook (with repeated variables)

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    T4: Tensor = torch.relu(T0)
    T5: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T4, T5)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T5, T0), hook=_hook)
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 1, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_08(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. (joint) single variable hook
    # 2. joint graph
    # 3. order 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T0, T0), hook=_hook)
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 1, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None

def test_hooks_09() -> None:

    ### TEST:
    # 1. (joint) single variable hook
    # 2. mix of contractive and direct functions (in joint)
    # 3. joint graph
    # 4. order 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.sigmoid(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test0_gfn_map()
    grad_fn: Union[None, AutogradFunction] = torch.sigmoid(T0).grad_fn
    assert grad_fn is not None
    test_func_index[type(grad_fn)] = TestUnivariableXBackward1
    controller.index = test_func_index

    ### Define hook
    hook_calls: list[int] = [0]

    def _hook(
        grad_data: EDData,
        context: dict[AutogradFunction, set[Tensor]],
    ) -> EDData:
        # call hook model
        _grad_data: EDData = hook_model(grad_data=grad_data, context=context)
        # account call
        hook_calls[0] += 1
        # return untouched grad data
        return _grad_data

    ### Register hook and run backward
    controller.register_backward_hook(variables=(T0, T0, T0), hook=_hook)
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    assert hook_calls[0] == 1, hook_calls[0]

    ### Remove hgrad argument from tensors
    controller.clear()

    return None