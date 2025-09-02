# Standard Library Dependencies
import pytest
from typing import Callable, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.base import (
    ExtendedAutogradFunction,
)
from thoad.typing import AutogradFunction
from thoad.user.interface import Controller
from tests.graph.utils import (
    TestUnivariableXBackward1,
    acquire_test0_gfn_map,
    acquire_test1_gfn_map,
)


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(
    params=[acquire_test0_gfn_map, acquire_test1_gfn_map], ids=["contractive", "direct"]
)
def gfn_map(request) -> dict[Type, Type]:
    """Yields the function index dict for each variation."""
    return request.param


def test_graph_01(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    # order 1
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    # order 2
    try:
        controller.fetch_hgrad(variables=(T0, T0))
        assert False
    except KeyError:
        pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_02(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    hgrad, _ = controller.fetch_hgrad(variables=(T1,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T1.hgrad[0].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_03(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. joint graph
    # 2. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_04(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    # order 1
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    # order 2
    hgrad, _ = controller.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[1].flatten())
    # order 3
    hgrad, _ = controller.fetch_hgrad(variables=(T0, T0, T0))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[2].flatten())
    # order 4
    try:
        controller.fetch_hgrad(variables=(T0, T0, T0, T0))
        assert False
    except KeyError:
        pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_05(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad, _ = controller.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad, Tensor)
        if Ti is Tj:
            hgrad, _ = controller.fetch_hgrad(variables=(Ti, Tj))
            assert isinstance(hgrad, Tensor)
        else:
            try:
                controller.fetch_hgrad(variables=(Ti, Tj))
                raise AssertionError()
            except KeyError:
                pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_06(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. joint graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    hgrad, _ = controller.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[1].flatten())

    return None


def test_graph_07(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    hgrad, _ = controller.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[1].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_08(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    hgrad: Tensor
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad, _ = controller.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad, Tensor)
        hgrad, _ = controller.fetch_hgrad(variables=(Ti, Tj))
        assert isinstance(hgrad, Tensor)

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_09(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. joint graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    hgrad, _ = controller.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[1].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_10(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T00: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T01: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T02: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T10: Tensor = torch.relu(T00)
    T11: Tensor = torch.relu(T01)
    T12: Tensor = torch.relu(T02)
    GO: Tensor = torch.addmm(input=T10, mat1=T11, mat2=T12)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=False, groups=[[T01, T02]], keep_batch=False)

    ### Checks
    hgrad: Tensor
    for Ti, Tj in [[T00, T02], [T00, T01]]:
        try:
            controller.fetch_hgrad(variables=(Ti, Tj))
            raise AssertionError()
        except KeyError:
            pass
    hgrad, _ = controller.fetch_hgrad(variables=(T01, T02))
    assert isinstance(hgrad, Tensor)

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_11(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. asymetric tree graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    T4: Tensor = torch.relu(T3)
    GO: Tensor = torch.mm(T2, T4)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    hgrad: Tensor
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad, _ = controller.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad, Tensor)
        hgrad, _ = controller.fetch_hgrad(variables=(Ti, Tj))
        assert isinstance(hgrad, Tensor)

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_12() -> None:

    ### TEST:
    # 1. mix of contractive and direct functions
    # 2. linear graph
    # 3. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.sigmoid(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test0_gfn_map()
    grad_fn: Union[None, AutogradFunction] = torch.sigmoid(T0).grad_fn
    assert grad_fn is not None
    test_func_index[type(grad_fn)] = TestUnivariableXBackward1
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    # order 1
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    # order 2
    try:
        controller.fetch_hgrad(variables=(T0, T0))
        assert False
    except KeyError:
        pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_13() -> None:

    ### TEST:
    # 1. mix of contractive and direct functions
    # 2. tree graph
    # 3. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.sigmoid(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test0_gfn_map()
    grad_fn: Union[None, AutogradFunction] = torch.sigmoid(T0).grad_fn
    assert grad_fn is not None
    test_func_index[type(grad_fn)] = TestUnivariableXBackward1
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())
    hgrad, _ = controller.fetch_hgrad(variables=(T1,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T1.hgrad[0].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_14() -> None:

    ### TEST:
    # 1. mix of contractive and direct functions
    # 2. joint graph
    # 3. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.sigmoid(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test0_gfn_map()
    grad_fn: Union[None, AutogradFunction] = torch.sigmoid(T0).grad_fn
    assert grad_fn is not None
    test_func_index[type(grad_fn)] = TestUnivariableXBackward1
    controller.index = test_func_index
    controller.backward(order=1, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    hgrad, _ = controller.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad, Tensor)
    assert torch.allclose(hgrad.flatten(), T0.hgrad[0].flatten())

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_15(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. input with deactivated grad
    # 2. asymetric tree graph
    # 3. order 3

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=False, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    T4: Tensor = torch.relu(T3)
    GO: Tensor = torch.mm(T2, T4)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=False, keep_batch=False)

    ### Checks
    hgrad: Tensor
    for k, (Ti, Tj) in enumerate([(T0, T0), (T0, T1), (T1, T0), (T1, T1)]):
        i: int = k // 2
        j: int = k % 2
        if i == 0:
            hgrad, _ = controller.fetch_hgrad(variables=(Ti,))
            assert isinstance(hgrad, Tensor)
        else:
            try:
                controller.fetch_hgrad(variables=(Ti,))
                assert False
            except ValueError:
                pass
        if 1 not in (i, j):
            hgrad, _ = controller.fetch_hgrad(variables=(Ti, Tj))
            assert isinstance(hgrad, Tensor)
        else:
            try:
                controller.fetch_hgrad(variables=(Ti, Tj))
                assert False
            except ValueError:
                pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None


def test_graph_16(gfn_map: Callable[[], dict[Type, Type]]) -> None:

    ### TEST:
    # 1. input with deactivated grad
    # 2. asymetric tree graph
    # 3. order 3
    # 4. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=False, device=device)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    T4: Tensor = torch.relu(T3)
    GO: Tensor = torch.mm(T2, T4)

    ### Configure controller and run backward
    controller: Controller = Controller(tensor=GO)
    assert controller.compatible
    test_func_index: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
    test_func_index = gfn_map()
    controller.index = test_func_index
    controller.backward(order=3, crossings=True, keep_batch=False)

    ### Checks
    hgrad: Tensor
    for k, (Ti, Tj) in enumerate([(T0, T0), (T0, T1), (T1, T0), (T1, T1)]):
        i: int = k // 2
        j: int = k % 2
        if i == 0:
            hgrad, _ = controller.fetch_hgrad(variables=(Ti,))
            assert isinstance(hgrad, Tensor)
        else:
            try:
                controller.fetch_hgrad(variables=(Ti,))
                assert False
            except ValueError:
                pass
        if 1 not in (i, j):
            hgrad, _ = controller.fetch_hgrad(variables=(Ti, Tj))
            assert isinstance(hgrad, Tensor)
        else:
            try:
                controller.fetch_hgrad(variables=(Ti, Tj))
                assert False
            except ValueError:
                pass

    ### Remove hgrad argument from tensors
    controller.clear()

    return None
