# Standard Library Dependencies
from typing import Type

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import (
    ExtendedAutogradFunction,
)

from thoad.typing.data import EDData
from thoad.user.interface import Operator
from tests.graph.utils import (
    acquire_test_gfn_map,
)


def test_graph_01() -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=1)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())

    return None


def test_graph_02() -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=1)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())
    hgrad = operator.fetch_hgrad(variables=(T1,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T1.hgrad[0].flatten())

    return None


def test_graph_03() -> None:

    ### TEST:
    # 1. joint graph
    # 2. ordern 1

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=1)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())

    return None


def test_graph_04() -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())
    hgrad = operator.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[1].flatten())

    return None


def test_graph_05() -> None:

    ### TEST:
    # 1. tree graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3)

    ### Checks
    hgrad: EDData
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad = operator.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad[0], Tensor)
        if Ti is Tj:
            hgrad = operator.fetch_hgrad(variables=(Ti, Tj))
            assert isinstance(hgrad[0], Tensor)
        else:
            try:
                hgrad = operator.fetch_hgrad(variables=(Ti, Tj))
                raise AssertionError()
            except KeyError:
                pass

    return None


def test_graph_06() -> None:

    ### TEST:
    # 1. joint graph
    # 2. order 3
    # 3. no terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())
    hgrad = operator.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[1].flatten())

    return None


def test_graph_07() -> None:

    ### TEST:
    # 1. linear graph
    # 2. order 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T1)
    GO: Tensor = torch.relu(T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3, crossings=True)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())
    hgrad = operator.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[1].flatten())

    return None


def test_graph_08() -> None:

    ### TEST:
    # 1. tree graph
    # 2. ordern 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    GO: Tensor = torch.mm(T2, T3)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3, crossings=True)

    ### Checks
    hgrad: EDData
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad = operator.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad[0], Tensor)
        hgrad = operator.fetch_hgrad(variables=(Ti, Tj))
        assert isinstance(hgrad[0], Tensor)

    return None


def test_graph_09() -> None:

    ### TEST:
    # 1. joint graph
    # 2. ordern 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.relu(T0)
    T2: Tensor = torch.relu(T0)
    GO: Tensor = torch.mm(T1, T2)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3, crossings=True)

    ### Checks
    hgrad: EDData
    hgrad = operator.fetch_hgrad(variables=(T0,))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[0].flatten())
    hgrad = operator.fetch_hgrad(variables=(T0, T0))
    assert isinstance(hgrad[0], Tensor)
    assert torch.allclose(hgrad[0].flatten(), T0.hgrad[1].flatten())

    return None


def test_graph_10() -> None:

    ### TEST:
    # 1. tree graph
    # 2. ordern 3
    # 3. terminal crossings

    ### Create graph
    T00: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T01: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T02: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T10: Tensor = torch.relu(T00)
    T11: Tensor = torch.relu(T01)
    T12: Tensor = torch.relu(T02)
    GO: Tensor = torch.addmm(input=T10, mat1=T11, mat2=T12)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3, crossings=False, groups=[[T01, T02]])

    ### Checks
    hgrad: EDData
    for Ti, Tj in [[T00, T02], [T00, T01]]:
        try:
            hgrad = operator.fetch_hgrad(variables=(Ti, Tj))
            raise AssertionError()
        except KeyError:
            pass
    hgrad = operator.fetch_hgrad(variables=(T01, T02))
    assert isinstance(hgrad[0], Tensor)

    return None


def test_graph_11() -> None:

    ### TEST:
    # 1. asymetric tree graph
    # 2. ordern 3
    # 3. terminal crossings

    ### Create graph
    T0: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T1: Tensor = torch.rand(size=(10, 10), requires_grad=True)
    T2: Tensor = torch.relu(T0)
    T3: Tensor = torch.relu(T1)
    T4: Tensor = torch.relu(T3)
    GO: Tensor = torch.mm(T2, T4)

    ### Configure operator and run backward
    operator: Operator = Operator(tensor=GO)
    assert operator.compatible
    test_func_index: dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]
    test_func_index = acquire_test_gfn_map()
    operator.index = test_func_index
    operator.backward(order=3, crossings=True)

    ### Checks
    hgrad: EDData
    for Ti, Tj in [(T0, T0), (T0, T1), (T1, T0), (T1, T1)]:
        hgrad = operator.fetch_hgrad(variables=(Ti,))
        assert isinstance(hgrad[0], Tensor)
        hgrad = operator.fetch_hgrad(variables=(Ti, Tj))
        assert isinstance(hgrad[0], Tensor)

    return None
