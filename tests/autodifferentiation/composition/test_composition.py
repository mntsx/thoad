# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.backprop.composition.composition import Loader
from thoad.typing.data import Notation


def _calculate_test_independent_sizes(
    external_shapes: dict[int, Tuple[Union[None, int], ...]],
    external_indeps: dict[int, Tuple[Union[None, int], ...]],
) -> list[int]:
    independent_sizes: list[int] = list()
    for row in zip(*external_indeps.values()):
        candidates: list[int] = list()
        for i, dim in enumerate(row):
            candidates.append(1 if dim is None else external_shapes[dim][i])
        assert len(set(row)) == 1
        independent_sizes.append(set(candidates).pop())
    return independent_sizes


def _calculate_test_distributed_shape(
    key: Tuple[int, ...],
    external_shapes: dict[int, Tuple[int, ...]],
    external_indeps: dict[int, Tuple[Union[None, int], ...]],
) -> list[list[int]]:
    shapes: list[Tuple[int, ...]] = [external_shapes[v] for v in key]
    indeps: list[Tuple[int, ...]] = [external_indeps[v] for v in key]
    distributed_shapes: list[list[int]] = [list() for _ in shapes]
    for (
        i,
        (shape, indep),
    ) in enumerate(zip(shapes, indeps)):
        for j, sz in enumerate(shape):
            if j not in indep:
                distributed_shapes[i].append(sz)
    return distributed_shapes


def _generate_test_external_differentials(
    GOnumel: int,
    external_keys: list[Tuple[int, ...]],
    external_shapes: dict[int, Tuple[int, ...]],
    external_indeps: dict[int, Tuple[Union[None, int], ...]],
) -> dict[Tuple[int, ...], Tensor]:
    differential_shapes: dict[Tuple[int, ...], list[int]] = dict()
    independent_sizes: Tuple[int, ...] = _calculate_test_independent_sizes(
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    for key in external_keys:
        distributed_shapes: Tuple[int, ...] = _calculate_test_distributed_shape(
            key=key,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
        )
        differential_shape: list[int] = [GOnumel, *independent_sizes]
        for distributed_shape in distributed_shapes:
            differential_shape.extend(distributed_shape)
        differential_shapes[key] = torch.rand(size=tuple(differential_shape))
    return differential_shapes


### LINEAR GRAPH (o -> x_1 -> x_2) (NOT DIAGONAL INTERNALS)


def test_01a() -> None:
    # 01. [no independent dims, no batch, not diagonal]
    # variables: Tuple[int, int, Tuple[int, ...]] = (1, 1, (0, 0, 0))
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1: int = 4
    D2: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    # define internal differentials
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1, D2))  # a,aA->A
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))  # a,aAB->AB
    internal_differentials[(0, (0, 0, 0))] = torch.rand(
        size=(D1, D2, D2, D2)
    )  # a,aABC->ABC
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[0, (0, 0)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[0, (0, 0, 0)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: dict[int, Tuple[Tuple[int, ...]]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2, D2, D2)
    assert tuple(shapes.values()) == ((D2,),), (shapes, ((D2,)))
    assert tuple(indeps.values()) == ((None,),), indeps
    return None


def test_01b() -> None:
    # 01. [no independent dims, no batch, not diagonal] (but with permutations)
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D11: int = 4
    D12: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D11, D12)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    # define internal differentials
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(*(D12, D11), D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(*(D12, D11), D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(*(D12, D11), D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1, 0, 2]], [[2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1, 0, 2, 3]], [[2], [3]]]
    einstein_notations[0, (0, 0, 0)] = [[[0, 1], [1, 0, 2, 3, 4]], [[2], [3], [4]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2, D2, D2)
    assert tuple(shapes.values()) == ((D2,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_01c() -> None:
    # 01. [no independent dims, no batch, not diagonal] (but with permutations)
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D11: int = 4
    D12: int = 5
    D21: int = 6
    D22: int = 7
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D11, D12)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(*(D12, D11), *(D22, D21)))
    internal_differentials[(0, (0, 0))] = torch.rand(
        size=(*(D12, D11), *(D22, D21), *(D22, D21))
    )
    internal_differentials[(0, (0, 0, 0))] = torch.rand(
        size=(*(D12, D11), *(D22, D21), *(D22, D21), *(D22, D21))
    )
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1, 0, 2, 3]], [[3, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1, 0, 2, 3, 4, 5]], [[3, 2], [5, 4]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1], [1, 0, 2, 3, 4, 5, 6, 7]],
        [[3, 2], [5, 4], [7, 6]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D21, D22, D21, D22, D21, D22)
    assert tuple(shapes.values()) == ((D21, D22),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_01d() -> None:
    # 01. [no independent dims, no batch, not diagonal] (but with permutations)
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D11: int = 5
    D12: int = 6
    D2: int = 7
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B, D11, D12)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(*(D12, D11), D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(*(D12, D11), D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(*(D12, D11), D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1, 2], [2, 1, 3]], [[0, 3]]]
    einstein_notations[0, (0, 0)] = [[[0, 1, 2], [2, 1, 3, 4]], [[0, 3], [0, 4]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1, 2], [2, 1, 3, 4, 5]],
        [[0, 3], [0, 4], [0, 5]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_02() -> None:
    # 02. [pre-independent dims, no batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1, 2]], [[0, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1, 2, 3]], [[0, 2], [0, 3]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1], [1, 2, 3, 4]],
        [[0, 2], [0, 3], [0, 4]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, D2, D2, D2)
    assert tuple(shapes.values()) == ((I, D2),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_03() -> None:
    # 03. [full-independent dims, no batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(I, D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(I, D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(I, D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [0, 1, 2]], [[0, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [0, 1, 2, 3]], [[0, 2], [0, 3]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1], [0, 1, 2, 3, 4]],
        [[0, 2], [0, 3], [0, 4]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, D2, D2, D2)
    assert tuple(shapes.values()) == ((I, D2),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_04() -> None:
    # 04. [no independent dims, prebatch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1, 2]], [[0, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1, 2, 3]], [[0, 2], [0, 3]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1], [1, 2, 3, 4]],
        [[0, 2], [0, 3], [0, 4]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_05() -> None:
    # 05. [no independent dims, postbatch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(B, D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(B, D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(B, D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[1], [0, 1, 2]], [[0, 2]]]
    einstein_notations[0, (0, 0)] = [[[1], [0, 1, 2, 3]], [[0, 2], [0, 3]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[1], [0, 1, 2, 3, 4]],
        [[0, 2], [0, 3], [0, 4]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_06() -> None:
    # 06. [no independent dims, full batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(B, D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(B, D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(B, D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [0, 1, 2]], [[0, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [0, 1, 2, 3]], [[0, 2], [0, 3]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1], [0, 1, 2, 3, 4]],
        [[0, 2], [0, 3], [0, 4]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_07() -> None:
    # 07. [full-independent dims, full-batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    B: int = 5
    D1: int = 6
    D2: int = 7
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(I, B, D1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(I, B, D1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(I, B, D1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1, 2], [0, 1, 2, 3]], [[0, 1, 3]]]
    einstein_notations[0, (0, 0)] = [
        [[0, 1, 2], [0, 1, 2, 3, 4]],
        [[0, 1, 3], [0, 1, 4]],
    ]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1, 2], [0, 1, 2, 3, 4, 5]],
        [[0, 1, 3], [0, 1, 4], [0, 1, 5]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, B, D2, B, D2, B, D2), tensor.shape
    assert tuple(shapes.values()) == ((I, B, D2),)
    assert tuple(indeps.values()) == ((0,),)
    return None


### LINEAR GRAPH (o -> x_1 -> x_2) (DIAGONAL INTERNALS)


def test_08() -> None:
    # 08. [no independent dims, no batch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1: int = 4
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0], [0]], [[0]]]
    einstein_notations[0, (0, 0)] = [[[0], [0]], [[0], [0]]]
    einstein_notations[0, (0, 0, 0)] = [[[0], [0]], [[0], [0], [0]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D1, D1, D1)
    assert tuple(shapes.values()) == ((D1,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_09() -> None:
    # 09. [pre-independent dims, no batch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1]], [[0, 1]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1]], [[0, 1], [0, 1]]]
    einstein_notations[0, (0, 0, 0)] = [[[0, 1], [1]], [[0, 1], [0, 1], [0, 1]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_10() -> None:
    # 10. [full-independent dims, no batch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(I, D1))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [0, 1]], [[0, 1]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
    einstein_notations[0, (0, 0, 0)] = [[[0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_11() -> None:
    # 11. [no independent dims, prebatch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [1]], [[0, 1]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [1]], [[0, 1], [0, 1]]]
    einstein_notations[0, (0, 0, 0)] = [[[0, 1], [1]], [[0, 1], [0, 1], [0, 1]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_12() -> None:
    # 12. [no independent dims, postbatch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[1], [0, 1]], [[0, 1]]]
    einstein_notations[0, (0, 0)] = [[[1], [0, 1]], [[0, 1], [0, 1]]]
    einstein_notations[0, (0, 0, 0)] = [[[1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_13() -> None:
    # 13. [no independent dims, full batch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1], [0, 1]], [[0, 1]]]
    einstein_notations[0, (0, 0)] = [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
    einstein_notations[0, (0, 0, 0)] = [[[0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_14() -> None:
    # 14. [full-independent dims, full-batch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    B: int = 5
    D1: int = 6
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (I, B, D1)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (0,)
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_tensor: Tensor = torch.rand(size=(I, B, D1))
    internal_differentials[(0, (0,))] = internal_tensor
    internal_differentials[(0, (0, 0))] = internal_tensor
    internal_differentials[(0, (0, 0, 0))] = internal_tensor
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2]]]
    einstein_notations[0, (0, 0)] = [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]
    einstein_notations[0, (0, 0, 0)] = [
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    ]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, I, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((I, B, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


### OPENING TREE GRAPH (o -> x_1, o -> x_2, x_1 -> x_3, x_2 -> x_4)


def test_15() -> None:
    # 15. [no independent dims, no batch, not diagonal]
    external_size: int = 2
    internal_size: int = 2
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    D2_1: int = 7
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    external_keys: list[Tuple[int, ...]] = [
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_differentials[(1, (1,))] = torch.rand(size=(D1_1, D2_1))
    internal_differentials[(1, (1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1))
    internal_differentials[(1, (1, 1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1, D2_1))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[0, (0, 0)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[0, (0, 0, 0)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    einstein_notations[1, (1,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[1, (1, 1)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[1, (1, 1, 1)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2_0, D2_0, D2_0)
    assert tuple(shapes.values()) == ((D2_0,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_16() -> None:
    # 16. [no independent dims, no batch, not diagonal]
    external_size: int = 2
    internal_size: int = 2
    variables: Tuple[int, ...] = (0, 1, 0)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    D2_1: int = 7
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    external_keys: list[Tuple[int, ...]] = [
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_differentials[(1, (1,))] = torch.rand(size=(D1_1, D2_1))
    internal_differentials[(1, (1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1))
    internal_differentials[(1, (1, 1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1, D2_1))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[0, (0, 0)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[0, (0, 0, 0)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    einstein_notations[1, (1,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[1, (1, 1)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[1, (1, 1, 1)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2_0, D2_1, D2_0)
    assert tuple(shapes.values()) == ((D2_0,), (D2_1,))
    assert tuple(indeps.values()) == ((None,), (None,))
    return None


### CLOSING TREE GRAPH (o -> x_1, o -> x_2, x_1 -> x_3, x_2 -> x_3)


def test_17() -> None:
    # 17. [no independent dims, no batch, not diagonal]
    external_size: int = 2
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    external_keys: list[Tuple[int, ...]] = [
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_differentials[(1, (0,))] = torch.rand(size=(D1_1, D2_0))
    internal_differentials[(1, (0, 0))] = torch.rand(size=(D1_1, D2_0, D2_0))
    internal_differentials[(1, (0, 0, 0))] = torch.rand(size=(D1_1, D2_0, D2_0, D2_0))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (0,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[0, (0, 0)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[0, (0, 0, 0)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    einstein_notations[1, (0,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[1, (0, 0)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[1, (0, 0, 0)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2_0, D2_0, D2_0)
    assert tuple(shapes.values()) == ((D2_0,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


### GIVING EXTRA VARIABLES


def test_18() -> None:
    # 18. [no independent dims, no batch, not diagonal]
    external_size: int = 5
    internal_size: int = 3
    variables: Tuple[int, ...] = (1, 1, 1)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[2] = (D1_1,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    external_indeps[2] = (None,)
    external_keys: list[Tuple[int, ...]] = [
        (0,),
        (2,),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2),
        (2, 0, 0),
        (2, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
    ]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (1,))] = torch.rand(size=(D1_0, D2_0))
    internal_differentials[(0, (1, 1))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_differentials[(0, (1, 1, 1))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_differentials[(2, (1,))] = torch.rand(size=(D1_1, D2_0))
    internal_differentials[(2, (1, 1))] = torch.rand(size=(D1_1, D2_0, D2_0))
    internal_differentials[(2, (1, 1, 1))] = torch.rand(size=(D1_1, D2_0, D2_0, D2_0))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[0, (1,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[0, (1, 1)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[0, (1, 1, 1)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    einstein_notations[2, (1,)] = [[[0], [0, 1]], [[1]]]
    einstein_notations[2, (1, 1)] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[2, (1, 1, 1)] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    for key, tensor in internal_differentials.items():
        einstein_notations[key].append([tuple(tensor.shape)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: Tuple[Tuple[int, ...]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2_0, D2_0, D2_0)
    assert tuple(shapes.values()) == ((D2_0,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


# DIMENSION BROADCASTING


def test_19() -> None:
    # Test: batch broadcasting
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D2: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Tuple[int, ...]] = dict()
    external_shapes[0] = (B,)
    external_indeps: dict[int, Tuple[Union[None, int], ...]] = dict()
    external_indeps[0] = (None,)
    # generate external differentials
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_differentials: dict[Tuple[int, ...], Tensor]
    external_differentials = _generate_test_external_differentials(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    # define internal differentials
    internal_differentials: dict[Tuple[int, Tuple[int, ...]], Tensor] = dict()
    internal_differentials[(0, (0,))] = torch.rand(size=(1, D2))
    internal_differentials[(0, (0, 0))] = torch.rand(size=(1, D2, D2))
    internal_differentials[(0, (0, 0, 0))] = torch.rand(size=(1, D2, D2, D2))
    einstein_notations: dict[Tuple[int, Tuple[int, ...]], Notation] = dict()
    einstein_notations[(0, (0,))] = [[[0], [0, 1]], [[1]]]
    einstein_notations[(0, (0, 0))] = [[[0], [0, 1, 2]], [[1], [2]]]
    einstein_notations[(0, (0, 0, 0))] = [[[0], [0, 1, 2, 3]], [[1], [2], [3]]]
    einstein_notations[(0, (0,))].append([(B, D2)])
    einstein_notations[(0, (0, 0))].append([(B, D2, D2)])
    einstein_notations[(0, (0, 0, 0))].append([(B, D2, D2, D2)])
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_differentials=external_differentials,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        internal_differentials=internal_differentials,
    )
    for key, val in einstein_notations.items():
        test.register_einstein_notation(key=key, val=val)
    tensor: Tensor
    shapes: dict[int, [Tuple[int, ...]]]
    indeps: Tuple[Tuple[Union[None, int], ...]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor.shape == (G, 1, D2, D2, D2)
    assert tuple(shapes.values()) == ((D2,),), (shapes, ((D2,)))
    assert tuple(indeps.values()) == ((None,),), indeps
    return None
