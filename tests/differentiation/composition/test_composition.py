import itertools
from typing import Any, Tuple, Union

import torch
from torch import Tensor

from thoad.typing import Shape, Indep, VPerm, Notation
from thoad.differentiation.engine.composition.numeric.combination import (
    produce_variations,
)
from thoad.differentiation.engine.composition.numeric.composition import (
    Loader,
    InternalKey,
)


def _populated(obj: Union[None, Any]) -> Any:
    assert obj is not None
    return obj


### Test Function to Produce Variations


def test_produce_variations() -> None:
    sizes: list[int] = list(range(1, 4))
    sets: list[list[int]] = [list(range(sz)) for sz in sizes]
    for elements, size in itertools.product(sets, sizes):
        variations: list[Tuple[int, ...]]
        variations = produce_variations(elements=elements, size=size)
    return None


def _calculate_test_independent_sizes(
    external_shapes: dict[int, Shape],
    external_indeps: dict[int, Indep],
) -> Tuple[int, ...]:
    independent_sizes: list[int] = list()
    for row in zip(*external_indeps.values()):
        candidates: list[int] = list()
        for i, dim in enumerate(row):
            candidates.append(1 if dim is None else external_shapes[i][dim])
        assert len(set(row)) == 1
        independent_sizes.append(set(candidates).pop())
    return tuple(independent_sizes)


def _calculate_test_distributed_shapes(
    key: Tuple[int, ...],
    external_shapes: dict[int, Shape],
    external_indeps: dict[int, Indep],
) -> Tuple[Shape, ...]:
    shapes: list[Shape] = [external_shapes[v] for v in key]
    indeps: list[Indep] = [external_indeps[v] for v in key]
    distributed_shapes: list[list[int]] = [list() for _ in key]
    for (
        i,
        (shape, indep),
    ) in enumerate(zip(shapes, indeps)):
        for j, sz in enumerate(shape):
            if j not in indep:
                distributed_shapes[i].append(sz)
    return tuple(tuple(S) for S in distributed_shapes)


def _generate_test_external_derivatives(
    GOnumel: int,
    external_keys: list[Tuple[int, ...]],
    external_shapes: dict[int, Shape],
    external_indeps: dict[int, Indep],
) -> dict[Tuple[int, ...], Union[None, Tensor]]:
    internal_derivatives: dict[Tuple[int, ...], Union[None, Tensor]] = dict()
    independent_sizes: Tuple[int, ...] = _calculate_test_independent_sizes(
        external_shapes=external_shapes,
        external_indeps=external_indeps,
    )
    for key in external_keys:
        distributed_shapes: Tuple[Shape, ...] = _calculate_test_distributed_shapes(
            key=key,
            external_shapes=external_shapes,
            external_indeps=external_indeps,
        )
        derivative_shape: list[int] = [GOnumel, *independent_sizes]
        for distributed_shape in distributed_shapes:
            derivative_shape.extend(distributed_shape)
        internal_derivatives[key] = torch.rand(size=tuple(derivative_shape))
    return internal_derivatives


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1, D2))  # a,aA->A
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))  # a,aAB->AB
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(
        size=(D1, D2, D2, D2)
    )  # a,aABC->ABC
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation = notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D11, D12)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D11, D12)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(*(D12, D11), D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(*(D12, D11), D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(*(D12, D11), D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1, 0, 2)), ((2,),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1, 0, 2, 3)), ((2,), (3,))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (1, 0, 2, 3, 4)),
        ((2,), (3,), (4,)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 2)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D11, D12)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D11, D12)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(*(D12, D11), *(D22, D21)))
    internal_derivatives[(0, (0, 0))] = torch.rand(
        size=(*(D12, D11), *(D22, D21), *(D22, D21))
    )
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(
        size=(*(D12, D11), *(D22, D21), *(D22, D21), *(D22, D21))
    )
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1, 0, 2, 3)), ((3, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1, 0, 2, 3, 4, 5)), ((3, 2), (5, 4))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (1, 0, 2, 3, 4, 5, 6, 7)),
        ((3, 2), (5, 4), (7, 6)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) * 2 + 2)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D11, D12)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D11, D12)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(*(D12, D11), D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(*(D12, D11), D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(*(D12, D11), D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1, 2), (2, 1, 3)), ((0, 3),)]
    einstein_notations[0, (0, 0)] = [((0, 1, 2), (2, 1, 3, 4)), ((0, 3), (0, 4))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1, 2), (2, 1, 3, 4, 5)),
        ((0, 3), (0, 4), (0, 5)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 2)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1, 2)), ((0, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1, 2, 3)), ((0, 2), (0, 3))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (1, 2, 3, 4)),
        ((0, 2), (0, 3), (0, 4)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D2, D2, D2)
    assert tuple(shapes.values()) == ((I, D2),), shapes.values()
    assert tuple(indeps.values()) == ((0,),), indeps.values()
    return None


def test_03a() -> None:
    # 03. [full-independent dims, no batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(I, D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(I, D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(I, D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1, 2)), ((0, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1, 2, 3)), ((0, 2), (0, 3))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (0, 1, 2, 3, 4)),
        ((0, 2), (0, 3), (0, 4)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, *(False for _ in range(len(key[1]) + 1))),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D2, D2, D2)
    assert tuple(shapes.values()) == ((I, D2),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_03b() -> None:
    # 03. [full-independent dims, no batch, not diagonal] + broadcasting
    # internal_flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I1: int = 1
    I2: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivative: Tensor = torch.rand(size=(I2,))
    internal_derivatives[(0, (0,))] = internal_derivative
    internal_derivatives[(0, (0, 0))] = internal_derivative
    internal_derivatives[(0, (0, 0, 0))] = internal_derivative
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I2)
    assert tuple(shapes.values()) == ((I2,),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_03c() -> None:
    # 03. [full-independent dims, no batch, not diagonal] + broadcasting
    # internal_flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I1: int = 1
    I2: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivative: Tensor = torch.rand(size=(I2,))
    internal_derivatives[(0, (0,))] = internal_derivative
    internal_derivatives[(0, (0, 0))] = internal_derivative
    internal_derivatives[(0, (0, 0, 0))] = internal_derivative
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I2)
    assert tuple(shapes.values()) == ((I2,),)
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1, 2)), ((0, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1, 2, 3)), ((0, 2), (0, 3))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (1, 2, 3, 4)),
        ((0, 2), (0, 3), (0, 4)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(B, D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(B, D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(B, D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((1,), (0, 1, 2)), ((0, 2),)]
    einstein_notations[0, (0, 0)] = [((1,), (0, 1, 2, 3)), ((0, 2), (0, 3))]
    einstein_notations[0, (0, 0, 0)] = [
        ((1,), (0, 1, 2, 3, 4)),
        ((0, 2), (0, 3), (0, 4)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 2)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_06a() -> None:
    # 06. [no independent dims, full batch, not diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    D2: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(B, D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(B, D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(B, D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1, 2)), ((0, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1, 2, 3)), ((0, 2), (0, 3))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1), (0, 1, 2, 3, 4)),
        ((0, 2), (0, 3), (0, 4)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 2)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B, D2, B, D2, B, D2)
    assert tuple(shapes.values()) == ((B, D2),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_06b() -> None:
    # 06. [no independent dims, full batch, not diagonal] + broacasting
    # internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B1: int = 1
    B2: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivative: Tensor = torch.rand(size=(B2,))
    internal_derivatives[(0, (0,))] = internal_derivative
    internal_derivatives[(0, (0, 0))] = internal_derivative
    internal_derivatives[(0, (0, 0, 0))] = internal_derivative
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B2, B2, B2)
    assert tuple(shapes.values()) == ((B2,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_06c() -> None:
    # 06. [no independent dims, full batch, not diagonal] + broacasting
    # internal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B1: int = 1
    B2: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivative: Tensor = torch.rand(size=(B2,))
    internal_derivatives[(0, (0,))] = internal_derivative
    internal_derivatives[(0, (0, 0))] = internal_derivative
    internal_derivatives[(0, (0, 0, 0))] = internal_derivative
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, B2, B2, B2)
    assert tuple(shapes.values()) == ((B2,),)
    assert tuple(indeps.values()) == (tuple(),)
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
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(I, B, D1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(I, B, D1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(I, B, D1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1, 2), (0, 1, 2, 3)), ((0, 1, 3),)]
    einstein_notations[0, (0, 0)] = [
        ((0, 1, 2), (0, 1, 2, 3, 4)),
        ((0, 1, 3), (0, 1, 4)),
    ]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1, 2), (0, 1, 2, 3, 4, 5)),
        ((0, 1, 3), (0, 1, 4), (0, 1, 5)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True, *(False for _ in range(len(key[1]) + 1))),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, B, D2, B, D2, B, D2), tensor.shape
    assert tuple(shapes.values()) == ((I, B, D2),)
    assert tuple(indeps.values()) == ((0,),)
    return None


### LINEAR GRAPH (o -> x_1 -> x_2) (DIAGONAL INTERNALS)


def test_08a() -> None:
    # 08. [no independent dims, no batch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, D1, D1, D1)
    assert tuple(shapes.values()) == ((D1,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_08b() -> None:
    # 08. [no independent dims, no batch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1: int = 4
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0,)), ((0,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0,)), ((0,), (0,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0,)), ((0,), (0,), (0,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, D1, D1, D1), tensor.shape
    assert tuple(shapes.values()) == ((D1,),)
    assert tuple(indeps.values()) == (tuple(),), indeps.values()
    return None


def test_09a() -> None:
    # 09. [pre-independent dims, no batch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1,)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_09b() -> None:
    # 09. [pre-independent dims, no batch, diagonal]
    # Inernal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1,)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_10a() -> None:
    # 10. [full-independent dims, no batch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(I, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_10b() -> None:
    # 10. [full-independent dims, no batch, diagonal]
    # Internal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(I, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, D1, D1, D1)
    assert tuple(shapes.values()) == ((I, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_11a() -> None:
    # 11. [no independent dims, prebatch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1,)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_11b() -> None:
    # 11. [no independent dims, prebatch, diagonal]
    # Internal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(D1,))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (1,)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (1,)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True,),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == (tuple(),)
    return None


def test_12a() -> None:
    # 12. [no independent dims, postbatch, diagonal]
    # Inernal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((1,), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((1,), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((1,), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_12b() -> None:
    # 12. [no independent dims, postbatch, diagonal]
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((1,), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((1,), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((1,), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, B, D1, B, D1, B, D1), tensor.shape
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == (tuple(),)
    return None


def test_13a() -> None:
    # 13. [no independent dims, full batch, diagonal]
    # internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_13b() -> None:
    # 13. [no independent dims, full batch, diagonal]
    # internal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    B: int = 4
    D1: int = 5
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1), (0, 1)), ((0, 1),)]
    einstein_notations[0, (0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1))]
    einstein_notations[0, (0, 0, 0)] = [((0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((B, D1),)
    assert tuple(indeps.values()) == (tuple(),)
    return None


def test_14a() -> None:
    # 14. [full-independent dims, full-batch, diagonal]
    # Internal flexibilities False
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    B: int = 5
    D1: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(I, B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1, 2), (0, 1, 2)), ((0, 1, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1, 2), (0, 1, 2)), ((0, 1, 2), (0, 1, 2))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1, 2), (0, 1, 2)),
        ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((I, B, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


def test_14b() -> None:
    # 14. [full-independent dims, full-batch, diagonal]
    # Internal flexibilities True
    external_size: int = 1
    internal_size: int = 1
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    I: int = 4
    B: int = 5
    D1: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (I, B, D1)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (I, B, D1)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_tensor: Tensor = torch.rand(size=(I, B, D1))
    internal_derivatives[(0, (0,))] = internal_tensor
    internal_derivatives[(0, (0, 0))] = internal_tensor
    internal_derivatives[(0, (0, 0, 0))] = internal_tensor
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0, 1, 2), (0, 1, 2)), ((0, 1, 2),)]
    einstein_notations[0, (0, 0)] = [((0, 1, 2), (0, 1, 2)), ((0, 1, 2), (0, 1, 2))]
    einstein_notations[0, (0, 0, 0)] = [
        ((0, 1, 2), (0, 1, 2)),
        ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
    ]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (True, True, True),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, I, B, D1, B, D1, B, D1)
    assert tuple(shapes.values()) == ((I, B, D1),)
    assert tuple(indeps.values()) == ((0,),)
    return None


### OPENING TREE GRAPH (o -> x_1, o -> x_2, x_1 -> x_3, x_2 -> x_4)


def test_15a() -> None:
    # 15. [no independent dims, no batch, not diagonal]
    external_size: int = 2
    internal_size: int = 2
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    D2_1: int = 7
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1_0,)
    expected_shapes[1] = (D1_1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    expected_indeps[(1, 1)] = (None,)
    # generate external derivatives
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
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(1,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 1)] = (0, 1)
    external_vperms[(1, 0)] = (0, 1)
    external_vperms[(1, 1)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    external_vperms[(0, 0, 1)] = (0, 1, 2)
    external_vperms[(0, 1, 0)] = (0, 1, 2)
    external_vperms[(0, 1, 1)] = (0, 1, 2)
    external_vperms[(1, 0, 0)] = (0, 1, 2)
    external_vperms[(1, 0, 1)] = (0, 1, 2)
    external_vperms[(1, 1, 0)] = (0, 1, 2)
    external_vperms[(1, 1, 1)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_derivatives[(1, (1,))] = torch.rand(size=(D1_1, D2_1))
    internal_derivatives[(1, (1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1))
    internal_derivatives[(1, (1, 1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1, D2_1))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    einstein_notations[1, (1,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[1, (1, 1)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[1, (1, 1, 1)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, D2_0, D2_0, D2_0)
    assert tuple(shapes.values()) == ((D2_0,),)
    assert tuple(indeps.values()) == ((None,),)
    return None


def test_15b() -> None:
    # 15. [no independent dims, no batch, not diagonal]
    external_size: int = 2
    internal_size: int = 2
    variables: Tuple[int, ...] = (0, 0, 0)
    G: int = 3
    D1_0: int = 4
    D1_1: int = 5
    D2_0: int = 6
    D2_1: int = 7
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1_0,)
    expected_shapes[1] = (D1_1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    expected_indeps[(1, 1)] = (None,)
    # generate external derivatives
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
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_derivatives[(1, 0)] = external_derivatives[(0, 1)]
    external_derivatives[(0, 1, 0)] = external_derivatives[(0, 0, 1)]
    external_derivatives[(1, 0, 0)] = external_derivatives[(0, 0, 1)]
    external_derivatives[(1, 1, 0)] = external_derivatives[(0, 1, 1)]
    external_derivatives[(1, 0, 1)] = external_derivatives[(0, 1, 1)]
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(1,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 1)] = (0, 1)
    external_vperms[(1, 0)] = (1, 0)  # permuted
    external_vperms[(1, 1)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    external_vperms[(0, 0, 1)] = (0, 1, 2)
    external_vperms[(0, 1, 0)] = (0, 2, 1)  # permuted
    external_vperms[(0, 1, 1)] = (0, 1, 2)
    external_vperms[(1, 0, 0)] = (2, 0, 1)  # permuted
    external_vperms[(1, 0, 1)] = (1, 0, 2)  # permuted
    external_vperms[(1, 1, 0)] = (1, 2, 0)  # permuted
    external_vperms[(1, 1, 1)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_derivatives[(1, (1,))] = torch.rand(size=(D1_1, D2_1))
    internal_derivatives[(1, (1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1))
    internal_derivatives[(1, (1, 1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1, D2_1))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    einstein_notations[1, (1,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[1, (1, 1)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[1, (1, 1, 1)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1_0,)
    expected_shapes[1] = (D1_1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    expected_indeps[(1, 1)] = (None,)
    # generate external derivatives
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
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(1,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 1)] = (0, 1)
    external_vperms[(1, 0)] = (0, 1)
    external_vperms[(1, 1)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    external_vperms[(0, 0, 1)] = (0, 1, 2)
    external_vperms[(0, 1, 0)] = (0, 1, 2)
    external_vperms[(0, 1, 1)] = (0, 1, 2)
    external_vperms[(1, 0, 0)] = (0, 1, 2)
    external_vperms[(1, 0, 1)] = (0, 1, 2)
    external_vperms[(1, 1, 0)] = (0, 1, 2)
    external_vperms[(1, 1, 1)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_derivatives[(1, (1,))] = torch.rand(size=(D1_1, D2_1))
    internal_derivatives[(1, (1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1))
    internal_derivatives[(1, (1, 1, 1))] = torch.rand(size=(D1_1, D2_1, D2_1, D2_1))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    einstein_notations[1, (1,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[1, (1, 1)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[1, (1, 1, 1)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[1] = (D1_1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    external_indeps[1] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1_0,)
    expected_shapes[1] = (D1_1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    expected_indeps[(1, 0)] = (None,)
    # generate external derivatives
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
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(1,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 1)] = (0, 1)
    external_vperms[(1, 0)] = (0, 1)
    external_vperms[(1, 1)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    external_vperms[(0, 0, 1)] = (0, 1, 2)
    external_vperms[(0, 1, 0)] = (0, 1, 2)
    external_vperms[(0, 1, 1)] = (0, 1, 2)
    external_vperms[(1, 0, 0)] = (0, 1, 2)
    external_vperms[(1, 0, 1)] = (0, 1, 2)
    external_vperms[(1, 1, 0)] = (0, 1, 2)
    external_vperms[(1, 1, 1)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D1_0, D2_0))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_derivatives[(1, (0,))] = torch.rand(size=(D1_1, D2_0))
    internal_derivatives[(1, (0, 0))] = torch.rand(size=(D1_1, D2_0, D2_0))
    internal_derivatives[(1, (0, 0, 0))] = torch.rand(size=(D1_1, D2_0, D2_0, D2_0))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    einstein_notations[1, (0,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[1, (0, 0)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[1, (0, 0, 0)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1_0,)
    external_shapes[2] = (D1_1,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    external_indeps[2] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1_0,)
    expected_shapes[2] = (D1_1,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 1)] = (None,)
    expected_indeps[(2, 1)] = (None,)
    # generate external derivatives
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
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(2,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 2)] = (0, 1)
    external_vperms[(2, 0)] = (0, 1)
    external_vperms[(2, 2)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    external_vperms[(0, 0, 2)] = (0, 1, 2)
    external_vperms[(0, 2, 0)] = (0, 1, 2)
    external_vperms[(0, 2, 2)] = (0, 1, 2)
    external_vperms[(2, 0, 0)] = (0, 1, 2)
    external_vperms[(2, 0, 2)] = (0, 1, 2)
    external_vperms[(2, 2, 0)] = (0, 1, 2)
    external_vperms[(2, 2, 2)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (1,))] = torch.rand(size=(D1_0, D2_0))
    internal_derivatives[(0, (1, 1))] = torch.rand(size=(D1_0, D2_0, D2_0))
    internal_derivatives[(0, (1, 1, 1))] = torch.rand(size=(D1_0, D2_0, D2_0, D2_0))
    internal_derivatives[(2, (1,))] = torch.rand(size=(D1_1, D2_0))
    internal_derivatives[(2, (1, 1))] = torch.rand(size=(D1_1, D2_0, D2_0))
    internal_derivatives[(2, (1, 1, 1))] = torch.rand(size=(D1_1, D2_0, D2_0, D2_0))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[0, (1,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[0, (1, 1)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[0, (1, 1, 1)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    einstein_notations[2, (1,)] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[2, (1, 1)] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[2, (1, 1, 1)] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
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
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (B,)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (None,)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (B,)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (None,)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0), (0, 0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(1, D2))
    internal_derivatives[(0, (0, 0))] = torch.rand(size=(1, D2, D2))
    internal_derivatives[(0, (0, 0, 0))] = torch.rand(size=(1, D2, D2, D2))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: False for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[(0, (0,))] = [((0,), (0, 1)), ((1,),)]
    einstein_notations[(0, (0, 0))] = [((0,), (0, 1, 2)), ((1,), (2,))]
    einstein_notations[(0, (0, 0, 0))] = [((0,), (0, 1, 2, 3)), ((1,), (2,), (3,))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            (B, *tensor.shape[1:]),
            tuple(False for _ in range(len(key[1]) + 1)),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, 1, D2, D2, D2)
    assert tuple(shapes.values()) == ((D2,),), (shapes, ((D2,)))
    assert tuple(indeps.values()) == ((None,),), indeps
    return None


def test_20() -> None:
    # Test: Matrix Multiplication
    external_size: int = 1
    internal_size: int = 2
    variables: Tuple[int, ...] = (0, 1)
    G: int = 3
    D1: int = 4
    D2: int = 5
    D3: int = 6
    # define external attributes (shapes and indeps)
    external_shapes: dict[int, Union[None, Shape]] = dict()
    external_shapes[0] = (D1, D2)
    external_indeps: dict[int, Union[None, Indep]] = dict()
    external_indeps[0] = (0, 1)
    # define expected attributes (shapes and indeps)
    expected_shapes: dict[int, Union[None, Shape]] = dict()
    expected_shapes[0] = (D1, D2)
    expected_indeps: dict[Tuple[int, int], Union[None, Indep]] = dict()
    expected_indeps[(0, 0)] = (0, None)
    expected_indeps[(0, 1)] = (None, 1)
    # generate external derivatives
    external_keys: list[Tuple[int, ...]] = [(0,), (0, 0)]
    external_derivatives: dict[Tuple[int, ...], Union[None, Tensor]]
    external_derivatives = _generate_test_external_derivatives(
        GOnumel=G,
        external_keys=external_keys,
        external_shapes={k: _populated(S) for k, S in external_shapes.items()},
        external_indeps={k: _populated(I) for k, I in external_indeps.items()},
    )
    external_vperms: dict[Tuple[int, ...], Union[None, VPerm]] = dict()
    external_vperms[(0,)] = (0,)
    external_vperms[(0, 0)] = (0, 1)
    external_vperms[(0, 0, 0)] = (0, 1, 2)
    # define internal derivatives
    internal_derivatives: dict[Tuple[int, Tuple[int, ...]], Union[None, Tensor]]
    internal_derivatives = dict()
    internal_derivatives[(0, (0,))] = torch.rand(size=(D3, D2))
    internal_derivatives[(0, (1,))] = torch.rand(size=(D1, D3))
    internal_derivatives[(0, (0, 1))] = torch.rand(size=(D3, D3))
    internal_derivatives[(0, (1, 0))] = torch.rand(size=(D3, D3))
    internal_flexibilities: dict[int, bool]
    internal_flexibilities = {i: True for i in range(internal_size)}
    einstein_notations: dict[InternalKey, Union[None, Notation]] = dict()
    einstein_notations[(0, (0,))] = [((0, 2), (1, 2)), ((0, 1),)]
    einstein_notations[(0, (1,))] = [((0, 2), (0, 1)), ((1, 2),)]
    einstein_notations[(0, (0, 1))] = [((0, 2), (1, 3)), ((0, 1), (3, 2))]
    einstein_notations[(0, (1, 0))] = [((0, 2), (1, 3)), ((1, 2), (0, 3))]
    for key, tensor in internal_derivatives.items():
        assert tensor is not None
        internal_info: Tuple[Tuple[int, ...], Tuple[bool, ...]] = (
            tuple(tensor.shape),
            (False, False),
        )
        notation: Union[None, Notation] = einstein_notations[key]
        assert notation is not None
        notation.append(internal_info)
    test = Loader(
        external_size=external_size,
        internal_size=internal_size,
        max_order=len(variables),
        external_derivatives=external_derivatives,
        external_shapes=external_shapes,
        external_indeps=external_indeps,
        external_vperms=external_vperms,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        internal_derivatives=internal_derivatives,
        internal_flexibilities=internal_flexibilities,
        einstein_notations=einstein_notations,
        dtype=torch.float32,
        device=device,
    )
    for key, val in einstein_notations.items():
        assert val is not None
        test.register_einstein_notation(key=key, val=val)
    tensor: Union[None, Tensor]
    shapes: Union[None, dict[int, Shape]]
    indeps: Union[None, dict[int, Indep]]
    (tensor, shapes, indeps) = test.compose(variables=variables)
    assert tensor is not None
    assert shapes is not None
    assert indeps is not None
    assert tensor.shape == (G, D1, D2, D3, D3)
    assert tuple(shapes.values()) == ((D1, D3), (D3, D2)), (shapes, ((D2,)))
    assert tuple(indeps.values()) == ((0, None), (None, 1)), indeps
    return None
