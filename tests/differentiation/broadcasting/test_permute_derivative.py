# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import (
    permute_derivative,
    unbroadcast_derivative,
)
from thoad.typing import Shape, Indep


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_permute_derivative_01() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((1, 2, 3),)

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((1, 2, 3), (2, 3, 2))

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (
        XX,
        *(1,),
        *(1, 2, 3),
        *(2, 3, 2),
        *(1, 2, 3),
    )
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == indeps


def test_permute_derivative_02() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((1, 2, 3),)

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (XX, *(2,), *(1, 3), *(1, 3))
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((1,), (2,))
    expected_shapes = ((1, 2, 3), (2, 3, 2))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    assert unbroadcasted_shapes == expected_shapes
    assert unbroadcasted_indeps == indeps


def test_permute_derivative_03() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((3, 1, 2),)

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (XX, *(1,), *(3, 1, 2), *(3, 1, 2))
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((3, 1, 2), (2, 2, 3))

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (
        XX,
        *(1,),
        *(3, 1, 2),
        *(2, 2, 3),
        *(3, 1, 2),
    )
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == indeps


def test_permute_derivative_04() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    permuted_derivative: Tensor
    permuted_shapes: Tuple[Shape, ...]
    permuted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((3, 1, 2),)

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (XX, *(1,), *(3, 1), *(3, 1))
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == ((2,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((1,), (2,))
    expected_shapes = ((3, 1, 2), (2, 2, 3))

    (permuted_derivative, permuted_shapes, permuted_indeps) = permute_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
    )

    assert permuted_derivative.shape == (XX, *(2,), *(3, 1), *(2, 3), *(3, 1))
    assert permuted_shapes == expected_shapes
    assert permuted_indeps == ((2,), (1,))
