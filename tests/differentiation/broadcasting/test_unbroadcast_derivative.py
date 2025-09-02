# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import unbroadcast_derivative
from thoad.typing import Shape, Indep


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_unbroadcast_derivative_01() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: False
    # Reduce shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]
    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((1, 2, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
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

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 2, 3),
        *(2, 3, 2),
        *(1, 2, 3),
    )
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_02() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: False
    # Reduce shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((1, 2, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(1, 3), *(1, 3))
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps

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
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_03() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: True
    # Reduce shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((3, 1, 2),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    assert unbroadcasted_shapes == shapes  # unbroadcast must not permute dims
    assert unbroadcasted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((3, 1, 2), (2, 2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 2, 3),
        *(2, 3, 2),
        *(1, 2, 3),
    )
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_04() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: True
    # Reduce shape: False

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((3, 1, 2),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(1, 3), *(1, 3))
    assert unbroadcasted_shapes == shapes  # unbroadcast must not permute dims
    assert unbroadcasted_indeps == indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((1,), (2,))
    expected_shapes = ((3, 1, 2), (2, 2, 3))

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
    assert unbroadcasted_shapes == shapes
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_05() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: False
    # Reduce shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((2, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(2, 3),
        *(2, 3),
    ), unbroadcasted_derivative.shape
    assert unbroadcasted_shapes == ((2, 3),)
    assert unbroadcasted_indeps == ((None,),), unbroadcasted_indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((2, 3), (2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(2, 3), *(3, 2), *(2, 3))
    assert unbroadcasted_shapes == ((2, 3), (3, 2))
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_06() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: False
    # Reduce shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((2, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(3,), *(3,))
    assert unbroadcasted_shapes == ((2, 3),)
    assert unbroadcasted_indeps == ((0,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((1,), (2,))
    expected_shapes = ((2, 3), (3, 2))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(3,), *(3,), *(3,))
    assert unbroadcasted_shapes == ((2, 3), (3, 2))
    assert unbroadcasted_indeps == ((0,), (1,))


def test_unbroadcast_derivative_07() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: True
    # Reduce shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((None,),)
    expected_shapes = ((3, 2),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(2, 3), *(2, 3))
    assert unbroadcasted_shapes == ((2, 3),)
    assert unbroadcasted_indeps == ((None,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(1, 2, 3), *(2, 3, 2), *(1, 2, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((3, 2), (2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(2, 3), *(3, 2), *(2, 3))
    assert unbroadcasted_shapes == ((2, 3), (3, 2))
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_08() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: True
    # Reduce shape: True

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((1, 2, 3),)
    indeps = ((1,),)
    expected_shapes = ((3, 2),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(3,), *(3,))
    assert unbroadcasted_shapes == ((2, 3),)
    assert unbroadcasted_indeps == ((0,),), unbroadcasted_indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(1, 3), *(2, 3), *(1, 3))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((1, 2, 3), (2, 3, 2))
    indeps = ((1,), (2,))
    expected_shapes = ((3, 2), (2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(2,), *(3,), *(3,), *(3,))
    assert unbroadcasted_shapes == ((2, 3), (3, 2))
    assert unbroadcasted_indeps == ((0,), (1,))


def test_unbroadcast_derivative_09() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: False
    # Reduce shape: True  (Sum dimension to 1)

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(2, 3, 4), *(2, 3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((2, 3, 4),)
    indeps = ((None,),)
    expected_shapes = ((1, 3, 4),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(1, 3, 4), *(1, 3, 4))
    assert unbroadcasted_shapes == ((1, 3, 4),)
    assert unbroadcasted_indeps == ((None,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(2, 3, 4), *(2, 3, 2), *(2, 3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((1, 3, 4), (2, 3, 1))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 3, 4),
        *(2, 3, 1),
        *(1, 3, 4),
    )
    assert unbroadcasted_shapes == ((1, 3, 4), (2, 3, 1))
    assert unbroadcasted_indeps == indeps


def test_unbroadcast_derivative_10() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: False
    # Reduce shape: True  (Sum dimension to 1)

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((2, 3, 4),)
    indeps = ((0,),)
    expected_shapes = ((1, 3, 4),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 3, 4),
        *(1, 3, 4),
    ), unbroadcasted_derivative.shape
    assert unbroadcasted_shapes == ((1, 3, 4),), unbroadcasted_shapes
    assert unbroadcasted_indeps == ((None,),), unbroadcasted_indeps

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(2, 3), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((0,), (2,))
    expected_shapes = ((1, 3, 4), (2, 3, 1))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 3, 4),
        *(2, 3, 1),
        *(1, 3, 4),
    ), unbroadcasted_derivative.shape
    assert unbroadcasted_shapes == ((1, 3, 4), (2, 3, 1))
    assert unbroadcasted_indeps == ((None,), (None,))

    ### Test case C

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(2, 3), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((0,), (2,))
    expected_shapes = ((1, 3, 4), (2, 3, 2))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(2,),
        *(1, 3, 4),
        *(2, 3),
        *(1, 3, 4),
    )
    assert unbroadcasted_shapes == ((1, 3, 4), (2, 3, 2))
    assert unbroadcasted_indeps == ((None,), (2,))


def test_unbroadcast_derivative_11() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: False
    # Permute shape: True
    # Reduce shape: True  (Sum dimension to 1)

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(1,), *(2, 3, 4), *(2, 3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((2, 3, 4),)
    indeps = ((None,),)
    expected_shapes = ((4, 1, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(1, 3, 4), *(1, 3, 4))
    assert unbroadcasted_shapes == ((1, 3, 4),)
    assert unbroadcasted_indeps == ((None,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(1,), *(2, 3, 4), *(2, 3, 2), *(2, 3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((None,), (None,))
    expected_shapes = ((4, 1, 3), (2, 1, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 3, 4),
        *(2, 3, 1),
        *(1, 3, 4),
    )
    assert unbroadcasted_shapes == ((1, 3, 4), (2, 3, 1))
    assert unbroadcasted_indeps == ((None,), (None,))


def test_unbroadcast_derivative_12() -> None:

    # TEST DESCRIPTION
    # Independent dimensions: True
    # Permute shape: False
    # Reduce shape: True  (Sum dimension to 1)

    XX: int
    derivative_view: Tuple[int, ...]
    derivative: Tensor

    variables: Tuple[int, ...]
    shapes: Tuple[Shape, ...]
    indeps: Tuple[Indep, ...]
    expected_shapes: Tuple[Shape, ...]

    unbroadcasted_derivative: Tensor
    unbroadcasted_shapes: Tuple[Shape, ...]
    unbroadcasted_indeps: Tuple[Indep, ...]

    ### Test case A

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 0)
    shapes = ((2, 3, 4),)
    indeps = ((0,),)
    expected_shapes = ((4, 1, 3),)

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (XX, *(1,), *(1, 3, 4), *(1, 3, 4))
    assert unbroadcasted_shapes == ((1, 3, 4),)
    assert unbroadcasted_indeps == ((None,),)

    ### Test case B

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(2, 3), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((0,), (0,))
    expected_shapes = ((4, 1, 3), (1, 2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(1,),
        *(1, 3, 4),
        *(1, 3, 2),
        *(1, 3, 4),
    ), unbroadcasted_derivative.shape
    assert unbroadcasted_shapes == ((1, 3, 4), (1, 3, 2))
    assert unbroadcasted_indeps == ((None,), (None,))

    ### Test case C

    XX = 10
    derivative_view = (XX, *(2,), *(3, 4), *(2, 3), *(3, 4))
    derivative = torch.rand(size=derivative_view, device=device)

    variables = (0, 1, 0)
    shapes = ((2, 3, 4), (2, 3, 2))
    indeps = ((0,), (2,))
    expected_shapes = ((4, 1, 3), (2, 2, 3))

    (unbroadcasted_derivative, unbroadcasted_shapes, unbroadcasted_indeps) = (
        unbroadcast_derivative(
            derivative=derivative,
            variables=variables,
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
        )
    )

    assert unbroadcasted_derivative.shape == (
        XX,
        *(2,),
        *(1, 3, 4),
        *(2, 3),
        *(1, 3, 4),
    )
    assert unbroadcasted_shapes == ((1, 3, 4), (2, 3, 2))
    assert unbroadcasted_indeps == ((None,), (2,))
