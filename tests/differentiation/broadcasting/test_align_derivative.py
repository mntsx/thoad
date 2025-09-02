# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import align_derivative


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_align_derivative_01() -> None:

    # Test case: no modifications

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 9, 12, 9, 12, 9, 12), deriv_shape

    return None


def test_align_derivative_02() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 9, 12, 4, 9, 12, 4, 9, 12), deriv_shape

    return None


def test_align_derivative_03() -> None:

    # Test case:
    #   1. permute non independent dimensions

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 12, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 12, 9, 12, 9, 12, 9), deriv_shape

    return None


def test_align_derivative_04() -> None:

    # Test case:
    #   1. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 12, 9, 12, 9, 12, 9), deriv_shape

    return None


def test_align_derivative_05() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 12, 4, 9, 12, 4, 9, 12, 4, 9), deriv_shape

    return None


def test_align_derivative_06() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 9, 12, 9, 12, 9, 12), deriv_shape

    return None


def test_align_derivative_07() -> None:

    # Test case:
    #   1. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    # ### Checks
    # deriv_shape: Tuple[int, ...] = modified_derivative.shape
    # assert deriv_shape == (XX, 2, 9, 12, 9, 12, 9, 12), deriv_shape

    return None


def test_align_derivative_08() -> None:

    # Test case:
    #   1. partial collapse independent dimension
    #   2. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    # ### Checks
    # deriv_shape: Tuple[int, ...] = modified_derivative.shape
    # assert deriv_shape == (XX, 2, 3, 12, 3, 12, 3, 12), deriv_shape

    return None


def test_align_derivative_09() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 9, 12, 1, 9, 12, 1, 9, 12), deriv_shape

    return None


def test_align_derivative_10() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension
    #   3. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 9, 12, 9, 12, 9, 12)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 1, 12, 1, 1, 12, 1, 1, 12), deriv_shape

    return None


def test_align_derivative_11() -> None:

    # Test case: no modifications

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(9, 12), *(6, 7), *(9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(9, 12), *(6, 7), *(9, 12)), deriv_shape

    return None


def test_align_derivative_12() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 9, 12), *(4, 6, 7), *(4, 9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(4, 9, 12),
        *(4, 6, 7),
        *(4, 9, 12),
    ), deriv_shape

    return None


def test_align_derivative_13() -> None:

    # Test case:
    #   1. permute non independent dimensions

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 12, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(12, 9), *(6, 7), *(12, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(12, 9), *(6, 7), *(12, 9)), deriv_shape

    return None


def test_align_derivative_14() -> None:

    # Test case:
    #   1. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(12, 9), *(6, 7), *(12, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(12, 9), *(6, 7), *(12, 9)), deriv_shape

    return None


def test_align_derivative_15() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(12, 4, 9), *(4, 6, 7), *(12, 4, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(12, 4, 9),
        *(4, 6, 7),
        *(12, 4, 9),
    ), deriv_shape

    return None


def test_align_derivative_16() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(9, 12), *(4, 6, 7), *(9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1,), *(9, 12), *(4, 6, 7), *(9, 12)), deriv_shape

    return None


def test_align_derivative_17() -> None:

    # Test case:
    #   1. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    # ### Checks
    # deriv_shape: Tuple[int, ...] = modified_derivative.shape
    # assert deriv_shape == (XX,*(2,), *(9, 12), *(6, 7), *(9, 12)), deriv_shape

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    return None


def test_align_derivative_18() -> None:

    # Test case:
    #   1. partial collapse independent dimension
    #   2. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    return None


def test_align_derivative_19() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 9, 12), (1, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 9, 12), *(1, 6, 7), *(1, 9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(1, 9, 12),
        *(1, 6, 7),
        *(1, 9, 12),
    ), deriv_shape

    return None


def test_align_derivative_20() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension
    #   3. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 12), (1, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 1, 12), *(1, 6, 7), *(1, 1, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(1, 1, 12),
        *(1, 6, 7),
        *(1, 1, 12),
    ), deriv_shape

    return None


test_align_derivative_11()
print("test align_derivative 11 passed")
test_align_derivative_12()
print("test align_derivative 12 passed")
test_align_derivative_13()
print("test align_derivative 13 passed")
test_align_derivative_14()
print("test align_derivative 14 passed")
test_align_derivative_15()
print("test align_derivative 15 passed")
test_align_derivative_16()
print("test align_derivative 16 passed")
test_align_derivative_17()
print("test align_derivative 17 passed")
test_align_derivative_18()
print("test align_derivative 18 passed")
test_align_derivative_19()
print("test align_derivative 19 passed")
test_align_derivative_20()
print("test align_derivative 20 passed")


def test_align_derivative_21() -> None:

    # Test case: no modifications

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(1,), *(1, 1), *(1, 1), *(1, 1))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 1, 1, 1, 1, 1, 1), deriv_shape

    return None


def test_align_derivative_22() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(1,), *(1, 1), *(1, 1), *(1, 1))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 1, 1), *(1, 1, 1), *(1, 1, 1)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1,), *(1, 1, 1), *(1, 1, 1), *(1, 1, 1))

    return None


def test_align_derivative_23() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(1,), *(1, 1), *(1, 1), *(1, 1))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,),)

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 1), *(1, 1), *(1, 1)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1,), *(1, 1), *(1, 1), *(1, 1)), deriv_shape

    return None


def test_align_derivative_24() -> None:

    # Test case:
    # 1. permutation in independent dimensions

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 6, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1),)
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1, 0),)

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 6, 6), deriv_shape

    return None


def test_align_derivative_25() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 7, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1, 0), (1, 0))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 7, 6), deriv_shape

    return None


def test_align_derivative_26() -> None:

    # Test case:
    # 1. distribution of 1 independent dimension for subset of variables

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 7, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, None))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 5, 7, 6), deriv_shape

    return None


def test_align_derivative_27() -> None:

    # Test case:
    # 1. distribution of 2 independent dimensions for subset of variables

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 7, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None, 1),
        (0, None),
    )

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 4, 6, 5, 7, 4, 6), deriv_shape

    return None


def test_align_derivative_28() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables
    # 2. distribution of 1 independent dimension for subset of variables

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 7, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1, 0), (1, None))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 5, 7, 6), deriv_shape

    return None


def test_align_derivative_29() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables
    # 2. distribution of 1 independent dimension for subset of variables

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 7, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None, 0),
        (1, None),
    )

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 4, 6, 5, 7, 4, 6), deriv_shape

    return None


def test_align_derivative_30() -> None:

    # Test case:
    # 1. input null independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 1, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 4, 6, 4, 5), deriv_shape

    return None


test_align_derivative_21()
print("test align_derivative 21 passed")
test_align_derivative_22()
print("test align_derivative 22 passed")
test_align_derivative_23()
print("test align_derivative 23 passed")
test_align_derivative_24()
print("test align_derivative 24 passed")
test_align_derivative_25()
print("test align_derivative 25 passed")
test_align_derivative_26()
print("test align_derivative 26 passed")
test_align_derivative_27()
print("test align_derivative 27 passed")
test_align_derivative_28()
print("test align_derivative 28 passed")
test_align_derivative_29()
print("test align_derivative 29 passed")
test_align_derivative_30()
print("test align_derivative 30 passed")


def test_align_derivative_31() -> None:

    # Test case:
    # 1. differently distributed input independent dims

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 4, 5, 7, 5, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 4, 5, 7, 5, 6), deriv_shape

    return None


def test_align_derivative_32() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. permute all dimensions

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 4, 5, 7, 5, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 4, 5, 7, 5, 6), deriv_shape

    return None


def test_align_derivative_33() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. distribute independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 4, 5, 7, 5, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 4, 5, 6, 4, 5, 7, 4, 5, 6), deriv_shape

    return None


def test_align_derivative_34() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. permute all dimensions
    # 3. distribute independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 4, 5, 6, 4, 5, 7, 5, 6)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 5, 4, 6, 4, 5, 7, 5, 4, 6), deriv_shape

    return None


test_align_derivative_31()
print("test align_derivative 31 passed")
test_align_derivative_32()
print("test align_derivative 32 passed")
test_align_derivative_33()
print("test align_derivative 33 passed")
test_align_derivative_34()
print("test align_derivative 34 passed")

### KEEPDIMS TESTS


def test_align_derivative_35() -> None:

    # Test case:
    # no distribution | 1 -> 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 1, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 4, 5, 4, 6, 4, 5), deriv_shape

    return None


def test_align_derivative_36() -> None:

    # Test case:
    # no distribution | not 1 -> 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 2, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=tuple(range(len(variables))),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=True,
        )
        raise ValueError
    except AssertionError:
        pass

    # ### Checks
    # deriv_shape: Tuple[int, ...] = modified_derivative.shape
    # assert deriv_shape == (XX, 1, 4, 5, 4, 6, 4, 5), deriv_shape

    return None


def test_align_derivative_37() -> None:

    # Test case:
    # no distribution | not 1 -> not 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 2, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 2, 4, 5, 4, 6, 4, 5), deriv_shape

    return None


def test_align_derivative_38() -> None:

    # Test case:
    # distribution | 1 -> 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 1, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 1, 4, 5, 1, 4, 6, 1, 4, 5), deriv_shape

    return None


def test_align_derivative_39() -> None:

    # Test case:
    # distribution | not 1 -> 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 2, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 1, 4, 5, 1, 4, 6, 1, 4, 5), deriv_shape

    return None


def test_align_derivative_40() -> None:

    # Test case:
    # distribution | not 1 -> not 1

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, 2, 4, 5, 4, 6, 4, 5)
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, 1, 2, 4, 5, 2, 4, 6, 2, 4, 5), deriv_shape

    return None


test_align_derivative_35()
print("test align_derivative 35 passed")
test_align_derivative_36()
print("test align_derivative 36 passed")
test_align_derivative_37()
print("test align_derivative 37 passed")
test_align_derivative_38()
print("test align_derivative 38 passed")
test_align_derivative_39()
print("test align_derivative 39 passed")
test_align_derivative_40()
print("test align_derivative 40 passed")


def test_align_derivative_41() -> None:

    # Test case:
    # distribution  | (False, True)
    # sizes         | (not 1, not 1) -> (not 1, 1)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 24
    derivative_view: Tuple[int, ...] = (XX, *(4, 6))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 6), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 6), (1, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (None, 1))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6), *(1,)), deriv_shape

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6), *(1,)), deriv_shape

    return None


def test_align_derivative_42() -> None:

    # Test case:
    # distribution  | (False, True)
    # sizes         | (not 1, not 1) -> (not 1, 1)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 24
    derivative_view: Tuple[int, ...] = (XX, *(4, 6))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 6), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 6), (6,))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (None, 0))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6)), deriv_shape

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=tuple(range(len(variables))),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6)), deriv_shape

    return None


test_align_derivative_41()
print("test align_derivative 41 passed")
test_align_derivative_42()
print("test align_derivative 42 passed")


def test_align_derivative_43() -> None:

    # Test case: no modifications

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(9, 12), *(9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(9, 12), *(9, 12)), deriv_shape

    return None


def test_align_derivative_44() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_derivative
    modified_derivative: Tensor = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6, 7), *(4, 9, 12), *(4, 9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(4, 6, 7),
        *(4, 9, 12),
        *(4, 9, 12),
    ), deriv_shape

    return None


def test_align_derivative_45() -> None:

    # Test case:
    #   1. permute non independent dimensions

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 12, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(12, 9), *(12, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(12, 9), *(12, 9)), deriv_shape

    return None


def test_align_derivative_46() -> None:

    # Test case:
    #   1. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1,), (0,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(12, 9), *(12, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4,), *(6, 7), *(12, 9), *(12, 9)), deriv_shape

    return None


def test_align_derivative_47() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. permute all dimensions (including independent dimension)

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6, 7), *(12, 4, 9), *(12, 4, 9)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(4, 6, 7),
        *(12, 4, 9),
        *(12, 4, 9),
    ), deriv_shape

    return None


def test_align_derivative_48() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(4, 6, 7), *(9, 12), *(9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1,), *(4, 6, 7), *(9, 12), *(9, 12)), deriv_shape

    return None


def test_align_derivative_49() -> None:

    # Test case:
    #   1. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=(1, 2, 0),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=(1, 2, 0),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    return None


def test_align_derivative_50() -> None:

    # Test case:
    #   1. partial collapse independent dimension
    #   2. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=(1, 2, 0),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    ### Call align_derivative
    try:
        align_derivative(
            derivative=external_derivative,
            variables=variables,
            variable_perm=(1, 2, 0),
            shapes=shapes,
            indeps=indeps,
            expected_shapes=expected_shapes,
            expected_indeps=expected_indeps,
            keepdim=False,
        )
        raise ValueError
    except AssertionError:
        pass

    return None


def test_align_derivative_51() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 9, 12), (1, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 6, 7), *(1, 9, 12), *(1, 9, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(1, 6, 7),
        *(1, 9, 12),
        *(1, 9, 12),
    ), deriv_shape

    return None


def test_align_derivative_52() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension
    #   3. partial colapse non-independent dimension

    ### Define align_derivative inputs
    # derivative data
    XX: int = 1
    derivative_view: Tuple[int, ...] = (XX, *(4,), *(9, 12), *(6, 7), *(9, 12))
    external_derivative: Tensor = torch.rand(size=derivative_view, device=device)
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_derivative requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 12), (1, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    # typing
    modified_derivative: Tensor

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (XX, *(1, 6, 7), *(1, 1, 12), *(1, 1, 12)), deriv_shape

    ### Call align_derivative
    modified_derivative = align_derivative(
        derivative=external_derivative,
        variables=variables,
        variable_perm=(1, 2, 0),
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    deriv_shape: Tuple[int, ...] = modified_derivative.shape
    assert deriv_shape == (
        XX,
        *(1,),
        *(1, 6, 7),
        *(1, 1, 12),
        *(1, 1, 12),
    ), deriv_shape

    return None
