# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import (
    align_differential,
)


def test_align_differential_01() -> None:

    # Test case: no modifications

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 9, 12, 9, 12, 9, 12), diff_shape


def test_align_differential_02() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 9, 12, 4, 9, 12, 4, 9, 12), diff_shape


def test_align_differential_03() -> None:

    # Test case:
    #   1. permute non independent dimensions

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 12, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 12, 9, 12, 9, 12, 9), diff_shape


def test_align_differential_04() -> None:

    # Test case:
    #   1. permute all dimensions (including independent dimension)

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (1,),
    )  # ((False, True, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 12, 9, 12, 9, 12, 9), diff_shape


def test_align_differential_05() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. permute all dimensions (including independent dimension)

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 12, 4, 9, 12, 4, 9, 12, 4, 9), diff_shape


def test_align_differential_06() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 9, 12, 9, 12, 9, 12), diff_shape


def test_align_differential_07() -> None:

    # Test case:
    #   1. partial collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 9, 12, 9, 12, 9, 12), diff_shape


def test_align_differential_08() -> None:

    # Test case:
    #   1. partial collapse independent dimension
    #   2. partial colapse non-independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 3, 12, 3, 12, 3, 12), diff_shape


def test_align_differential_09() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 9, 12, 2, 9, 12, 2, 9, 12), diff_shape


def test_align_differential_10() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension
    #   3. partial colapse non-independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 9, 12, 9, 12))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 3, 12, 2, 3, 12, 2, 3, 12), diff_shape


def test_align_differential_11() -> None:

    # Test case: no modifications

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 9, 12, 6, 7, 9, 12), diff_shape


def test_align_differential_12() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
        (None,),
    )  # ((False, False, False), (False, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 9, 12, 4, 6, 7, 4, 9, 12), diff_shape


def test_align_differential_13() -> None:

    # Test case:
    #   1. permute non independent dimensions

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 12, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 12, 9, 6, 7, 12, 9), diff_shape


def test_align_differential_14() -> None:

    # Test case:
    #   1. permute all dimensions (including independent dimension)

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (1,),
        (0,),
    )  # ((False, True, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 12, 9, 6, 7, 12, 9), diff_shape


def test_align_differential_15() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. permute all dimensions (including independent dimension)

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((12, 4, 9), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
        (None,),
    )  # ((False, False, False), (False, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 12, 4, 9, 4, 6, 7, 12, 4, 9), diff_shape


def test_align_differential_16() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((9, 12), (4, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
        (None,),
    )  # ((False, False), (False, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 9, 12, 4, 6, 7, 9, 12), diff_shape


def test_align_differential_17() -> None:

    # Test case:
    #   1. partial collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 9, 12, 6, 7, 9, 12), diff_shape


def test_align_differential_18() -> None:

    # Test case:
    #   1. partial collapse independent dimension
    #   2. partial colapse non-independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 3, 12, 6, 7, 3, 12), diff_shape


def test_align_differential_19() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 9, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
        (None,),
    )  # ((False, False, False), (False, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 9, 12, 2, 6, 7, 2, 9, 12), diff_shape


def test_align_differential_20() -> None:

    # Test case:
    #   1. deindependice independent dimension
    #   2. partial collapse independent dimension
    #   3. partial colapse non-independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
        (0,),
    )  # ((True, False, False), (True, False, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 3, 12), (2, 6, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
        (None,),
    )  # ((False, False, False), (False, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 3, 12, 2, 6, 7, 2, 3, 12), diff_shape


def test_align_differential_21() -> None:

    # Test case: no modifications

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 1, 1, 1, 1, 1, 1))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 1, 1, 1, 1, 1, 1), diff_shape


def test_align_differential_22() -> None:

    # Test case:
    #   1. deindependice independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 1, 1, 1, 1, 1, 1))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 1, 1, 1, 1, 1, 1, 1, 1), diff_shape


def test_align_differential_23() -> None:

    # Test case:
    #   1. full collapse independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 1, 1, 1, 1, 1, 1))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 1, 1),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0,),
    )  # ((True, False, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 1),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None,),
    )  # ((False, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 1, 1, 1, 1, 1), diff_shape


def test_align_differential_24() -> None:

    # Test case:
    # 1. permutation in independent dimensions

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 6, 6))
    variables: Tuple[int, ...] = (0, 0, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6),)
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
    )  # ((True, True, False),)
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6),)
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (1, 0),
    )  # ((True, True, False),)

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 5, 4, 6, 6, 6), diff_shape


def test_align_differential_25() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 7, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
        (0, 1),
    )  # ((True, True, False), (True, True, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (1, 0),
        (1, 0),
    )  # ((True, True, False), (True, True, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 5, 4, 6, 7, 6), diff_shape


def test_align_differential_26() -> None:

    # Test case:
    # 1. distribution of 1 independent dimension for subset of variables

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 7, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
        (0, 1),
    )  # ((True, True, False), (True, True, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
        (0, None),
    )  # ((True, True, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 6, 5, 7, 6), diff_shape


def test_align_differential_27() -> None:

    # Test case:
    # 1. distribution of 2 independent dimensions for subset of variables

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 7, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
        (0, 1),
    )  # ((True, True, False), (True, True, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (None, 1),
        (0, None),
    )  # ((False, True, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 4, 6, 5, 7, 4, 6), diff_shape


def test_align_differential_28() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables
    # 2. distribution of 1 independent dimension for subset of variables

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 7, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (0, 1),
        (0, 1),
    )  # ((True, True, False), (True, True, False))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = (
        (1, 0),
        (1, None),
    )  # ((True, True, False), (True, False, False))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 5, 4, 6, 5, 7, 6), diff_shape


def test_align_differential_29() -> None:

    # Test case:
    # 1. permutation in independent dimensions with multiple variables
    # 2. distribution of 1 independent dimension for subset of variables

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 7, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (0, 1))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (5, 4, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None, 0), (1, None))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 4, 6, 5, 7, 4, 6), diff_shape


def test_align_differential_30() -> None:

    # Test case:
    # 1. input null independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 4, 6, 4, 5), diff_shape


def test_align_differential_31() -> None:

    # Test case:
    # 1. differently distributed input independent dims

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 4, 5, 7, 5, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 6, 4, 5, 7, 5, 6), diff_shape


def test_align_differential_32() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. permute all dimensions

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 4, 5, 7, 5, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((1,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 6, 4, 5, 7, 5, 6), diff_shape


def test_align_differential_33() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. distribute independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 4, 5, 7, 5, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 4, 5, 6, 4, 5, 7, 4, 5, 6), diff_shape


def test_align_differential_34() -> None:

    # Test case:
    # 1. differently distributed input independent dims
    # 2. permute all dimensions
    # 3. distribute independent dimension

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 4, 5, 6, 4, 5, 7, 5, 6))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 6), (4, 5, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((5, 4, 6), (4, 5, 7))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=False,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 5, 4, 6, 4, 5, 7, 5, 4, 6), diff_shape


### KEEPDIMS TESTS


def test_align_differential_35() -> None:

    # Test case:
    # no distribution | 1 -> 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((4, 5), (4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 4, 5, 4, 6, 4, 5), diff_shape


def test_align_differential_36() -> None:

    # Test case:
    # no distribution | not 1 -> 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 2, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 4, 5, 4, 6, 4, 5), diff_shape


def test_align_differential_37() -> None:

    # Test case:
    # no distribution | not 1 -> not 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 2, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 2, 4, 5, 4, 6, 4, 5), diff_shape


def test_align_differential_38() -> None:

    # Test case:
    # distribution | 1 -> 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 1, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 1, 4, 5, 1, 4, 6, 1, 4, 5), diff_shape


def test_align_differential_39() -> None:

    # Test case:
    # distribution | not 1 -> 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 2, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((1, 4, 5), (1, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 1, 4, 5, 1, 4, 6, 1, 4, 5), diff_shape


def test_align_differential_40() -> None:

    # Test case:
    # distribution | not 1 -> not 1

    ### Define align_differential inputs
    # differential data
    XX: int = 1
    external_differential: Tensor = torch.rand(size=(XX, 2, 4, 5, 4, 6, 4, 5))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))
    # align_differential requirements
    expected_shapes: Tuple[Tuple[int, ...], ...] = ((2, 4, 5), (2, 4, 6))
    expected_indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    ### Call align_differential
    modified_differential: Tensor = align_differential(
        differential=external_differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        expected_shapes=expected_shapes,
        expected_indeps=expected_indeps,
        keepdim=True,
    )

    ### Checks
    diff_shape: Tuple[int, ...] = modified_differential.shape
    assert diff_shape == (XX, 1, 2, 4, 5, 2, 4, 6, 2, 4, 5), diff_shape
