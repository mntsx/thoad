# Standard Library dependencies
from typing import Tuple, Union

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.alignment import (
    adjust_indep,
)


def test_adjust_indep_01() -> None:
    shape: Tuple[int, ...] = (3, 4, 5)
    indep: Tuple[Union[None, int], ...] = (0, None, 1)
    expected_shape: Tuple[int, ...] = (3, 4, 5)
    adjusted_indep: Tuple[Union[None, int], ...] = adjust_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == indep
    return None


def test_adjust_indep_02() -> None:
    shape: Tuple[int, ...] = (3, 4, 5)
    indep: Tuple[Union[None, int], ...] = (0, None, 1)
    expected_shape: Tuple[int, ...] = (3, 5, 4)
    adjusted_indep: Tuple[Union[None, int], ...] = adjust_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (0, None, 2)
    return None


def test_adjust_indep_03() -> None:
    shape: Tuple[int, ...] = (3, 4, 5)
    indep: Tuple[Union[None, int], ...] = (0, None, 1)
    expected_shape: Tuple[int, ...] = (3, 2, 5)
    adjusted_indep: Tuple[Union[None, int], ...] = adjust_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (0, None, None)
    return None


def test_adjust_indep_04() -> None:
    shape: Tuple[int, ...] = (2, 3, 4, 5)
    indep: Tuple[Union[None, int], ...] = (0, None, 1)
    expected_shape: Tuple[int, ...] = (3, 4, 5)
    adjusted_indep: Tuple[Union[None, int], ...] = adjust_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (None, None, 0), adjusted_indep
    return None
