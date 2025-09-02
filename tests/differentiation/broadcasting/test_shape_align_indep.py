from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.typing import Shape, Indep


def test_shape_align_indep_01() -> None:
    shape: Shape = (3, 4, 5)
    indep: Indep = (0, None, 1)
    expected_shape: Shape = (3, 4, 5)
    adjusted_indep: Indep = shape_align_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == indep
    return None


def test_shape_align_indep_02() -> None:
    shape: Shape = (3, 4, 5)
    indep: Indep = (0, None, 1)
    expected_shape: Shape = (3, 5, 4)
    adjusted_indep: Indep = shape_align_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (0, None, 2)
    return None


def test_shape_align_indep_03() -> None:
    shape: Shape = (3, 4, 5)
    indep: Indep = (0, None, 1)
    expected_shape: Shape = (3, 1, 5)
    adjusted_indep: Indep = shape_align_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (0, None, None)
    return None


def test_shape_align_indep_04() -> None:
    shape: Shape = (2, 3, 4, 5)
    indep: Indep = (0, None, 3)
    expected_shape: Shape = (4, 5)
    adjusted_indep: Indep = shape_align_indep(
        shape=shape, indep=indep, expected_shape=expected_shape
    )
    assert adjusted_indep == (None, None, 1), adjusted_indep
    return None
