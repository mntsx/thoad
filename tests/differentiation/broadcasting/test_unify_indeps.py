from thoad.differentiation.engine.broadcasting.alignment import unify_indeps
from thoad.typing import Indep


def test_unify_indeps_01() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (0, None, 1)
    indep2: Indep = (0, None, 1)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 2),
        inclusive=False,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 2, 2),
        inclusive=False,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_02() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (0, None, None)
    indep2: Indep = (None, None, 1)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 2),
        inclusive=False,
    )
    assert unified_indep == (0, None, None)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 2, 2),
        inclusive=False,
    )
    assert unified_indep == (None, None, None)
    return None


def test_unify_indeps_03() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (0, None, 1)
    indep2: Indep = (0, None, 1)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 2),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 2, 2),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_04() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (0, None, None)
    indep2: Indep = (None, None, 1)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 2),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 2, 2),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_05() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (0, None, None)
    indep2: Indep = (1, None, 1)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 2),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    try:
        unify_indeps(
            indeps=(indep0, indep1, indep2),
            shapes_ndims=(2, 2, 2),
            inclusive=True,
        )
        raise Exception
    except AssertionError:
        pass
    return None


def test_unify_indeps_06() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (1, None, 2)
    indep2: Indep = (2, None, 3)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 3),
        inclusive=False,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 3, 4),
        inclusive=False,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_07() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (1, None, None)
    indep2: Indep = (None, None, 3)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 3),
        inclusive=False,
    )
    assert unified_indep == (0, None, None)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 3, 4),
        inclusive=False,
    )
    assert unified_indep == (None, None, None)
    return None


def test_unify_indeps_08() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (1, None, 2)
    indep2: Indep = (2, None, 3)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 3),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 3, 4),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_09() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (1, None, None)
    indep2: Indep = (None, None, 3)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1),
        shapes_ndims=(2, 3),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    unified_indep = unify_indeps(
        indeps=(indep0, indep1, indep2),
        shapes_ndims=(2, 3, 4),
        inclusive=True,
    )
    assert unified_indep == (0, None, 1)
    return None


def test_unify_indeps_10() -> None:
    indep0: Indep = (0, None, 1)
    indep1: Indep = (1, None, None)
    indep2: Indep = (2, None, 2)
    unified_indep: Indep
    unified_indep = unify_indeps(
        indeps=(indep0, indep1), shapes_ndims=(2, 3), inclusive=True
    )
    assert unified_indep == (0, None, 1)
    try:
        unify_indeps(
            indeps=(indep0, indep1, indep2),
            shapes_ndims=(2, 3, 4),
            inclusive=True,
        )
        raise Exception
    except AssertionError:
        pass
    return None
