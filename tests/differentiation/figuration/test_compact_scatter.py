# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.figuration import (
    compact_derivative,
    scatter_derivative,
)


def test_compact_scatter_01() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 1, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((9, 12), (6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    derivative_1: Tensor = compact_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_1.shape == (XX, 1, 108, 42, 108)

    derivative_2: Tensor = scatter_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_2.shape == derivative_0.shape
    return None


def test_compact_scatter_02() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (4, 6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (0,))

    derivative_1: Tensor = compact_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_1.shape == (XX, 4, 108, 42, 108)

    derivative_2: Tensor = scatter_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_2.shape == derivative_0.shape
    return None


def test_compact_scatter_03() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 4, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))

    derivative_1: Tensor = compact_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_1.shape == (XX, 4, 108, 42, 108)

    derivative_2: Tensor = scatter_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_2.shape == derivative_0.shape
    return None


def test_compact_scatter_04() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 4, 108, 42, 108))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 9, 12), (6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0,), (None,))

    derivative_1: Tensor = scatter_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_1.shape == (XX, 4, 9, 12, 6, 7, 9, 12), derivative_1.shape

    derivative_2: Tensor = compact_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_2.shape == derivative_0.shape
    return None


def test_compact_scatter_05() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 4, 5, 108, 42, 108))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((4, 5, 9, 12), (6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, 1), (None, None))

    derivative_1: Tensor = scatter_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_1.shape == (XX, 4, 5, 9, 12, 6, 7, 9, 12), derivative_1.shape

    derivative_2: Tensor = compact_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=False,
    )
    assert derivative_2.shape == derivative_0.shape
    return None


def test_compact_scatter_06() -> None:

    XX: int = 1
    derivative_0: Tensor = torch.rand(size=(XX, 9, 12, 6, 7, 9, 12))
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((9, 12), (6, 7))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((None,), (None,))

    derivative_1: Tensor = compact_derivative(
        derivative=derivative_0,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=True,
    )
    assert derivative_1.shape == (XX, 108, 42, 108)

    derivative_2: Tensor = scatter_derivative(
        derivative=derivative_1,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        indeps_squeezed=True,
    )
    assert derivative_2.shape == derivative_0.shape
    return None
