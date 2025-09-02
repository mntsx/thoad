# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.figuration import denull_derivative


def test_denull_01() -> None:

    # Test:
    # 1. not null derivative

    # define sizes
    XX: int = 1
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((3, 4), (5, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, None), (None, 1))

    # build derivative
    indep_shape: Tuple[int, ...] = (3, 6)
    distributed_sizes: Tuple[int, ...] = (4, 5, 4)
    derivative: Tensor = torch.randn((XX, *indep_shape, *distributed_sizes))
    dtype: torch.dtype = torch.float32
    device = torch.device("cpu")

    new_derivative: Tensor
    new_shapes: Tuple[Tuple[int, ...], ...]
    new_indeps: Tuple[Tuple[Union[None, int], ...], ...]
    new_derivative, new_shapes, new_indeps = denull_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        dtype=dtype,
        device=device,
    )
    assert new_derivative.shape == (XX, 3, 6, 4, 5, 4), new_derivative.shape
    assert new_shapes == shapes, new_shapes
    assert new_indeps == indeps, new_indeps
    return None


def test_denull_02() -> None:

    # Test:
    # 1. null derivative

    # define sizes
    XX: int = 1
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...], ...] = ((3, 0), (5, 6))
    indeps: Tuple[Tuple[Union[None, int], ...], ...] = ((0, None), (None, 1))

    # build derivative
    indep_shape: Tuple[int, ...] = (3, 6)
    distributed_sizes: Tuple[int, ...] = (0, 5, 0)
    derivative: Tensor = torch.randn((XX, *indep_shape, *distributed_sizes))
    dtype: torch.dtype = torch.float32
    device = torch.device("cpu")

    new_derivative: Tensor
    new_shapes: Tuple[Tuple[int, ...], ...]
    new_indeps: Tuple[Tuple[Union[None, int], ...], ...]
    new_derivative, new_shapes, new_indeps = denull_derivative(
        derivative=derivative,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        dtype=dtype,
        device=device,
    )
    assert new_derivative.shape == (XX, 3, 6, 1, 5, 1), new_derivative.shape
    assert torch.allclose(new_derivative, torch.zeros(size=(1,)))
    assert new_shapes == ((3, 1), (5, 6)), new_shapes
    assert new_indeps == indeps, new_indeps
    return None
