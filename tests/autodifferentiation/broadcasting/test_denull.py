# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.engine.broadcasting.figuration import (
    denull_differential,
)
from thoad.typing.data import Shape, Indep


def test_denull_01() -> None:

    # Test:
    # 1. not null differential

    # define sizes
    XX: int = 1
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...]] = ((3, 4), (5, 6))
    indeps: Tuple[Tuple[Union[None, int], ...]] = ((0, None), (None, 1))

    # build differential
    indep_shape: Tuple[int, ...] = (3, 6)
    distributed_sizes: Tuple[int, ...] = (4, 5, 4)
    differential: torch.Tensor = torch.randn((XX, *indep_shape, *distributed_sizes))
    dtype: torch.dtype = torch.float32
    device = torch.device("cpu")

    new_differential: Tensor
    new_shapes: list[Shape]
    new_indeps: list[Indep]
    new_differential, new_shapes, new_indeps = denull_differential(
        differential=differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        dtype=dtype,
        device=device,
    )
    assert new_differential.shape == (XX, 3, 6, 4, 5, 4), new_differential.shape
    assert new_shapes == shapes, new_shapes
    assert new_indeps == indeps, new_indeps
    return None


def test_denull_02() -> None:

    # Test:
    # 1. null differential

    # define sizes
    XX: int = 1
    variables: Tuple[int, ...] = (0, 1, 0)
    shapes: Tuple[Tuple[int, ...]] = ((3, 0), (5, 6))
    indeps: Tuple[Tuple[Union[None, int], ...]] = ((0, None), (None, 1))

    # build differential
    indep_shape: Tuple[int, ...] = (3, 6)
    distributed_sizes: Tuple[int, ...] = (0, 5, 0)
    differential: torch.Tensor = torch.randn((XX, *indep_shape, *distributed_sizes))
    dtype: torch.dtype = torch.float32
    device = torch.device("cpu")

    new_differential: Tensor
    new_shapes: list[Shape]
    new_indeps: list[Indep]
    new_differential, new_shapes, new_indeps = denull_differential(
        differential=differential,
        variables=variables,
        shapes=shapes,
        indeps=indeps,
        dtype=dtype,
        device=device,
    )
    assert new_differential.shape == (XX, 3, 6, 1, 5, 1), new_differential.shape
    assert torch.allclose(new_differential, torch.zeros(size=(1,)))
    assert new_shapes == ((3, 1), (5, 6)), new_shapes
    assert new_indeps == indeps, new_indeps
    return None
