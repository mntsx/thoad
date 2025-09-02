# Standard Library dependencies
from typing import Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast


def test_infer_broadcast_01() -> None:

    shapes: list[Tuple[int, ...]]
    shapes = [(4, 5), (6, 4, 5), (1, 4, 5), (5,)]
    broadcast_shape: Tuple[int, ...] = infer_broadcast(shapes=shapes)

    assert broadcast_shape == (6, 4, 5)

    return None


def test_infer_broadcast_02() -> None:

    shapes: list[Tuple[int, ...]]
    shapes = [(4, 5), (6, 4, 5), (3, 4, 5), (5,)]
    try:
        infer_broadcast(shapes=shapes)
        raise RuntimeError
    except AssertionError:
        pass
    return None
