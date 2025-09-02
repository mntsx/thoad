# Standard Library dependencies
from typing import Tuple

# Pytorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing import Shape


def denull_shape(shape: Shape) -> Tuple[int, ...]:
    shape = shape if len(shape) > 0 else (1,)
    return tuple(max(1, size) for size in shape)


def denull_tensor(tensor: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    if tensor is not None:
        shape: Shape = tuple(tensor.shape)
        if 0 in shape:
            shape: Shape = denull_shape(shape=shape)
            tensor = torch.zeros(size=shape, dtype=dtype, device=device)
        if len(shape) == 0:
            tensor = torch.full(
                fill_value=tensor.item(), size=(1,), dtype=dtype, device=device
            )
    return tensor
