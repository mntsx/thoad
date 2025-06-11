from typing import Sequence, Tuple, Union
import torch
from torch import Tensor

type Shape = Tuple[int, ...]
type Indep = Tuple[bool, ...]
type Notation = Union[None, Sequence[Union[int, Sequence[int]]]]
type EDData = Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]
type IDData = Tuple[Tensor, Notation]

type AutogradFunction = torch.autograd.Function
