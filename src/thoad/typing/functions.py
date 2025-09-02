# Standard Libray dependencies
from typing import Callable

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing.derivatives import EDData


type AutogradFunction = torch.autograd.graph.Node
type Hook = Callable[[EDData, dict[AutogradFunction, set[Tensor]]], EDData]
