from typing import Tuple, Union
from torch import Tensor


type Shape = Tuple[int, ...]
type Indep = Tuple[Union[None, int], ...]
type VPerm = Tuple[int, ...]  # permutation for: (tensor, shapes, indeps) -> variables
type Metadata = Tuple[Tuple[Shape, ...], Tuple[Indep, ...], VPerm]
type EDData = Tuple[
    Union[None, Tensor],
    Union[None, Tuple[Shape, ...]],
    Union[None, Tuple[Indep, ...]],
    Union[None, VPerm],
]
type StaticEDData = Tuple[
    Union[None, Tensor],
    Union[None, Tuple[Shape, ...]],
    Union[None, Tuple[Indep, ...]],
]
type PopulatedEDData = Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...], VPerm]

type Notation = list[Tuple[Union[Tuple[bool, ...], Tuple[int, ...]], ...]]
type IDData = Tuple[Union[None, Tensor], Union[None, Notation]]
