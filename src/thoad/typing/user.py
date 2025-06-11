from typing import Callable, Optional, Tuple
from torch import Tensor

type Hook = Callable[
    [Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Optional[Tuple[Tensor, ...]],
]
