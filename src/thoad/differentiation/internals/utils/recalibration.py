import operator
from typing import Any, Sequence, SupportsIndex


def recalibrate_dim(dim: SupportsIndex, shape: Sequence[Any]) -> int:
    """
    Normalize `dim` to a valid non-negative axis in [0, len(shape)),
    accepting Python ints, NumPy ints, PyTorch ints, and unsigned encodings
    of negatives coming from C/C++ (e.g., 2**64-1 for -1).
    """
    d: int = operator.index(dim)  # robust int extraction
    n: int = len(shape)

    # Fast paths: already valid, or normal Python negative
    if 0 <= d < n:
        return d
    if d < 0:
        d += n
        if 0 <= d < n:
            return d

    # Heuristic: decode common unsigned encodings (uint64 then uint32)
    for bits in (64, 32):
        base: int = 1 << bits
        max_pos: int = (base >> 1) - 1  # largest signed positive
        if max_pos < d <= base - 1:
            d = d - base  # reinterpret as signed (negative)
            break

    # Re-check after decoding two's complement
    if d < 0:
        d += n
    if 0 <= d < n:
        return d

    raise IndexError(f"dim {d} out of range for rank {n}")


def recalibrate_index(index: int, dim_size: int) -> int:
    max_idx: int = 2**32
    if index > dim_size:
        if dim_size - index > 0:
            raise ValueError(
                f"Index {index} is ambiguous in grad_fn context data. "
                f"Consider using [{index}: ({index} + 1)] instead."
            )
        else:
            index = dim_size + index - max_idx
    return index
