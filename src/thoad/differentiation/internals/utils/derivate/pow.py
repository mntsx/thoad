# Standard Library dependencies
import math
from typing import Iterable, Tuple

# Pytorch dependencies
import torch
from torch import Tensor


def pow0_derivate(tensor: Tensor, exponent: float, order: int) -> Tensor:
    """
    Compute the `order`-th derivative of x**exponent, elementwise
    on `tensor`.  Returns a flattened Tensor of shape (tensor.numel(),).

    d^order/dx^order [x^exponent] =
      (exponent)(exponent-1)…(exponent-order+1) * x**(exponent-order)
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    # flatten so we match the original behaviour
    x: Tensor = tensor.flatten()
    # trivial 0-th derivative
    if order == 0:
        return tensor.pow(exponent)
    # falling-factorial coefficient: exponent * (exponent-1) * … * (exponent-order+1)
    coeff: float = math.prod(exponent - i for i in range(order))
    x_pow: Tensor = x.pow(exponent - order)
    x_mul: Tensor = x_pow.mul_(coeff)
    return x_mul.view_as(other=tensor)


def pow1_derivate(
    base: Tensor,
    exponent: Tensor,
    derivations: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Computes cross derivative of element-wise power x^y
    Args:
        base (Tensor): base elements
        exponent (Tensor): exponent elements
        derivations (Tuple[int,...]): sequence of 0 (w.r.t. base) or 1 (w.r.t. exponent)
    Returns:
        Tensor: result of mixed partial derivative
    """
    assert base.shape == exponent.shape
    assert all(i in (0, 1) for i in derivations)

    n: int = len(derivations)
    f: Tensor = base**exponent
    if n == 0:
        return f

    def partial_E(r: int, s: int) -> Tensor:
        # returns ∂^(r,s) E where E = y*ln(x)
        if s > 1:
            return torch.zeros_like(base, dtype=dtype, device=device)
        if s == 1:
            if r == 0:
                return torch.log(base)
            return ((-1) ** (r - 1) * math.factorial(r - 1)) * base.pow(-r)
        # s == 0
        if r == 0:
            return torch.zeros_like(base, dtype=dtype, device=device)
        return ((-1) ** (r - 1) * math.factorial(r - 1)) * exponent * base.pow(-r)

    def gen_parts(seq: list[int]) -> Iterable[list[list[int]]]:
        # generate all set partitions of seq (as lists of blocks)
        if not seq:
            yield []
            return
        first: int = seq[0]
        for part in gen_parts(seq[1:]):
            # start new block with `first`
            yield [[first]] + [blk[:] for blk in part]
            # insert `first` into each existing block
            for idx, blk in enumerate(part):
                new_part = [b[:] for b in part]
                new_part[idx] = blk + [first]
                yield new_part

    # deduplicate partitions (order-insensitive) via canonical tuple form
    seq: list[int] = list(range(n))
    seen: set[Tuple[Tuple[int, ...], ...]] = set()
    parts: list[list[list[int]]] = []
    for p in gen_parts(seq):
        blks_t: Tuple[Tuple[int, ...], ...] = tuple(tuple(sorted(b)) for b in p)
        blks_sorted: Tuple[Tuple[int, ...], ...]
        blks_sorted = tuple(sorted(blks_t, key=lambda b: b[0]))
        if blks_sorted in seen:
            continue
        seen.add(blks_sorted)
        parts.append([list(b) for b in blks_sorted])

    Y: Tensor = torch.zeros_like(base, dtype=dtype, device=device)
    for blk in parts:
        prod: Tensor = torch.ones_like(base, dtype=dtype, device=device)
        for b in blk:
            r: int = sum(1 for i in b if derivations[i] == 0)
            s: int = len(b) - r
            prod = prod * partial_E(r, s)
        Y = Y + prod

    return f * Y


def pow1_base_derivate(
    input: Tensor,
    exponent: Tensor,
    order: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    D^order_x x^E = (E)(E-1)…(E-order+1) * x^(E-order)
    computed in-place and fully vectorized.
    """
    # flatten for simplicity
    x_flat: Tensor = input.flatten().to(device=device)
    E_flat: Tensor = exponent.flatten().to(dtype=dtype, device=device)
    # build tensor [0,1,…,order-1] on device
    k: Tensor = torch.arange(order, dtype=dtype, device=device)
    # shape (order, N): each row is E - k_i
    falling: Tensor = E_flat.unsqueeze(0) - k.unsqueeze(1)
    # in-place cumprod along dim=0 yields row i = prod_{j=0..i}(E-j)
    coeff: Tensor = falling.cumprod_(dim=0)[-1]  # shape (N,)
    # exponent for x
    exp_pow: Tensor = E_flat - order
    # compute x^(E-order) in-place
    x_pow: Tensor = x_flat.pow(exp_pow)
    # multiply in-place by coeff
    x_mul: Tensor = x_pow.mul_(coeff)

    return x_mul.view_as(input)


def pow1_exponent_derivate(
    input: Tensor,
    output: Tensor,
    order: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Compute the `order`-th derivative of x^E w.r.t. E,
    given:
      input: x,
      output: x**E,
    i.e. D^order_E [ x^E ] = x^E * (ln x)^order
    """
    # flatten & cast
    x: Tensor = input.flatten().to(dtype=dtype, device=device)
    y: Tensor = output.flatten().to(dtype=dtype, device=device)
    # 0th derivative is just the function itself
    if order == 0:
        return y.view_as(input)
    # ln(x)
    ln_x: Tensor = x.log()
    # nth derivative = y * (ln x)^order
    d_n: Tensor = y.mul(ln_x.pow_(order))
    return d_n.view_as(input)
