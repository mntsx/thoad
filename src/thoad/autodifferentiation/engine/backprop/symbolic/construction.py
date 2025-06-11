# Standard Library Dependencies
import math
import itertools
from typing import Iterator, Tuple

# Internal dependencies
from thoad.autodifferentiation.engine.backprop.symbolic.structure import (
    Partial,
    ProductGroup,
    SumGroup,
)


# =============================================================================
# GENERATOR: generate_ordered_partitions
# =============================================================================
def generate_ordered_partitions(
    elements: list[int], sizes: list[int]
) -> Iterator[list[list[int]]]:
    """Yield all ordered partitions of *elements* into blocks with the
    prescribed *sizes* sequence.
    """
    if not sizes:
        if not elements:
            yield []
        return
    k: int = sizes[0]
    for combo in itertools.combinations(elements, k):
        remaining = list(elements)
        for x in combo:
            remaining.remove(x)
        for tail in generate_ordered_partitions(remaining, sizes[1:]):
            yield [list(combo)] + tail


# =============================================================================
# FUNCTION: raw_n_derivate
# =============================================================================
def raw_n_derivate(n: int) -> SumGroup:
    """Faà di Bruno (univariate) derivative of F(X)=A(B(X)) up to order *n*."""
    X = Partial("X")  # independent variable
    B = Partial("B", input=X)  # B depends on X
    A = Partial("A", input=B)  # A depends on B

    if n == 0:
        return SumGroup(ProductGroup(A))

    # Helper to enumerate integer partitions “à la Faà di Bruno”.
    def generate_sequences(left: int, j: int, max_j: int) -> list[list[int]]:
        if j > max_j:
            return [[]] if left == 0 else []
        out = []
        for k in range(0, left // j + 1):
            for tail in generate_sequences(left - k * j, j + 1, max_j):
                out.append([k] + tail)
        return out

    seqs: list[list[int]] = generate_sequences(n, 1, n)
    total = SumGroup()

    for seq in seqs:
        m: int = sum(seq)  # total derivatives on A
        coeff: int = math.factorial(n)
        for j, m_j in enumerate(seq, start=1):
            coeff //= math.factorial(m_j) * (math.factorial(j) ** m_j)

        pg = ProductGroup()
        pg.coefficient = coeff
        pg.extend(Partial("A", input=B, order=m))  # outer factor

        # inner factors -- append in ascending *order* (B(1) … B(2)…), but *without*
        # any later sorting so that we can re-order by first-index afterwards.
        for j, m_j in enumerate(seq, start=1):
            for _ in range(m_j):
                pg.extend(Partial("B", input=X, order=j))

        total.products.append(pg)

    return total


# =============================================================================
# FUNCTION: separate_dims
# =============================================================================
def separate_dims(grouped: SumGroup) -> SumGroup:
    """Assign concrete tensor indices to each Partial and disentangle
    blocks with identical inner partials so that ordering is driven
    by the *first* index each derivative touches (not by the order of
    differentiation)."""
    out = SumGroup()

    for pg in grouped.products:

        # trivial case (n = 0)
        if len(pg.partials) <= 1:
            p: Partial = pg.partials[0]
            if p.dims is None:
                p.dims = list(range(p.order))
            out.products.append(pg)
            continue

        outer: Partial
        inner: list[Partial]
        outer, inner = pg.partials[0], pg.partials[1:]
        outer_dims = list(range(outer.order))  # canonical: [0, 1, …]

        n_inner_idx: int = sum(p.order for p in inner)
        full_set = list(range(n_inner_idx))
        sizes: list[int] = [p.order for p in inner]

        #  generate *all* possible ordered partitions of index set
        parts = list(generate_ordered_partitions(full_set, sizes))

        #  deduplicate across blocks of identical derivative order
        #   (so that e.g. B(1)×B(1) is insensitive to swapping).
        unique: dict[tuple, list[list[int]]] = {}
        blocks: list[Tuple[int, int]] = []
        cur_ord, cnt = sizes[0], 0
        for s in sizes:
            if s == cur_ord:
                cnt += 1
            else:
                blocks.append((cur_ord, cnt))
                cur_ord, cnt = s, 1
        blocks.append((cur_ord, cnt))

        for part in parts:
            key_slices = []
            idx = 0
            for o, c in blocks:
                block = tuple(sorted(tuple(sorted(v)) for v in part[idx : idx + c]))
                key_slices.append(block)
                idx += c
            key = tuple(key_slices)
            unique.setdefault(key, part)  # keep first occurrence

        #  build one ProductGroup per unique assignment
        for part in unique.values():
            new_pg = ProductGroup()
            new_pg.coefficient = 1

            new_outer = Partial(outer.name, outer.input, outer.order)
            new_outer.dims = outer_dims
            new_pg.extend(new_outer)

            # create inner partials, then *re-order* by first index touched
            tmp: list[Tuple[int, Partial]] = []
            for dims, orig in zip(part, inner):
                q = Partial(orig.name, orig.input, orig.order)
                q.dims = sorted(dims)
                tmp.append((q.dims[0], q))  # key = smallest index

            tmp.sort(key=lambda kv: kv[0])  # order drives final print
            for _, q in tmp:
                new_pg.extend(q)

            out.products.append(new_pg)

    return out


# =============================================================================
# FUNCTION: assemble_symbolic_composition
# =============================================================================
def assemble_symbolic_composition(order: int) -> SumGroup:
    """User-facing helper: Faà di Bruno → index assignment."""
    return separate_dims(raw_n_derivate(order))
