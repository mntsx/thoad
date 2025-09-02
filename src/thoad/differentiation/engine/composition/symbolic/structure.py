from typing import Optional, List, Tuple, Union, Iterator
import math
import itertools


# =============================================================================
# CLASS: Partial
# =============================================================================
class Partial:
    """Represents a univariate partial-derivative component with (optional)
    tensor-contraction index information.
    """

    def __init__(
        self, name: str, input: Optional["Partial"] = None, order: int = 0
    ) -> None:
        self._name: str = name
        self._order: int = order
        self._input: Optional["Partial"] = input
        self._dims: Optional[List[int]] = None  # unordered list of indices

    # ──────────────────────────────────────────────────────────────────────
    # Pretty-printing helpers
    # ──────────────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        if self._dims is not None:
            dims_str = "{" + ",".join(str(d) for d in sorted(self._dims)) + "}"
            return f"{self._name}({self._order}:{dims_str})"
        return f"{self._name}({self._order})"

    # ──────────────────────────────────────────────────────────────────────
    # Public read-only views
    # ──────────────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._name

    @property
    def order(self) -> int:
        return self._order

    @property
    def input(self) -> Optional["Partial"]:
        return self._input

    @property
    def dims(self) -> list[int]:
        assert self._dims is not None
        return self._dims

    # ──────────────────────────────────────────────────────────────────────
    # Mutator
    # ──────────────────────────────────────────────────────────────────────
    @dims.setter
    def dims(self, d: list[int]) -> None:
        self._dims = d


# =============================================================================
# CLASS: ProductGroup
# =============================================================================
class ProductGroup:
    """Ordered product of Partial objects (no automatic re-sorting)."""

    def __init__(self, partial: Optional[Partial] = None) -> None:
        self._partials: List[Partial] = [] if partial is None else [partial]
        self._coefficient: int = 1

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._partials)

    def __str__(self) -> str:
        coeff: str = "" if self._coefficient == 1 else f"{self._coefficient}[ "
        tail: str = " ".join(str(p) for p in self._partials)
        suffix: str = "" if self._coefficient == 1 else " ]"
        return f"{coeff}{tail}{suffix}"

    # ──────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────
    @property
    def partials(self) -> List[Partial]:
        return self._partials

    @property
    def coefficient(self) -> int:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: int) -> None:
        self._coefficient = value

    def extend(self, thing: Union["ProductGroup", Partial]) -> None:
        if isinstance(thing, ProductGroup):
            self._partials.extend(thing.partials)
        elif isinstance(thing, Partial):
            self._partials.append(thing)
        else:
            raise TypeError("extend() requires a ProductGroup or Partial")


# =============================================================================
# CLASS: SumGroup
# =============================================================================
class SumGroup:
    """Linear sum of ProductGroups (keeps insertion order)."""

    def __init__(self, product: Optional[ProductGroup] = None) -> None:
        self._products: List[ProductGroup] = [] if product is None else [product]

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._products)

    def __str__(self) -> str:
        parts: list[str] = []
        for prod in self._products:
            partials: str = " ".join(str(p) for p in prod.partials)
            if prod.coefficient == 1:
                parts.append(partials)
            else:
                parts.append(f"{prod.coefficient}[ {partials} ]")
        return " + ".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    @property
    def products(self) -> List[ProductGroup]:
        return self._products

    def extend(self, other: "SumGroup") -> None:
        self._products.extend(other.products)
