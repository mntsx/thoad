# Standard Library Dependencies
from typing import Optional, Sequence, Tuple, Union


class SymIndex:

    def __init__(self) -> None:
        self._id: Union[None, int] = None
        self._size: Union[None, int] = None

    def assert_size(
        self,
        size: int,
        allow_broadcasting: Optional[bool] = False,
    ) -> None:
        if self._size is None:
            self._size = size
        else:
            if allow_broadcasting:
                if size == 1 or self._size == 1:
                    self._size = max(size, self._size)
                else:
                    assert self._size == size, (self._size, size)
            else:
                assert self._size == size, (self._size, size)
        return None

    @property
    def id(self) -> Union[None, int]:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = value
        return None

    @property
    def size(self) -> Union[None, int]:
        return self._size


class LinkedSymIndex(SymIndex):
    """
    A SymIndex that shares its size with another SymIndex.

    LinkedSymIndex wraps an existing SymIndex and ensures that any size
    assertions apply to the linked index rather than to this instance.
    """

    def __init__(self, sym_index: SymIndex) -> None:
        """
        Initialize a LinkedSymIndex.

        Args:
            sym_index (SymIndex):
                The SymIndex to link to. All size assertions and queries will be
                forwarded to this linked index.
        """
        super().__init__()
        self._link: SymIndex = sym_index

    def assert_size(self, size: int) -> None:
        """
        Assert that the linked SymIndex has the given size.

        If the linked SymIndex has no size yet, set it to `size`. Otherwise, verify
        that its existing size matches `size`.

        Args:
            size (int): The dimension size to assert on the linked index.
        """
        if self._link._size is None:
            self._link._size = size
        else:
            assert self._link._size == size
        return None

    @property
    def size(self) -> Union[None, int]:
        """
        Get the size of the linked SymIndex.

        Returns:
            int or None: The size of the linked SymIndex, or None if not set.
        """
        return self._link._size


def _symbolize_array(
    array: Sequence[int],
    sym_map: Optional[dict[int, SymIndex]] = None,
) -> Tuple[list[SymIndex], dict[int, SymIndex]]:

    ### checks
    assert isinstance(array, Sequence)
    assert all(isinstance(i, int) for i in array)
    assert isinstance(sym_map, (type(None), dict))
    if sym_map is not None:
        assert all(isinstance(i, int) for i in sym_map.keys())
        assert all(isinstance(sym, (type(None), SymIndex)) for sym in sym_map.values())

    ### map ints to symbolic indices
    mapper: dict[int:SymIndex] = dict()
    if sym_map is not None:
        for i, sym in sym_map.items():
            mapper[i] = sym
    symbolic_array: list = list()
    for i in array:
        assert isinstance(i, int)
        if i not in mapper:
            sym: SymIndex = SymIndex()
            mapper[i] = sym
        symbolic_array.append(mapper[i])

    return (symbolic_array, mapper)


def symbolize_notation(
    notation: Sequence[Sequence[int]],
) -> Tuple[
    list[list[SymIndex]],
    list[list[Union[None, SymIndex]]],
]:

    ### checks
    assert all(isinstance(ii, int) for i in notation for ii in i)

    ### Symbolize notation
    # with independent symbolic indices
    symbolic_notation: list[list[Union[None, SymIndex]]] = list()
    mapper: dict[int, SymIndex] = dict()
    for subnotation in notation:
        symbolic_subnotation: Tuple[Union[None, SymIndex], ...]
        symbolic_subnotation, mapper = _symbolize_array(
            array=subnotation,
            sym_map=mapper,
        )
        symbolic_notation.append(symbolic_subnotation)

    return symbolic_notation


def numerize_indices(nested_indices: list[list[list[SymIndex]]]) -> None:
    flat_indices: set[SymIndex] = set()
    for sub in nested_indices:
        for subsub in sub:
            for sym_idx in subsub:
                flat_indices.add(sym_idx)
    for i, sym_idx in enumerate(flat_indices):
        sym_idx.id = i
    return None
