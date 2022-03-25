"""Provide the basic API for spliting the domain."""
from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, Type
from .typing import ArrayType


def _validate_splitting_axis(axis: int) -> None:
    """Allow splitting only along the horizontal axis."""
    # valid_min_axis = 1
    # ndim = 3
    # # if not (axis >= valid_min_axis or (axis >= valid_min_axis - ndim and axis < 0)):
    #     raise ValueError("You cannot split along the first dimension.")
    ...


class SplitVisitorBase(Generic[ArrayType]):
    """SplitVisitor base class defines methods for splitting arrays.

    Classes implementing this interface are used as arguments to `split`
    methods of classes implementing the `Splittable` interface.
    """

    _validate_axis = staticmethod(_validate_splitting_axis)

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)

    @abstractmethod
    def __hash__(self):  # pragma: no cover
        """Return hash based on number of parts and dimension."""
        return 1

    @abstractmethod
    def split_array(
        self, array: ArrayType
    ) -> tuple[ArrayType, ...]:  # pragma: no cover
        """Split numpy array in various parts."""
        ...

    @property
    @abstractmethod
    def parts(self) -> int:  # pragma: no cover
        """Return number of splits."""
        ...


class MergeVisitorBase(Generic[ArrayType]):
    """MergeVisitor base class defines methods for merging arrays.

    Classes implementing this interface are used as arguments to
    `merge` methods of classes implementing the `Splittable` interface.
    """

    _validate_axis = staticmethod(_validate_splitting_axis)

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)

    @abstractmethod
    def __hash__(self):  # pragma: no cover
        """Return hash based on number of parts and dimension."""
        return 1

    @abstractmethod
    def merge_array(self, arrays: Sequence[ArrayType]) -> ArrayType:  # pragma: no cover
        """Merge numpy array in various parts."""
        ...


class RegularSplitMergerBase(SplitVisitorBase[ArrayType], MergeVisitorBase[ArrayType]):
    """Implements splitting and merging into regular grid along single dimension."""

    __slots__ = ["dim", "_parts"]

    def __init__(self, parts: int, dim: tuple[int]):
        """Initialize class instance."""
        self._validate_axis(dim[0])
        self._parts = parts
        self.dim = dim

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return self._parts

    def __hash__(self):
        """Return hash based on number of parts and dimension."""
        return hash((self.parts, self.dim))


SplitableType = TypeVar("SplitableType", bound="Splitable")


class Splitable(ABC):
    """Splitable class has methods for splitting and merging its instances."""

    @abstractmethod
    def split(
        self: SplitableType, splitter: SplitVisitorBase
    ) -> tuple[SplitableType, ...]:  # pragma: no cover
        """Split the Domain into given number of parts along axis given by dim.

        For splitting among more than one axis pass tuple as dim.
        """
        ...

    @classmethod
    @abstractmethod
    def merge(
        cls: Type[SplitableType],
        others: Sequence[SplitableType],
        merger: MergeVisitorBase,
    ) -> SplitableType:  # pragma: no cover
        """Merge multiple Domains into one new domain."""
        ...
