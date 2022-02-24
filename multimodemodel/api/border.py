"""Provide API for spliting domain."""
from abc import abstractmethod
from enum import Enum
from typing import Generic, TypeVar, Type, Optional
from .core import (
    DomainBase,
    DomainType,
    StateDequeBase,
    StateType,
    ParameterType,
)
from .split import SplitVisitorBase, MergeVisitorBase
from .typing import ArrayType


def _left_border_slice(w: int) -> slice:
    return slice(None, w)


def _left_halo_border_slice(w: int) -> slice:
    return slice(w, 2 * w)


def _right_border_slice(w: int) -> slice:
    return slice(-w, None)


def _right_halo_border_slice(w: int) -> slice:
    return slice(-2 * w, -w)


def _center_border_slice(w1: int, w2: int) -> slice:
    return slice(w1, -w2)


class BorderDirection(Enum):
    """Enumerator of border possible directions."""

    LEFT = _left_border_slice
    LEFT_HALO = _left_halo_border_slice
    RIGHT = _right_border_slice
    RIGHT_HALO = _right_halo_border_slice
    CENTER = _center_border_slice


BorderSplitterType = TypeVar("BorderSplitterType", bound="BorderSplitterBase")


class BorderSplitterBase(SplitVisitorBase[ArrayType]):
    """Base class for border splitter."""

    __slots__ = ["_axis", "_slice"]

    def __init__(self, slice: slice, axis: int):
        """Initialize class instance."""
        self._validate_axis(axis)
        self._axis = axis
        self._slice = slice

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return 1

    def __hash__(self):
        """Return hash based on axis and slice object."""
        return hash((self._axis, self._slice.start, self._slice.stop))


BorderType = TypeVar("BorderType", bound="BorderBase")


class BorderBase(DomainBase[StateType, ParameterType]):
    """
    Base class for Border classes.

    Contains the state on the border and provides information about its
    width and dimension.
    """

    width: int
    dim: int

    __slots__ = DomainBase.__slots__ + ("width", "dim")

    def __init__(
        self,
        state: StateType,
        width: int,
        dim: int,
        iteration: int,
        history: Optional[StateDequeBase[StateType]] = None,
        parameter: Optional[ParameterType] = None,
        id: int = 0,
    ):
        """Create BorderState in the same way as DomainState."""
        super().__init__(
            state=state,
            history=history,
            parameter=parameter,
            iteration=iteration,
            id=id,
        )
        self.width = width
        self.dim = dim

    @classmethod
    def create_border(
        cls: Type[BorderType],
        base: DomainBase[StateType, ParameterType],
        width: int,
        direction: bool,
        dim: int,
    ) -> BorderType:
        """Split border from a Domain instance.

        The data of the boarder will be copied to avoid data races.
        """
        if direction:
            border_slice = BorderDirection.RIGHT(width)  # type: ignore
        else:
            border_slice = BorderDirection.LEFT(width)  # type: ignore
        splitter = cls._provide_border_splitter()(slice=border_slice, axis=dim)
        splitted_domain = base.split(splitter)[0]

        return cls.from_domain(splitted_domain, width=width, dim=dim)

    @staticmethod
    @abstractmethod
    def _provide_border_splitter() -> Type:
        """Return type of the associated border splitter class."""
        ...

    def get_width(self) -> int:
        """Provide border's width as int."""
        return self.width

    def get_dim(self) -> int:
        """Provide border's dimension as int."""
        return self.dim

    @classmethod
    def from_domain(
        cls: Type[BorderType],
        domain: DomainBase[StateType, ParameterType],
        width: int,
        dim: int,
    ) -> BorderType:
        """Create an instance from a domain instance.

        No copies are created.
        """
        return cls(
            state=domain.state,
            width=width,
            dim=dim,
            iteration=domain.iteration,
            history=domain.history,
            parameter=domain.parameter,
            id=domain.id,
        )


BorderMergerType = TypeVar("BorderMergerType", bound="BorderMergerBase")


class BorderMergerBase(MergeVisitorBase[ArrayType], Generic[ArrayType, BorderType]):
    """Base class for BorderMerger.

    Merges the borders with a Domain along a dimension.

    This merger is suppose to be used in the merge classmethod of the
    DomainState class.
    """

    __slots__ = ["_axis", "_slice_left", "_slice_right", "_slice_center", "_width"]

    def __init__(self, width: int, axis: int):
        """Initialize class instance."""
        self._validate_axis(axis)
        self._axis = axis
        self._slice_left = BorderDirection.LEFT(width)  # type: ignore
        self._slice_right = BorderDirection.RIGHT(width)  # type: ignore
        self._slice_center = BorderDirection.CENTER(width, width)  # type: ignore
        self._width = width

    @classmethod
    def from_borders(
        cls: Type[BorderMergerType], left_border: BorderType, right_border: BorderType
    ) -> BorderMergerType:
        """Create BorderMerger from left and right border instance."""
        assert left_border.width == right_border.width
        assert left_border.dim == right_border.dim
        return cls(width=left_border.width, axis=left_border.dim)

    def __hash__(self):
        """Return hash based on axis and slice objects."""
        return hash((self._axis, self._width))


class TailorBase(Generic[DomainType, BorderType]):
    """Tailor class keeps functions required for Border class."""

    @staticmethod
    @abstractmethod
    def _provide_border_type() -> Type:
        """Return type of the border instance associated with the domain class."""
        ...

    @staticmethod
    def split_domain(
        base: DomainType, splitter: SplitVisitorBase
    ) -> tuple[DomainType, ...]:
        """Split domain in subdomains.

        When splitting, the ids of the subdomains are set to `range(0, splitter.parts)`.
        """
        splitted = base.split(splitter)
        for i, s in enumerate(splitted):
            s.id = i
        return splitted

    def make_borders(
        self, base: DomainType, width: int, dim: int
    ) -> tuple[BorderType, BorderType]:
        """Implement make_borders method from API."""
        return (
            self._provide_border_type().create_border(base, width, False, dim),
            self._provide_border_type().create_border(base, width, True, dim),
        )

    @staticmethod
    @abstractmethod
    def stitch(
        base: DomainType,
        borders: tuple[BorderType, BorderType],
    ) -> DomainType:
        """Copy data from Border to Domain.

        Whether it mutates Domain object is up to implementation.
        The borders tuple contains left and right Border (in this order).
        Order of axis in the tuple is determined by order dims tuple used in
        magic function, which is passed to this function.
        """
        ...
