"""Core API."""
from abc import abstractmethod
from dataclasses import dataclass, field
from collections import deque
from enum import unique, Enum
from typing import (
    Any,
    Generator,
    TypeVar,
    Generic,
    Optional,
    Type,
    Sequence,
)
from copy import deepcopy
import numpy as np

from multimodemodel.api.split import (
    Splitable,
    SplitVisitorBase,
    MergeVisitorBase,
)
from multimodemodel.util import add_time, average_npdatetime64
from .typing import Shape, ArrayType


@unique
class GridShift(Enum):
    """Direction of shift of staggered grids with respect to the eta-grid.

    E.g., `GridShift.LR` indicates that the grid points of the other grids which share
    the same index are located on the lower and/or left face of the eta Grid. The
    value of the enumerator is a tuple giving the direction of shift in
    y- and x-direction.
    """

    LR = (1, -1)  #: Subgrids are shifted to the lower right
    UR = (1, 1)  #: Subgrids are shifted to the upper right
    LL = (-1, -1)  #: Subgrids are shifted to the lower left
    UL = (-1, 1)  #: Subgrids are shifted to the upper left


GridType = TypeVar("GridType", bound="GridBase")


class GridBase(Splitable, Generic[ArrayType]):
    """Base class for all Grids.

    This class is generic w.r.t. the array type to store the data.
    """

    x: ArrayType
    y: ArrayType
    z: ArrayType
    mask: ArrayType
    dx: ArrayType
    dy: ArrayType
    dz: ArrayType

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> Shape:  # pragma: no cover
        """Return shape tuple of grid."""
        ...

    @property
    @abstractmethod
    def dim_x(self) -> int:  # pragma: no cover
        """Return axis of x dimension."""
        return -1

    @property
    @abstractmethod
    def dim_y(self) -> int:  # pragma: no cover
        """Return axis of x dimension."""
        return -2

    @property
    @abstractmethod
    def dim_z(self) -> int:  # pragma: no cover
        """Return axis of x dimension."""
        return -3

    @abstractmethod
    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        """Compare to Grid objects."""
        ...

    @classmethod
    @abstractmethod
    def cartesian(
        cls: Type[GridType],
        x: ArrayType,
        y: ArrayType,
        z: Optional[ArrayType],
        mask: Optional[ArrayType],
        **kwargs,
    ) -> GridType:
        """Generate a Cartesian grid.

        Arguments
        ---------
        x : Array
          1D Array of coordinates along x dimension.
        y : Array
          1D Array of coordinates along y dimension.
        z : Array, default=None
          1D Array of coordinates along z dimension.
        mask : Array, default=None
          Optional ocean mask. Default is a closed domain.
        """

    ...

    @classmethod
    @abstractmethod
    def regular_lat_lon(
        cls: Type[GridType],
        lon_start: float,
        lon_end: float,
        lat_start: float,
        lat_end: float,
        nx: int,
        ny: int,
        z: Optional[ArrayType] = None,
        mask: Optional[ArrayType] = None,
        radius: float = 6_371_000.0,
    ) -> GridType:
        """Generate a regular spherical grid.

        Arguments
        ---------
        lon_start : float
          Smallest longitude in degrees
        lon_end : float
          larges longitude in degrees
        lat_start : float
          Smallest latitude in degrees
        lat_end : float
          larges latitude in degrees
        nx : int
          Number of grid points along x dimension.
        ny : int
          Number of grid points along y dimension.
        z : Array, default=None
          Optional 1D coordinate array along vertical dimension.
        mask : Array, default=None
          Optional ocean mask. Default is a closed domain.
        radius : float, default=6_371_000.0
          Radius of the sphere, defaults to Earths' radius measured in meters.
        """
        ...


StaggeredGridType = TypeVar("StaggeredGridType", bound="StaggeredGridBase")


class StaggeredGridBase(Splitable, Generic[GridType]):
    """Base class for staggered Grid.

    Subgrids are available as attributes `eta`, `u`, `v` and `q`.
    This class is generic w.r.t. the type of the individual grids.
    """

    eta: GridType
    u: GridType
    v: GridType
    q: GridType
    _gtype: Type[GridType]

    __slots__ = ("eta", "u", "v", "q")

    def __init__(self, eta: GridType, u: GridType, v: GridType, q: GridType):
        """Initialize StaggeredGrid instance."""
        self.eta = eta
        self.u = u
        self.v = v
        self.q = q

    def items(self) -> Generator[tuple[str, GridType], None, None]:
        """Generate tuples of (grid_name, grid_obj)."""
        for g in self.__slots__:
            yield (g, getattr(self, g))

    @classmethod
    def _grid_type(cls):
        return cls._gtype

    def __eq__(self, other: Any) -> bool:
        """Compare to other staggered grid."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return all(getattr(self, g) == getattr(other, g) for g in self.__slots__)

    def split(self, splitter: SplitVisitorBase):
        """Split staggered grids."""
        splitted_grids = {name: g.split(splitter) for name, g in self.items()}
        return tuple(
            self.__class__(**{g: splitted_grids[g][i] for g in self.__slots__})
            for i in range(splitter.parts)
        )

    @classmethod
    def merge(cls, others: Sequence["StaggeredGridBase"], merger: MergeVisitorBase):
        """Merge staggered grids."""
        return cls(
            **{
                g: cls._grid_type().merge(tuple(getattr(o, g) for o in others), merger)
                for g in cls.__slots__
            }
        )

    @classmethod
    @abstractmethod
    def cartesian_c_grid(
        cls: Type[StaggeredGridType],
        shift: GridShift = GridShift.LL,
        **grid_kwargs: dict[str, Any],
    ) -> StaggeredGridType:  # pragma: no cover
        """Generate a Cartesian Arakawa C-Grid.

        Arguments
        ---------
        shift : GridShift, default=GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See :py:class:`GridShift` for more details.
        **grid_kwargs : dict[str, Any]
          Keyword arguments are passed to :py:meth:`GridBase.cartesian` to create
          the `eta` subgrid, i.e. the grid of the box centeroids.
        """
        ...

    @classmethod
    @abstractmethod
    def regular_lat_lon_c_grid(
        cls: Type[StaggeredGridType],
        shift: GridShift = GridShift.LL,
        **kwargs: dict[str, Any],
    ) -> StaggeredGridType:  # pragma: no cover
        """Generate a Arakawa C-grid for a regular longitude/latitude grid.

        Arguments
        ---------
        shift : GridShift, default=GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See :py:class:`GridShift` for more details.
        **grid_kwargs : dict[str, Any]
          Keyword arguments are passed to :py:meth:`GridBase.regular_lat_lon` to create
          the `eta` subgrid, i.e. the grid of the box centeroids.
        """
        ...


class ParameterBase(Splitable):
    """Base class for all Parameter classes."""

    @abstractmethod
    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        """Compare to parameter objects."""
        ...


ParameterType = TypeVar("ParameterType", bound=ParameterBase)

VariableType = TypeVar("VariableType", bound="VariableBase")


class VariableBase(Splitable, Generic[ArrayType, GridType]):
    """Base class for all Variable classes."""

    data: Optional[ArrayType]
    grid: GridType
    time: np.datetime64

    __slots__ = ("data", "grid", "time")

    _gtype: Type[GridType]

    def __init__(self, data: Optional[ArrayType], grid: GridType, time: np.datetime64):
        """Initialize Variable object with given data and grid."""
        self.data = data
        self.grid = grid
        self.time = time

        self._validate_init()

    @classmethod
    def _grid_type(cls):
        return cls._gtype

    def _increment_time(self, time: float):
        """Increase timestamp by some time.

        Arguments
        ---------
        time: float
            Given in units of seconds.
        """
        self.time = add_time(self.time, time)

    def _avg_time(self, other_time: np.datetime64) -> np.datetime64:
        if self.time == other_time:
            return self.time
        return average_npdatetime64((self.time, other_time))

    def split(
        self: VariableType, splitter: SplitVisitorBase[ArrayType]
    ) -> tuple[VariableType, ...]:
        """Split variable."""
        splitted_grid = self.grid.split(splitter)
        if self.data is None:
            return tuple(
                self.__class__(data=None, grid=g, time=self.time) for g in splitted_grid
            )
        splitted_data = splitter.split_array(self.safe_data)
        return tuple(
            self.__class__(data=d, grid=g, time=self.time)
            for d, g in zip(splitted_data, splitted_grid)
        )

    @classmethod
    def merge(
        cls: Type[VariableType],
        others: Sequence[VariableType],
        merger: MergeVisitorBase[ArrayType],
    ) -> VariableType:
        """Merge variable."""
        if all(o.data is None for o in others):
            data = None
        else:
            data = merger.merge_array([o.safe_data for o in others])
        return cls(
            data=data,
            grid=cls._grid_type().merge(tuple(o.grid for o in others), merger),
            time=others[0].time,
        )

    @abstractmethod
    def _add_data(
        self, other_data: Optional[ArrayType]
    ) -> Optional[ArrayType]:  # pragma: no cover
        """Sum data of two variables.

        Should throw TypeError or AttributeError if addition is not possible.
        """
        ...

    @property
    @abstractmethod
    def safe_data(self) -> ArrayType:  # pragma: no cover
        """Return variable data or, if it is None, a zero array of appropriate shape."""
        ...

    @abstractmethod
    def copy(self: VariableType) -> VariableType:  # pragma: no cover
        """Return a deep copy of the variable."""
        ...

    def __add__(self: VariableType, other: VariableType):
        """Add data of to variables.

        The timestamp of the sum of two variables is set to their mean.
        `None` is treated as an array of zeros of correct shape.
        """
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid != other.grid
        ):
            raise ValueError(
                "Try to add variables defined on different grids. "
                "Got {self.grid.__class__},  {other.grid.__class__}"
            )
        try:
            new_data = self._add_data(other.data)
        except (TypeError, AttributeError):
            return NotImplemented

        new_time = self._avg_time(other.time)
        return self.__class__(data=new_data, grid=self.grid, time=new_time)

    @abstractmethod
    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        """Compare two variables."""
        ...

    @abstractmethod
    def _validate_init(self):  # pragma: no cover
        """Validate after initialization."""
        ...


StateType = TypeVar("StateType", bound="StateBase")


@dataclass
class StateBase(Splitable, Generic[VariableType]):
    """Base class for all State classes.

    Combines the dynamical variables u,v, eta into one state object.
    """

    u: VariableType
    v: VariableType
    eta: VariableType
    variables: dict[str, VariableType] = field(init=False, default_factory=dict)
    diagnostic_variables: dict[str, VariableType] = field(
        init=False, default_factory=dict
    )
    _vtype: Type[VariableType] = field(init=False)

    def __init__(self, **kwargs):
        """Create State object."""
        self.variables = dict()
        self._add_to_var_dict(self.variables, **kwargs)

        self.diagnostic_variables = dict()

    @classmethod
    def _variable_type(cls):
        return cls._vtype

    def _increment_time(self, time: float):
        """Increase timestamp of all variables by some time.

        Arguments
        ---------
        time: float
            Given in units of seconds.
        """
        for _, var in self.variables.items():
            var._increment_time(time)

    def split(self: StateType, splitter: SplitVisitorBase) -> tuple[StateType, ...]:
        """Split state."""
        # tuple comprehension since dict() is mutable
        splitted: tuple[dict[str, StateType], ...] = tuple(
            dict() for _ in range(splitter.parts)
        )
        for k, v in self.variables.items():
            for i, split in enumerate(v.split(splitter)):
                splitted[i][k] = split
        return tuple(self.__class__(**s) for s in splitted)

    @classmethod
    def merge(
        cls: Type[StateType],
        others: Sequence[StateType],
        merger: MergeVisitorBase,
    ):
        """Merge variables."""
        var_type = cls._variable_type()
        merged_vars = {
            k: var_type.merge([o.variables[k] for o in others], merger)
            for k in others[0].variables.keys()
        }
        return cls(**merged_vars)

    def __add__(self, other):
        """Add all variables of two states.

        If one of the state object is missing a variable, this variable is copied
        from the other state object. This implies, that the time stamp of
        this particular variable will remain unchanged.

        Returns
        -------
        State
          Sum of two states.
        """
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented  # pragma: no cover
        try:
            sum = dict()
            for k in self.variables:
                if k in other.variables:
                    sum[k] = self.variables[k] + other.variables[k]
                else:
                    sum[k] = self.variables[k].copy()
            for k in other.variables:
                if k not in self.variables:
                    sum[k] = other.variables[k].copy()
            return self.__class__(**sum)
        except (AttributeError, TypeError):  # pragma: no cover
            return NotImplemented

    def set_diagnostic_variable(self, **kwargs):
        """Set variables for diagnostic purposes.

        Diagnostic variables are given by keyword arguments.
        Attributes are not considered by the add function.
        """
        self._add_to_var_dict(self.diagnostic_variables, **kwargs)

    def _add_to_var_dict(self, var_dict, **kwargs):
        for k, v in kwargs.items():
            if type(v) is not self._vtype:
                raise ValueError(
                    f"Keyword arguments must be of type {self._vtype}. Got {type(v)} for variable {k}"
                )
            else:
                var_dict[k] = v
                self.__setattr__(k, var_dict[k])


StateDequeType = TypeVar("StateDequeType", bound="StateDequeBase")


class StateDequeBase(Splitable, deque[StateType]):
    """Base class for State deques.

    This is a Generic class w.r.t. the type of State objects stored.
    """

    _stype: Type[StateType]

    @classmethod
    def _state_type(cls):
        """Return type of state objects."""
        return cls._stype

    def split(
        self: StateDequeType, splitter: SplitVisitorBase
    ) -> tuple[StateDequeType, ...]:
        """Split StateDeque."""
        if len(self) == 0:
            return splitter.parts * (self.__class__([], maxlen=self.maxlen),)
        splitted_states = tuple(s.split(splitter) for s in self)
        return tuple(
            self.__class__(states, maxlen=self.maxlen)
            for states in zip(*splitted_states)
        )

    @classmethod
    def merge(
        cls: Type[StateDequeType],
        others: Sequence[StateDequeType],
        merger: MergeVisitorBase,
    ) -> StateDequeType:
        """Merge StateDeques."""
        state_class = cls._state_type()
        return cls(
            (state_class.merge(states, merger) for states in zip(*others)),
            maxlen=others[0].maxlen,
        )


DomainType = TypeVar("DomainType", bound="DomainBase")


class DomainBase(Splitable, Generic[StateType, ParameterType]):
    """Base class for all Domain classes."""

    state: StateType
    history: StateDequeBase[StateType]
    parameter: ParameterType
    iteration: int
    id: int

    __slots__ = ("id", "iteration", "state", "history", "parameter")

    _stype: Type[StateType]
    _htype: Type[StateDequeBase[StateType]]
    _ptype: Type[ParameterType]

    def __init__(
        self,
        state: StateType,
        history: Optional[StateDequeBase[StateType]] = None,
        parameter: Optional[ParameterType] = None,
        id: int = 0,
        iteration: int = 0,
    ):
        """Create new Domain instance."""
        self.state = state
        if history is None:
            self.history = self._history_type()([], maxlen=3)
        else:
            self.history = history
        if parameter is None:
            self.parameter = self._parameter_type()()
        else:
            self.parameter = parameter

        self.id = id
        self.iteration = iteration

    def increment_iteration(self) -> int:
        """Return incremented iteration from domain.

        Does not modify object itself.
        """
        return self.iteration + 1

    def split(self: DomainType, splitter: SplitVisitorBase) -> tuple[DomainType, ...]:
        """Split domain."""
        splitted = (
            self.state.split(splitter),
            self.history.split(splitter),
            self.parameter.split(splitter),
        )

        out = tuple(
            self.__class__(
                state=s,
                history=h,
                parameter=p,
                id=self.id,
                iteration=self.iteration,
            )
            for s, h, p in zip(*splitted)
        )

        return out

    @classmethod
    def merge(
        cls: Type[DomainType],
        others: Sequence[DomainType],
        merger: MergeVisitorBase,
    ) -> DomainType:
        """Merge domains."""
        if len(set(o.iteration for o in others)) != 1:
            raise ValueError("Try to merge Domains that differ in iteration counter.")
        state_type = cls._state_type()
        parameter_type = cls._parameter_type()
        history_type = cls._history_type()
        return cls(
            state=state_type.merge(tuple(o.state for o in others), merger),
            history=history_type.merge(tuple(o.history for o in others), merger),
            parameter=parameter_type.merge(tuple(o.parameter for o in others), merger),
            iteration=others[0].iteration,
            id=others[0].id,
        )

    def copy(self: DomainType) -> DomainType:
        """Return deepcopy of self."""
        return deepcopy(self)

    def __eq__(self, other) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self is other:
            return True
        for a in self.__slots__:
            if not getattr(self, a) == getattr(other, a):
                return False
        return True

    @classmethod
    def _state_type(cls) -> Type[StateType]:
        """Return type of state object."""
        return cls._stype

    @classmethod
    def _history_type(cls) -> Type:
        """Return type of state object."""
        return cls._htype

    @classmethod
    def _parameter_type(cls) -> Type[ParameterType]:
        """Return type of parameter object."""
        return cls._ptype
