"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

import numpy as np
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Union, Type, Optional, Mapping, Hashable, Any
from types import ModuleType

from .config import config
from multimodemodel.api import (
    DomainBase,
    ParameterBase,
    StateBase,
    StateDequeBase,
    VariableBase,
    Array,
    MergeVisitorBase,
    SplitVisitorBase,
)
from .jit import sum_arr
from .grid import Grid, StaggeredGrid
from .coriolis import CoriolisFunc

xarray: Union[ModuleType, Type["xr_mockup"]]

try:
    import xarray as xr

    has_xarray = True
    xarray = xr

except ModuleNotFoundError:  # pragma: no cover
    has_xarray = False

    class xr_mockup:
        """Necessary for type hinting to work."""

        class DataArray:
            """Necessary for type hinting to work."""

            ...

    xarray = xr_mockup


@dataclass(frozen=True)
class Parameter(ParameterBase):
    """Class to organise all parameters.

    The parameters may be constant in space and/or time.
    Note that the objects `compute_f` method needs to be called after creation
    to provide the coriolis parameter on all subgrids of a staggered grid instance.

    Parameters
    ----------
    g : float, default=9.81
      Gravitational acceleration m/s^2
    H : float, default=1000.0
      Depth of the fluid or thickness of the undisturbed layer in m
    rho_0 : float, default=1024.0
      Reference density of sea water in kg / m^3
    coriolis_func : CoriolisFunc, default=None
      Function used to compute the coriolis parameter on each subgrid
      of a staggered grid. It is called with the y coordinate of the respective grid.
    on_grid : StaggeredGrid, default=None
      StaggeredGrid object providing the necessary grid information if a
      parameter depends on space, such as the Coriolis parameter. Only
      required if such a parameter is part of the system to solve.

    Attributes
    ----------
    f: dict[str, numpy.ndarray]
      Mapping of the subgrid names (e.g. "u", "v", "eta") to the coriolis
      parameter on those grids
    """

    g: float = 9.81  # gravitational force m/s^2
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    _f: dict[str, Array] = field(init=False)
    _id: int = field(init=False)

    def __init__(
        self,
        g: float = 9.80665,
        H: float = 1_000.0,
        rho_0: float = 1024.0,
        coriolis_func: Optional[CoriolisFunc] = None,
        on_grid: Optional[StaggeredGrid] = None,
        f: Optional[dict[str, Array]] = None,
    ):
        """Initialize Parameter object."""
        super().__setattr__("_id", id(self))
        super().__setattr__("g", g)
        super().__setattr__("H", H)
        super().__setattr__("rho_0", rho_0)
        if f is None:
            super().__setattr__("_f", self._compute_f(coriolis_func, on_grid))
        else:
            super().__setattr__("_f", f)

    def __hash__(self):
        """Return id of instance as hash."""
        return self._id

    @property
    def f(self) -> dict[str, Array]:
        """Getter of the dictionary holding the Coriolis parameter.

        Raises
        ------
        RuntimeError
          Raised when there is no Coriolis parameter computed.
        """
        if not self._f:
            raise RuntimeError(
                "Coriolis parameter not available. "
                "Parameters object must be created with both `coriolis_func` "
                "and `on_grid` argument."
            )
        return self._f

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def split(self, splitter: SplitVisitorBase[np.ndarray]):
        """Split Parameter's spatially dependent data."""
        data = None
        try:
            data = self.f
        except RuntimeError:
            return splitter.parts * (self,)

        # Split array for each key, creating a new dictionary with the same keys
        # but holding lists of arrays
        new = {key: splitter.split_array(data[key]) for key in data}

        # Create list of dictionaries each holding just one part of splitted arrays
        out = [{key: new[key][i] for key in new} for i in range(splitter.parts)]

        return tuple(self._new_with_data(self, o) for o in out)

    @classmethod
    def _new_with_data(cls, template, data: dict[str, Array]):
        """Create instance of this class.

        Scalar parameters are copied from template.
        Dictionary of spatially varying parameters is set by data.

        Arguments
        ---------
        template: Parameter
            Some parameter object to copy scalar parameters from.
        data: dict[str, Array]
            Spatially varying parameters
        """
        # collect constant parameters
        kwargs = {
            f.name: getattr(template, f.name)
            for f in fields(template)
            if f.name not in ("_f", "_id")
        }
        kwargs["f"] = data
        return cls(**kwargs)

    @classmethod
    @lru_cache(maxsize=config.lru_cache_maxsize)
    def merge(cls, others: tuple["Parameter"], merger: MergeVisitorBase):
        """Merge Parameter's spatially varying data."""
        data = {}
        try:
            data = {
                key: merger.merge_array(tuple(o.f[key] for o in others))
                for key in others[0].f
            }
        except RuntimeError:
            pass

        return cls._new_with_data(others[0], data)

    def _compute_f(
        self, coriolis_func: Optional[CoriolisFunc], grids: Optional[StaggeredGrid]
    ) -> dict[str, Array]:
        """Compute the coriolis parameter for all subgrids.

        This method needs to be called before a rotating system
        can be set up.

        Arguments
        ---------
        grids: StaggeredGrid
          Grids on which the coriolis parameter shall be provided.

        Returns
        -------
        dict
          Mapping names of subgrids to arrays of the respective Coriolis parameter.
        """
        if coriolis_func is None or grids is None:
            return {}
        _f = {name: coriolis_func(grid.y) for name, grid in grids.items()}
        return _f

    def __eq__(self, other: Any) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, Parameter):
            return NotImplemented
        if self is other:
            return True
        return all(
            all((self._f[v] == other._f[v]).all() for v in self._f)
            if f.name == "_f"
            else getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
            if f.name != "_id"
        )


class Variable(VariableBase[np.ndarray, Grid]):
    """Variable class consisting of the data, a Grid instance and a time stamp.

    A Variable object contains the data for a single time slice of a variable as a Array,
    the grid object describing the grid arrangement and a single time stamp. The data attribute
    can take the value of :py:obj:`None` which is treated like an array of zeros when adding the
    variable to another variable.

    Variable implement summation with another Variable object, see :py:meth:`.Variable.__add__`.


    Parameters
    ----------
    data : Array, default=None
      Array containing a single time slice of a variable. If it is `None`, it will be interpreted
      as zero. To ensure a :py:class:`~numpy.ndarray` as return type, use the property :py:attr:`.safe_data`.
    grid: Grid
      Grid on which the variable is defined.
    time: np.datetime64
      Time stamp of the time slice.

    Raises
    ------
    ValueError
      Raised if `data.shape` does not match `grid.shape`.
    """

    _gtype = Grid

    @property
    def as_dataarray(self) -> xarray.DataArray:  # type: ignore
        """Return variable as :py:class:`xarray.DataArray`.

        The DataArray object contains a copy of (not a reference to) the `data` attribute of
        the variable. The horizontal coordinates are multidimensional arrays to support
        curvilinear grids and copied from the grids `x` and `y` attribute. Grid
        points for which the mask of the grid equals to 0 are converted to NaNs.

        Raises
        ------
        ModuleNotFoundError
          Raised if `xarray` is not present.
        """
        if not has_xarray:
            raise ModuleNotFoundError(  # pragma: no cover
                "Cannot convert variable to xarray.DataArray. Xarray is not available."
            )

        # copy to prevent side effects on self.data
        data = self.safe_data.copy()
        data[self.grid.mask == 0] = np.nan
        data = np.expand_dims(data, axis=0)

        coords: Mapping[Hashable, Any] = dict(
            x=(("j", "i"), self.grid.x),
            y=(("j", "i"), self.grid.y),
        )
        dims = ["j", "i"]

        if self.grid.ndim >= 3:
            coords["z"] = (("z",), self.grid.z)  # type: ignore
            dims.insert(0, "z")

        dims.insert(0, "time")
        coords["time"] = (("time",), [self.time])  # type: ignore

        return xarray.DataArray(  # type: ignore
            data=data,
            coords=coords,
            dims=dims,
        )

    @property
    def safe_data(self) -> Array:
        """Return `data` or, if it is `None`, a zero array of appropriate shape."""
        if self.data is None:
            return np.zeros(self.grid.shape)
        else:
            return self.data

    def copy(self):
        """Return a copy.

        `data` and `time` are deep copies while `grid` is a reference.
        """
        if self.data is None:
            data = None
        else:
            data = self.data.copy()
        return self.__class__(data, self.grid, self.time.copy())

    def _add_data(self, other_data: Optional[Array]) -> Optional[Array]:
        if self.data is None and other_data is None:
            new_data = None
        elif self.data is None:
            new_data = other_data.copy()  # type: ignore
        elif other_data is None:
            new_data = self.data.copy()
        else:
            new_data = sum_arr((self.data, other_data))
        return new_data

    def __eq__(self, other):
        """Return true if other is identical or the same as self."""
        if not isinstance(other, Variable):
            return NotImplemented
        if self is other:
            return True
        if (
            self.data is other.data and self.grid == other.grid
        ):  # captures both data attributes are None
            return True
        return all(
            (self.safe_data == other.safe_data).all()
            if f == "data"
            else getattr(self, f) == getattr(other, f)
            for f in self.__slots__
        )

    def _validate_init(self):
        """Validate after initialization."""
        if self.data is not None:
            if self.data.shape != self.grid.shape:
                raise ValueError(
                    f"Shape of data and grid missmatch. Got {self.data.shape} and {self.grid.shape}"
                )


class State(StateBase[Variable]):
    """State class.

    Combines the prognostic variables into a single object.

    The variables are passed as keyword arguments to :py:meth:`__init__`
    and stored in the dict :py:attr:`variables`.

    State objects can be added such that the individual variable objects are added.
    If an variable is missing in one state object, it is treated as zeros.

    For convenience, it is also possible to access the variables directly as atttributes
    of the state object.

    Parameters
    ---------
    `**kwargs` : dict
      Variables are given by keyword arguments.

    Raises
    ------
    ValueError
      Raised if a argument is not of type :py:class:`.Variable`.
    """

    _vtype = Variable


class StateDeque(StateDequeBase[State]):
    """Deque of State objects."""

    _stype = State


class Domain(DomainBase[State, Parameter]):
    """Domain classes.

    Domain objects keep references to the state of the domain, the
    history of the state (i.e. the previous evaluations of the rhs function),
    and the parameters necessary to compute the rhs function.
    """

    _stype = State
    _htype = StateDeque
    _ptype = Parameter
