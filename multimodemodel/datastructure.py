"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

import numpy as np
from dataclasses import dataclass, field, asdict, InitVar
from .grid import Grid, StaggeredGrid
from .coriolis import CoriolisFunc
from typing import Dict, Optional

try:
    import xarray

    has_xarray = True
except ModuleNotFoundError:  # pragma: no cover
    has_xarray = False

    # for type hinting
    class xarray:
        """Necessary for type hinting to work."""

        DataArray = None


@dataclass
class Parameters:
    """Class to organise all parameters.

    The parameters may be constant in space and/or time.
    Note that the objects `compute_f` method needs to be called after creation
    to provide the coriolis parameter on all subgrids of a staggered grid instance.

    Arguments
    ---------
    (of the __init__ method)

    g: float = 9.81
      Gravitational acceleration m/s^2
    H: float = 1000.0
      Depth of the fluid or thickness of the undisturbed layer in m
    rho_0: float = 1024.0
      Reference density of sea water in kg / m^3
    coriolis_func: Optional[CoriolisFunc] = None
      Function used to compute the coriolis parameter on each subgrid
      of a staggered grid. The signature of this function must match
      `coriolis_func(y: numpy.ndarray) -> numpy.ndarray`
      and they are called with the y coordinate of the respective grid.
    on_grid: Optional[StaggeredGrid] = None
      StaggeredGrid object providing the necessary grid information if a
      parameter depends on space, such as the Coriolis parameter. Only
      required if such a parameter is part of the system to solve.


    Attributes
    ----------
    f: Dict[str, numpy.ndarray]
      Mapping of the subgrid names ('u', 'v', 'eta') to the coriolis
      parameter on those grids
    """

    g: float = 9.81  # gravitational force m / s^2
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    nu: float = 2000.0  # horizontal mixing coefficient in m^2 / s
    coriolis_func: InitVar[Optional[CoriolisFunc]] = None
    on_grid: InitVar[Optional[StaggeredGrid]] = None
    _f: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(
        self,
        coriolis_func: Optional[CoriolisFunc],
        on_grid: Optional[StaggeredGrid],
    ):
        """Initialize derived fields."""
        self._f = self._compute_f(coriolis_func, on_grid)

    @property
    def f(self) -> Dict[str, np.ndarray]:
        """Getter of the dictionary holding the Coriolis parameter."""
        if not self._f:
            raise RuntimeError(
                "Coriolis parameter not available. "
                "Parameters object must be created with both `coriolis_func` "
                "and `on_grid` argument."
            )
        return self._f

    def _compute_f(
        self, coriolis_func: Optional[CoriolisFunc], grids: Optional[StaggeredGrid]
    ) -> Dict[str, np.ndarray]:
        """Compute the coriolis parameter for all subgrids.

        This method needs to be called before a rotating system
        can be set up. Returns None but set the attribute `f` of the object.

        Arguments
        ---------
        grids: StaggeredGrid
          Grids on which the coriolis parameter shall be provided.
        """
        if coriolis_func is None or grids is None:
            return {}
        _f = {name: coriolis_func(grid["y"]) for name, grid in asdict(grids).items()}
        return _f


@dataclass
class MultimodeParameters(Parameters):
    """Subclass to Parameters.

    Organises all parameters depending on the mode number.
    """

    H: np.ndarray = np.array([Parameters.H])

    def parameters_by_modenumber(self, k: int = 0) -> Parameters:
        """Return the parameters for one mode number as Parameters object.

        Arguments
        ---------
        k: int
          Respective mode number to be returned.
        """
        if k >= len(self.H):
            raise ValueError(
                "The requested mode number is higher then the available modes."
            )
        self.H = self.H[k]
        return self


@dataclass
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: Optional[np.ndarray]
    grid: Grid
    time: np.datetime64

    def __post_init__(self):
        """Validate."""
        if self.data is not None:
            if self.data.shape != self.grid.shape:
                raise ValueError(
                    "Shape of data and grid missmatch. "
                    f"Got {self.data.shape} and {self.grid.shape}"
                )

    @property
    def as_dataarray(self) -> xarray.DataArray:  # type: ignore
        """Return variable as xarray.DataArray.

        The DataArray object contains a copy of (not a reference to) the data of
        the variable. The coordinates are multidimensional arrays to support
        curvilinear grids and copied from the grids `x` and `y` attribute. Grid
        points for which the mask of the grid equals to 0 are converted to NaNs.
        """
        if not has_xarray:
            raise ModuleNotFoundError(  # pragma: no cover
                "Cannot convert variable to xarray.DataArray. Xarray is not available."
            )

        # copy to prevent side effects on self.data
        data = self.safe_data.copy()
        data[self.grid.mask == 0] = np.nan
        data = np.expand_dims(data, axis=0)

        coords = dict(
            x=(("j", "i"), self.grid.x),
            y=(("j", "i"), self.grid.y),
        )
        dims = ("j", "i")

        if self.grid.ndim >= 3:
            coords["z"] = (("z",), self.grid.z)  # type: ignore
            dims = ("z",) + dims

        dims = ("time",) + dims
        coords["time"] = (("time",), [self.time])  # type: ignore

        return xarray.DataArray(  # type: ignore
            data=data,
            coords=coords,
            dims=dims,
        )

    @property
    def safe_data(self) -> np.ndarray:
        """Return self.data or, if it is None, a zero array of appropriate shape."""
        if self.data is None:
            return np.zeros(self.grid.shape)
        else:
            return self.data

    def copy(self):
        """Return a copy.

        The data attribute is a deep copy while the grid is a reference.
        """
        if self.data is None:
            data = None
        else:
            data = self.data.copy()
        return self.__class__(data, self.grid, self.time.copy())

    def __add__(self, other):
        """Add data of two variables.

        The timestamp of the sum of two variables is set to their mean.
        """
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")

        try:
            if self.data is None and other.data is None:
                new_data = None
            elif self.data is None:
                new_data = other.data.copy()  # type: ignore
            elif other.data is None:
                new_data = self.data.copy()
            else:
                new_data = self.data + other.data
        except (AttributeError, TypeError):
            return NotImplemented

        new_time = self.time + (other.time - self.time) / 2

        return self.__class__(data=new_data, grid=self.grid, time=new_time)


class State:
    """State class.

    Combines the prognostic variables into one state object.

    The variables are passed as keyword arguments to the __init__ method
    of the state object and stored in the attribute `variables` which is a
    dict.

    State objects can be added such that the individual variable objects are added.
    If an variable is missing in one state object, it is treated as zeros.

    For convenience, it is also possible to access the variables directly as atttributes
    of the state object.
    """

    def __init__(self, **kwargs):
        """Create State object.

        Variables are given by keyword arguments.
        """
        self.variables = dict()

        for k, v in kwargs.items():
            if type(v) is not Variable:
                raise ValueError("Keyword arguments must be of type Variable.")
            else:
                self.variables[k] = v
                self.__setattr__(k, self.variables[k])

    def __add__(self, other):
        """Add all variables of two states.

        If one of the state object is missing a variable, this variable is copied
        from the other state object. Note that this means, that the time stamp of
        this particular variable will remain unchanged.
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
