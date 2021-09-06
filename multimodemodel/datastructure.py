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

    g: float = 9.81  # gravitational force m/s^2
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
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
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: Optional[np.ndarray]
    grid: Grid

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

        return xarray.DataArray(  # type: ignore
            data=data,
            coords={
                "x": (("i", "j"), self.grid.x),
                "y": (("i", "j"), self.grid.y),
            },
            dims=("i", "j"),
        )

    @property
    def safe_data(self) -> np.ndarray:
        """Return self.data or, if it is None, a zero array of appropriate shape."""
        if self.data is None:
            return np.zeros(self.grid.x.shape)
        else:
            return self.data

    def __add__(self, other):
        """Add data of to variables."""
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")
        try:
            if self.data is None:
                new_data = other.data
            elif other.data is None:
                new_data = self.data
            else:
                new_data = self.data + other.data
        except (TypeError, AttributeError):
            return NotImplemented
        return self.__class__(data=new_data, grid=self.grid)


@dataclass
class State:
    """State class.

    Combines the dynamical variables u,v, eta into one state object.
    """

    u: Variable
    v: Variable
    eta: Variable

    def __add__(self, other):
        """Add all variables of two states."""
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented  # pragma: no cover
        try:
            u_new = self.u + other.u
            v_new = self.v + other.v
            eta_new = self.eta + other.eta
        except (AttributeError, TypeError):  # pragma: no cover
            return NotImplemented
        return self.__class__(u=u_new, v=v_new, eta=eta_new)
