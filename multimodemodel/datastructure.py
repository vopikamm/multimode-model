"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Parameters:
    """Class to organise all constant parameters."""

    f: float = 0.0
    g: float = 9.81  # gravitational force m/s^2
    beta: float = 2.0 / (24 * 3600)  # beta parameter 1/ms with f=f_0+beta *y
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    dt: float = 1.0  # time stepping in s
    t_0: float = 0.0  # starting time
    t_end: float = 3600.0  # end time
    write: int = 20  # how many states should be output
    r: float = 6371000.0  # radius of the earth


@dataclass
class Grid:
    """Grid informtation."""

    x: np.array  # longitude on grid
    y: np.array  # latitude on grid
    mask: np.array  # ocean mask, 1 where ocean is, 0 where land is
    dim_x: int = 0  # x dimension in numpy array
    dim_y: int = 1  # y dimension in numpy array
    dx: int = field(init=False)  # grid spacing in x
    dy: int = field(init=False)  # grid spacing in y
    len_x: int = field(init=False)  # length of array in x dimension
    len_y: int = field(init=False)  # length of array in y dimension

    def _compute_grid_spacing(self):
        """Compute the spatial differences along x and y."""
        dx = np.diff(self.x, axis=self.dim_x)
        dy = np.diff(self.y, axis=self.dim_y)
        if self.dim_x == 0:
            dx_0 = dx[0, :]
            dy_0 = dy[:, 0]
        else:
            dx_0 = dx[:, 0]
            dy_0 = dy[0, :]
        dx = np.append(dx, np.expand_dims(dx_0, axis=self.dim_x), axis=self.dim_x)
        dy = np.append(dy, np.expand_dims(dy_0, axis=self.dim_y), axis=self.dim_y)
        return dx, dy

    @classmethod
    def regular_lat_lon(
        cls: Any,
        lon_start: float,
        lon_end: float,
        lat_start: float,
        lat_end: float,
        nx: int,
        ny: int,
        dim_x: int = 0,
        r_earth: float = 6_371_000.0,
    ):
        """Generate a regular lat/lon grid."""
        indexing = ["ij", "xy"]
        to_rad = np.pi / 180.0
        lon = np.linspace(lon_start, lon_end, nx)
        lat = np.linspace(lat_start, lat_end, ny)
        longitude, latitude = np.meshgrid(lon, lat, indexing=indexing[dim_x])
        mask = np.ones(longitude.shape)
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0

        grid = cls(
            x=longitude,
            y=latitude,
            mask=mask,
            dim_x=dim_x,
            dim_y=1 - dim_x,
        )

        grid.dx = r_earth * np.cos(grid.y * to_rad) * grid.dx * to_rad
        grid.dy = r_earth * grid.dy * to_rad

        return grid

    def __post_init__(self) -> None:
        """Set derived attributes of the grid."""
        self.dx, self.dy = self._compute_grid_spacing()
        self.len_x = self.x.shape[self.dim_x]
        self.len_y = self.x.shape[self.dim_y]


@dataclass
class StaggeredGrid:
    """The u, v, and eta grids are evaluated on staggered gridpoints."""

    q_grid: Grid
    u_grid: Grid
    v_grid: Grid
    eta_grid: Grid

    @classmethod
    def regular_lat_lon_c_grid(
        cls: Any,
        **kwargs_to_callable: Tuple[Any],
    ):
        """Generate an Arakawa C-grid based on a given Grid() classmethod."""
        q_grid = Grid.regular_lat_lon(**kwargs_to_callable)
        dx, dy = q_grid._compute_grid_spacing()

        u_lat_start, u_lat_end = (
            q_grid.y.min() + dy.min() / 2,
            q_grid.y.max() + dy.min() / 2,
        )
        u_kwargs = kwargs_to_callable.copy()
        u_kwargs.update(dict(lat_start=u_lat_start, lat_end=u_lat_end))
        u_grid = Grid.regular_lat_lon(**u_kwargs)
        u_grid.mask = (
            (np.roll(q_grid.mask, axis=q_grid.dim_y, shift=-1) == 1)
            & (q_grid.mask == 1)
        ).astype(int)

        v_lon_start, v_lon_end = (
            q_grid.x.min() + dx.min() / 2,
            q_grid.x.max() + dx.min() / 2,
        )
        v_kwargs = kwargs_to_callable.copy()
        v_kwargs.update(dict(lon_start=v_lon_start, lon_end=v_lon_end))
        v_grid = Grid.regular_lat_lon(**v_kwargs)
        v_grid.mask = (
            (np.roll(q_grid.mask, axis=q_grid.dim_x, shift=-1) == 1)
            & (q_grid.mask == 1)
        ).astype(int)

        eta_kwargs = v_kwargs.copy()
        eta_kwargs.update(dict(lat_start=u_lat_start, lat_end=u_lat_end))
        eta_grid = Grid.regular_lat_lon(**eta_kwargs)
        eta_grid.mask = (
            (q_grid.mask == 1)
            & (np.roll(q_grid.mask, axis=q_grid.dim_x, shift=-1) == 1)
            & (np.roll(q_grid.mask, axis=q_grid.dim_y, shift=-1) == 1)
            & (
                np.roll(
                    np.roll(q_grid.mask, axis=q_grid.dim_x, shift=-1),
                    axis=q_grid.dim_y,
                    shift=-1,
                )
                == 1
            )
        ).astype(int)

        return cls(q_grid, u_grid, v_grid, eta_grid)


@dataclass
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: np.array
    grid: Grid

    def __add__(self, other):
        """Add data of to variables."""
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")
        try:
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
            return NotImplemented
        try:
            u_new = self.u + other.u
            v_new = self.v + other.v
            eta_new = self.eta + other.eta
        except (AttributeError, TypeError):
            return NotImplemented
        return self.__class__(u=u_new, v=v_new, eta=eta_new)
