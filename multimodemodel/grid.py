"""Logic related to creation of grids."""

from dataclasses import dataclass, field
from typing import Any, Tuple
import numpy as np
from enum import Enum, unique


@unique
class GridShift(Enum):
    """Direction of shift of staggered grids with respect to the eta-grid.

    E.g., `GridShift.LR` indicates that the grid points of the other grids which share
    the same index are located on the lower and/or left face of the eta Grid. The
    value of the enumerator is a tuple giving the direction of shift in
    x- and y-direction.
    """

    LR = (1, -1)
    UR = (1, 1)
    LL = (-1, -1)
    UL = (-1, 1)


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
    """Staggered Grid.

    Subgrids are available as attributes `eta`, `u`, `v` and `q`.
    """

    eta: Grid
    u: Grid
    v: Grid
    q: Grid

    @classmethod
    def regular_lat_lon_c_grid(
        cls,
        shift: GridShift = GridShift.LL,
        **kwargs: Tuple[Any],
    ):
        """Generate a Arakawa C-grid for a regular longitude/latitude grid.

        Returns StaggeredGrid object with all four grids
        """
        eta_grid = Grid.regular_lat_lon(**kwargs)
        dx, dy = eta_grid._compute_grid_spacing()

        u_x_start, u_x_end = (
            eta_grid.x.min() + shift.value[0] * dx.min() / 2,
            eta_grid.x.max() + shift.value[0] * dx.min() / 2,
        )
        u_kwargs = kwargs.copy()
        u_kwargs.update(dict(lon_start=u_x_start, lon_end=u_x_end))
        u_grid = Grid.regular_lat_lon(**u_kwargs)
        u_grid.mask = (
            (
                np.roll(eta_grid.mask, axis=eta_grid.dim_x, shift=-1 * shift.value[0])
                == 1
            )
            & (eta_grid.mask == 1)
        ).astype(int)

        v_y_start, v_y_end = (
            eta_grid.y.min() + shift.value[1] * dy.min() / 2,
            eta_grid.y.max() + shift.value[1] * dy.min() / 2,
        )
        v_kwargs = kwargs.copy()
        v_kwargs.update(dict(lat_start=v_y_start, lat_end=v_y_end))
        v_grid = Grid.regular_lat_lon(**v_kwargs)
        v_grid.mask = (
            (
                np.roll(eta_grid.mask, axis=eta_grid.dim_y, shift=-1 * shift.value[1])
                == 1
            )
            & (eta_grid.mask == 1)
        ).astype(int)

        q_kwargs = v_kwargs.copy()
        q_kwargs.update(dict(lon_start=u_x_start, lon_end=u_x_end))
        q_grid = Grid.regular_lat_lon(**q_kwargs)
        q_grid.mask = (
            (eta_grid.mask == 1)
            | (
                np.roll(eta_grid.mask, axis=eta_grid.dim_x, shift=-1 * shift.value[0])
                == 1
            )
            | (
                np.roll(eta_grid.mask, axis=eta_grid.dim_y, shift=-1 * shift.value[1])
                == 1
            )
            | (
                np.roll(
                    np.roll(
                        eta_grid.mask, axis=eta_grid.dim_x, shift=-1 * shift.value[0]
                    ),
                    axis=eta_grid.dim_y,
                    shift=-1 * shift.value[1],
                )
                == 1
            )
        ).astype(int)

        return cls(eta_grid, u_grid, v_grid, q_grid)
