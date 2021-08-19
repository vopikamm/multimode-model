"""Logic related to creation of grids."""

from dataclasses import dataclass, field
from typing import Any, Dict, Union
import numpy as np
import numpy.typing as npt
from enum import Enum, unique

from .jit import _numba_2D_grid_iterator_i8


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
    """Grid information.

    A Grid object holds all information about coordinates of the grid points
    and grid spacing, i.e. size of the grid box faces. For convenience there
    are following class methods:

    Classmethods
    ------------
    cartesian: creates a Cartesian grid, potentially with unequal spacing.
    regular_lon_lat: create a regular spherical grid.

    Arguments
    ---------
    (of the __init__ method)

    x: np.ndarray
      2D np.ndarray of x-coordinates on grid
    y: np.ndarray
      2D np.ndarray of y-coordinates on grid
    mask: np.ndarray = None
      optional. Ocean mask, 1 where ocean is, 0 where land is.
      Default is a closed basin
    dim_x: int = 0  # axis of x dimension in numpy array
    dim_y: int = 1  # axis of y dimension in numpy array
    """

    x: np.ndarray  # 2D np.ndarray of x-coordinates on grid
    y: np.ndarray  # 2D np.ndarray of y-coordinates on grid
    mask: Union[
        np.ndarray, None
    ] = None  # ocean mask, 1 where ocean is, 0 where land is
    dim_x: int = 0  # x dimension in numpy array
    dim_y: int = 1  # y dimension in numpy array
    dx: np.ndarray = field(init=False)  # grid spacing in x
    dy: np.ndarray = field(init=False)  # grid spacing in y
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
    def cartesian(
        cls: Any,
        x: np.ndarray,  # longitude on grid
        y: np.ndarray,  # latitude on grid
        mask: Union[
            np.ndarray, None
        ] = None,  # ocean mask, 1 where ocean is, 0 where land is
        dim_x: int = 0,  # x dimension in numpy array
    ):
        """Generate a Cartesian grid.

        Arguments
        ---------
        x: np.ndarray
          1D Array of coordinates along x-dimension.
        y: np.ndarray
          1D Array of coordinates along y_dimension.
        mask: np.ndarray | None
          Optional ocean mask. Default is a closed domain.
        dim_x: int = 0
          Optional. Axis of the x-dimension.
        """
        assert x.ndim == y.ndim == 1
        indexing = ["ij", "xy"]
        x_2D, y_2D = np.meshgrid(x, y, indexing=indexing[dim_x])

        grid = cls(
            x=x_2D,
            y=y_2D,
            mask=mask,
            dim_x=dim_x,
            dim_y=dim_x - 1,
        )

        return grid

    @classmethod
    def regular_lat_lon(
        cls: Any,
        lon_start: float,
        lon_end: float,
        lat_start: float,
        lat_end: float,
        nx: int,
        ny: int,
        mask: Union[np.ndarray, None] = None,
        dim_x: int = 0,
        radius: float = 6_371_000.0,
    ):
        """Generate a regular spherical grid.

        Arguments
        ---------
        lon_start: float
          Smallest longitude in degrees
        lon_end: float
          larges longitude in degrees
        lat_start: float
          Smallest latitude in degrees
        lat_end: float
          larges latitude in degrees
        nx: int
          Number of grid points along x dimension.
        ny: int
          Number of grid points along y dimension.
        mask: np.ndarray | None
          Optional ocean mask. Default is a closed domain.
        dim_x: int = 0
          Optional. Axis of the x-dimension.
        radius: float = 6_371_000.0
          Radius of the sphere, defaults to Earths' radius measured in meters.
        """
        to_rad = np.pi / 180.0
        lon = np.linspace(lon_start, lon_end, nx)
        lat = np.linspace(lat_start, lat_end, ny)

        grid = cls.cartesian(
            x=lon,
            y=lat,
            mask=mask,
            dim_x=dim_x,
        )

        # reset to have a dimension of length
        grid.dx = radius * np.cos(grid.y * to_rad) * grid.dx * to_rad
        grid.dy = radius * grid.dy * to_rad

        return grid

    def __post_init__(self) -> None:
        """Set derived attributes of the grid and validate."""
        self.dx, self.dy = self._compute_grid_spacing()
        self.len_x = self.x.shape[self.dim_x]
        self.len_y = self.x.shape[self.dim_y]

        if self.mask is None:
            self.mask = self._get_default_mask(self.x.shape)

        self._validate()

    def _validate(self) -> None:
        """Validate Attributes of Grid class after init."""
        if self.mask is not None and self.mask.shape != self.x.shape:
            raise ValueError(
                f"Mask shape not matching grid shape. "
                f"Got {self.mask.shape} and {self.x.shape}."
            )

    @staticmethod
    def _get_default_mask(shape: npt._Shape):
        mask = np.ones(shape, dtype=np.int8)
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        return mask


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
    def cartesian_c_grid(
        cls: Any,
        x: np.ndarray,  # longitude on grid
        y: np.ndarray,  # latitude on grid
        mask: Union[
            np.ndarray, None
        ] = None,  # ocean mask, 1 where ocean is, 0 where land is
        dim_x: int = 0,  # x dimension in numpy array
        shift: GridShift = GridShift.LL,
    ):
        """Generate a Cartesian Arakawa C-Grid.

        Arguments
        ---------
        x: np.ndarray
          1D Array of coordinates along x-dimension.
        y: np.ndarray
          1D Array of coordinates along y_dimension.
        mask: np.ndarray | None
          Optional ocean mask. Default is a closed domain.
        dim_x: int = 0
          Optional. Axis of the x-dimension.
        shift: GridShift = GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See `GridShift` for more details.
        """
        eta_grid = Grid.cartesian(x, y, mask, dim_x)

        mask_args = (
            eta_grid.len_x,
            eta_grid.len_y,
            eta_grid.mask,
            shift.value[0],
            shift.value[1],
        )

        u_x, u_y = (eta_grid.x + shift.value[0] * eta_grid.dx / 2, eta_grid.y)
        v_x, v_y = (eta_grid.x, eta_grid.y + shift.value[1] * eta_grid.dy / 2)
        q_x, q_y = (u_x, v_y)
        u_grid = Grid(
            u_x,
            u_y,
            cls._u_mask_from_eta(*mask_args),
            dim_x=eta_grid.dim_x,
            dim_y=eta_grid.dim_y,
        )
        v_grid = Grid(
            v_x,
            v_y,
            cls._v_mask_from_eta(*mask_args),
            dim_x=eta_grid.dim_x,
            dim_y=eta_grid.dim_y,
        )
        q_grid = Grid(
            q_x,
            q_y,
            cls._q_mask_from_eta(*mask_args),
            dim_x=eta_grid.dim_x,
            dim_y=eta_grid.dim_y,
        )
        return StaggeredGrid(eta_grid, u_grid, v_grid, q_grid)

    @classmethod
    def regular_lat_lon_c_grid(
        cls,
        shift: GridShift = GridShift.LL,
        **kwargs: Dict[str, Any],
    ):
        """Generate a Arakawa C-grid for a regular longitude/latitude grid.

        Returns StaggeredGrid object with all four grids
        """
        eta_grid = Grid.regular_lat_lon(**kwargs)  # type: ignore
        dx, dy = eta_grid._compute_grid_spacing()

        u_x_start, u_x_end = (
            eta_grid.x.min() + shift.value[0] * dx.min() / 2,
            eta_grid.x.max() + shift.value[0] * dx.min() / 2,
        )
        u_kwargs = kwargs.copy()
        u_kwargs.update(dict(lon_start=u_x_start, lon_end=u_x_end))
        u_grid = Grid.regular_lat_lon(**u_kwargs)  # type: ignore
        u_grid.mask = cls._u_mask_from_eta(  # type: ignore
            u_grid.len_x,
            u_grid.len_y,
            eta_grid.mask,
            shift.value[0],
            shift.value[1],
        )

        v_y_start, v_y_end = (
            eta_grid.y.min() + shift.value[1] * dy.min() / 2,
            eta_grid.y.max() + shift.value[1] * dy.min() / 2,
        )
        v_kwargs = kwargs.copy()
        v_kwargs.update(dict(lat_start=v_y_start, lat_end=v_y_end))
        v_grid = Grid.regular_lat_lon(**v_kwargs)  # type: ignore
        v_grid.mask = cls._v_mask_from_eta(  # type: ignore
            u_grid.len_x,
            u_grid.len_y,
            eta_grid.mask,
            shift.value[0],
            shift.value[1],
        )

        q_kwargs = kwargs.copy()
        q_kwargs.update(
            dict(
                lon_start=u_x_start,
                lon_end=u_x_end,
                lat_start=v_y_start,
                lat_end=v_y_end,
            )
        )
        q_grid = Grid.regular_lat_lon(**q_kwargs)  # type: ignore
        q_grid.mask = cls._q_mask_from_eta(  # type: ignore
            u_grid.len_x,
            u_grid.len_y,
            eta_grid.mask,
            shift.value[0],
            shift.value[1],
        )

        return cls(eta_grid, u_grid, v_grid, q_grid)

    @staticmethod
    @_numba_2D_grid_iterator_i8
    def _u_mask_from_eta(
        i: int,
        j: int,
        ni: int,
        nj: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        if (eta_mask[i, j] + eta_mask[i_shift, j]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_2D_grid_iterator_i8
    def _v_mask_from_eta(
        i: int,
        j: int,
        ni: int,
        nj: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
        j_shift = (j + shift_y) % nj
        if (eta_mask[i, j] + eta_mask[i, j_shift]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_2D_grid_iterator_i8
    def _q_mask_from_eta(
        i: int,
        j: int,
        ni: int,
        nj: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        j_shift = (j + shift_y) % nj
        if (
            eta_mask[i, j]
            + eta_mask[i, j_shift]
            + eta_mask[i_shift, j]
            + eta_mask[i_shift, j_shift]
        ) == 4:
            return 1
        else:
            return 0
