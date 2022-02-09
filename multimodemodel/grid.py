"""Logic related to creation of grids."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import numpy as np
import numpy.typing as npt
from enum import Enum, unique

from .jit import _numba_3D_grid_iterator


@unique
class GridShift(Enum):
    """Direction of shift of staggered grids with respect to the eta-grid.

    E.g., `GridShift.LR` indicates that the grid points of the other grids which share
    the same index are located on the lower and/or left face of the q Grid. The
    value of the enumerator is a tuple giving the direction of shift in
    y- and x-direction.
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
    z: np.ndarray = np.array([0])
      1D np.ndarray of vertical normal modes.
    mask: Optional[np.ndarray] = None
      Ocean mask, 1 where ocean is, 0 where land is.
      Default is a closed basin

    Attributes
    ----------
    dx: np.ndarray
      Grid spacing in x.
    dy: np.ndarray
      Grid spacing in y.

    Properties
    ----------
    shape: npt._Shape
      Tuple of int defining the shape of the grid.
    ndim: int
      Number of dimensions on which the grid is defined
    dim_x: int =-1
      Axis of x-dimension
    dim_y: int =-2
      Axis of y-dimension
    dim_z: int = -3
      Axis of z-dimension
    """

    x: np.ndarray  # 2D np.ndarray of x-coordinates on grid
    y: np.ndarray  # 2D np.ndarray of y-coordinates on grid
    z: np.ndarray = np.array([0])  # 1D np.ndarray of z coordinates.
    mask: Optional[np.ndarray] = None  # ocean mask, 1 where ocean is, 0 where land is
    dx: np.ndarray = field(init=False)  # grid spacing in x
    dy: np.ndarray = field(init=False)  # grid spacing in y

    def __post_init__(self) -> None:
        """Set derived attributes of the grid and validate."""
        self.dx, self.dy = self._compute_grid_spacing()

        if self.mask is None:
            self.mask = self._get_default_mask(self.shape)

        self._validate()

    @property
    def shape(self) -> npt._Shape:
        """Return shape tuple of grid."""
        return self.z.shape + self.x.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.shape)

    @property
    def dim_x(self) -> int:
        """Return axis of x dimension."""
        return -1

    @property
    def dim_y(self) -> int:
        """Return axis of x dimension."""
        return -2

    @property
    def dim_z(self) -> int:
        """Return axis of x dimension."""
        return -3

    def _compute_grid_spacing(self):
        dx, dy = self._compute_horizontal_grid_spacing()
        return dx, dy

    def _compute_horizontal_grid_spacing(self):
        """Compute the spatial differences along x and y."""
        dx = np.diff(self.x, axis=self.dim_x)
        dy = np.diff(self.y, axis=self.dim_y)
        dx_0 = dx[:, 0]
        dy_0 = dy[0, :]
        dx = np.append(dx, np.expand_dims(dx_0, axis=self.dim_x), axis=self.dim_x)
        dy = np.append(dy, np.expand_dims(dy_0, axis=self.dim_y), axis=self.dim_y)
        return dx, dy

    @classmethod
    def cartesian(
        cls: Any,
        x: np.ndarray,  # x coordinate
        y: np.ndarray,  # y coordinate
        z: np.ndarray = np.array([0]),  # z coordinate
        mask: Optional[
            np.ndarray
        ] = None,  # ocean mask, 1 where ocean is, 0 where land is
    ):
        """Generate a Cartesian grid.

        Arguments
        ---------
        x: np.ndarray
          1D Array of coordinates along x dimension.
        y: np.ndarray
          1D Array of coordinates along y dimension.
        z: np.ndarray = np.array([0]),
          1D Array of vertical normal modes.
        mask: Optional[np.ndarray] = None
          Optional ocean mask. Default is a closed domain.
        """
        assert x.ndim == y.ndim == 1
        x_2D, y_2D = np.meshgrid(x, y, indexing="xy")

        grid = cls(
            x=x_2D,
            y=y_2D,
            z=z,
            mask=mask,
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
        z: np.ndarray = np.array([0]),
        mask: Optional[np.ndarray] = None,
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
        z: np.ndarray = np.array([0])
          Optional 1D coordinate array along vertical dimension.
        mask: np.ndarray | None
          Optional ocean mask. Default is a closed domain.
        radius: float = 6_371_000.0
          Radius of the sphere, defaults to Earths' radius measured in meters.
        """
        to_rad = np.pi / 180.0
        lon = np.linspace(lon_start, lon_end, nx)
        lat = np.linspace(lat_start, lat_end, ny)

        grid = cls.cartesian(
            x=lon,
            y=lat,
            z=z,
            mask=mask,
        )

        # reset to have a dimension of length
        grid.dx = radius * np.cos(grid.y * to_rad) * grid.dx * to_rad
        grid.dy = radius * grid.dy * to_rad

        return grid

    def _validate(self) -> None:
        """Validate Attributes of Grid class after init."""
        if self.mask is not None and self.mask.shape != self.shape:
            raise ValueError(
                f"Mask shape not matching grid shape. "
                f"Got {self.mask.shape} and {self.shape}."
            )
        assert self.x.ndim == 2
        assert self.y.ndim == 2
        assert self.z.ndim == 1

    @staticmethod
    def _get_default_mask(shape: npt._Shape):
        mask = np.ones(shape, dtype=np.float64)
        mask[..., 0, :] = 0.0
        mask[..., -1, :] = 0.0
        mask[..., :, 0] = 0.0
        mask[..., :, -1] = 0.0
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
        shift: GridShift = GridShift.UR,
        **grid_kwargs: Dict[str, Any],
    ):
        """Generate a Cartesian Arakawa C-Grid.

        Arguments
        ---------
        shift: GridShift = GridShift.RR
          Direction of shift of staggered grids with respect to the q-grid.
          See `GridShift` for more details.
        **grid_kwargs: Dict[str, Any]:
          keyword arguments are passed to Grid.cartesian().
        """
        q_grid = Grid.cartesian(**grid_kwargs)  # type: ignore

        u_x, u_y = (q_grid.x, q_grid.y + shift.value[1] * q_grid.dy / 2)
        v_x, v_y = (q_grid.x + shift.value[0] * q_grid.dx / 2, q_grid.y)
        eta_x, eta_y = (v_x, u_y)
        z = q_grid.z
        u_grid = Grid(
            x=u_x,
            y=u_y,
            z=z,
            mask=cls._compute_mask(cls._u_mask_from_q, q_grid, shift),
        )
        v_grid = Grid(
            x=v_x,
            y=v_y,
            z=z,
            mask=cls._compute_mask(cls._v_mask_from_q, q_grid, shift),
        )
        eta_grid = Grid(
            x=eta_x,
            y=eta_y,
            z=z,
            mask=cls._compute_mask(cls._eta_mask_from_q, q_grid, shift),
        )
        return StaggeredGrid(eta_grid, u_grid, v_grid, q_grid)

    @classmethod
    def regular_lat_lon_c_grid(
        cls,
        shift: GridShift = GridShift.UR,
        **kwargs: Dict[str, Any],
    ):
        """Generate a Arakawa C-grid for a regular longitude/latitude grid.

        Returns StaggeredGrid object with all four grids
        """
        q_grid = Grid.regular_lat_lon(**kwargs)  # type: ignore
        dx, dy = q_grid._compute_horizontal_grid_spacing()

        u_y_start, u_y_end = (
            q_grid.y.min() + shift.value[1] * dy.min() / 2,
            q_grid.y.max() + shift.value[1] * dy.min() / 2,
        )
        u_kwargs = kwargs.copy()
        u_kwargs.update(dict(lat_start=u_y_start, lat_end=u_y_end))
        u_grid = Grid.regular_lat_lon(**u_kwargs)  # type: ignore
        u_grid.mask = cls._compute_mask(
            cls._u_mask_from_q,  # type: ignore
            q_grid,
            shift,
        )

        v_x_start, v_x_end = (
            q_grid.x.min() + shift.value[0] * dx.min() / 2,
            q_grid.x.max() + shift.value[0] * dx.min() / 2,
        )
        v_kwargs = kwargs.copy()
        v_kwargs.update(dict(lon_start=v_x_start, lon_end=v_x_end))
        v_grid = Grid.regular_lat_lon(**v_kwargs)  # type: ignore
        v_grid.mask = cls._compute_mask(
            cls._v_mask_from_q,  # type: ignore
            q_grid,
            shift,
        )

        eta_kwargs = kwargs.copy()
        eta_kwargs.update(
            dict(
                lon_start=v_x_start,
                lon_end=v_x_end,
                lat_start=u_y_start,
                lat_end=u_y_end,
            )
        )
        eta_grid = Grid.regular_lat_lon(**eta_kwargs)  # type: ignore
        eta_grid.mask = cls._compute_mask(
            cls._eta_mask_from_q,  # type: ignore
            q_grid,
            shift,
        )

        return cls(eta_grid, u_grid, v_grid, q_grid)

    @staticmethod
    def _compute_mask(
        func: Callable[..., np.ndarray], from_grid: Grid, shift: GridShift
    ):
        return func(
            from_grid.shape[from_grid.dim_x],
            from_grid.shape[from_grid.dim_y],
            from_grid.shape[from_grid.dim_z],
            from_grid.mask,
            shift.value[0],
            shift.value[1],
        )

    @staticmethod
    @_numba_3D_grid_iterator
    def _u_mask_from_q(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        q_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        j_shift = (j + shift_y) % nj
        if (q_mask[k, j, i] + q_mask[k, j_shift, i]) == 0:
            return 0.0
        else:
            return 1.0

    @staticmethod
    @_numba_3D_grid_iterator
    def _v_mask_from_q(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        q_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        if (q_mask[k, j, i] + q_mask[k, j, i_shift]) == 0:
            return 0.0
        else:
            return 1.0

    @staticmethod
    @_numba_3D_grid_iterator
    def _eta_mask_from_q(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        q_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        j_shift = (j + shift_y) % nj
        if (
            q_mask[k, j, i]
            + q_mask[k, j_shift, i]
            + q_mask[k, j, i_shift]
            + q_mask[k, j_shift, i_shift]
        ) == 0:
            return 0.0
        else:
            return 1.0
