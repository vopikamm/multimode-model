"""Logic related to creation of grids."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import numpy as np
import numpy.typing as npt
from enum import Enum, unique

from .jit import _numba_3D_grid_iterator_i8


@unique
class GridShift(Enum):
    """Direction of shift of staggered grids with respect to the eta-grid.

    E.g., `GridShift.LR` indicates that the grid points of the other grids which share
    the same index are located on the lower and/or left face of the eta Grid. The
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
        mask = np.ones(shape, dtype=np.int8)
        mask[..., 0, :] = 0
        mask[..., -1, :] = 0
        mask[..., :, 0] = 0
        mask[..., :, -1] = 0
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
        shift: GridShift = GridShift.LL,
        **grid_kwargs: Dict[str, Any],
    ):
        """Generate a Cartesian Arakawa C-Grid.

        Arguments
        ---------
        shift: GridShift = GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See `GridShift` for more details.
        **grid_kwargs: Dict[str, Any]:
          keyword arguments are passed to Grid.cartesian().
        """
        eta_grid = Grid.cartesian(**grid_kwargs)  # type: ignore

        u_x, u_y = (eta_grid.x + shift.value[0] * eta_grid.dx / 2, eta_grid.y)
        v_x, v_y = (eta_grid.x, eta_grid.y + shift.value[1] * eta_grid.dy / 2)
        q_x, q_y = (u_x, v_y)
        z = eta_grid.z
        u_grid = Grid(
            x=u_x,
            y=u_y,
            z=z,
            mask=cls._compute_mask(cls._u_mask_from_eta, eta_grid, shift),
        )
        v_grid = Grid(
            x=v_x,
            y=v_y,
            z=z,
            mask=cls._compute_mask(cls._v_mask_from_eta, eta_grid, shift),
        )
        q_grid = Grid(
            x=q_x,
            y=q_y,
            z=z,
            mask=cls._compute_mask(cls._q_mask_from_eta, eta_grid, shift),
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
        dx, dy = eta_grid._compute_horizontal_grid_spacing()

        u_x_start, u_x_end = (
            eta_grid.x.min() + shift.value[0] * dx.min() / 2,
            eta_grid.x.max() + shift.value[0] * dx.min() / 2,
        )
        u_kwargs = kwargs.copy()
        u_kwargs.update(dict(lon_start=u_x_start, lon_end=u_x_end))
        u_grid = Grid.regular_lat_lon(**u_kwargs)  # type: ignore
        u_grid.mask = cls._compute_mask(
            cls._u_mask_from_eta,  # type: ignore
            eta_grid,
            shift,
        )

        v_y_start, v_y_end = (
            eta_grid.y.min() + shift.value[1] * dy.min() / 2,
            eta_grid.y.max() + shift.value[1] * dy.min() / 2,
        )
        v_kwargs = kwargs.copy()
        v_kwargs.update(dict(lat_start=v_y_start, lat_end=v_y_end))
        v_grid = Grid.regular_lat_lon(**v_kwargs)  # type: ignore
        v_grid.mask = cls._compute_mask(
            cls._v_mask_from_eta,  # type: ignore
            eta_grid,
            shift,
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
        q_grid.mask = cls._compute_mask(
            cls._q_mask_from_eta,  # type: ignore
            eta_grid,
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
    @_numba_3D_grid_iterator_i8
    def _u_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        if (eta_mask[k, j, i] + eta_mask[k, j, i_shift]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_3D_grid_iterator_i8
    def _v_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        j_shift = (j + shift_y) % nj
        if (eta_mask[k, j, i] + eta_mask[k, j_shift, i]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_3D_grid_iterator_i8
    def _q_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> float:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        j_shift = (j + shift_y) % nj
        if (
            eta_mask[k, j, i]
            + eta_mask[k, j_shift, i]
            + eta_mask[k, j, i_shift]
            + eta_mask[k, j_shift, i_shift]
        ) == 4:
            return 1
        else:
            return 0
