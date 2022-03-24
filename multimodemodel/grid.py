"""Logic related to creation of grids."""

from dataclasses import dataclass, field, fields

from typing import Any, Type, Optional, Callable
import numpy as np
from functools import lru_cache

from .api import (
    GridShift,
    GridBase,
    StaggeredGridBase,
    Array,
    Shape,
    MergeVisitorBase,
    SplitVisitorBase,
)
from .config import config
from .jit import _numba_3D_grid_iterator_i8_parallel_over_k


def _check_shape(arr1, expected, msg=""):
    if arr1.shape != expected.shape:
        raise ValueError(f"{msg}. Got {arr1.shape}, but expected {expected.shape}.")


@dataclass(frozen=True)
class Grid(GridBase[np.ndarray]):
    """Grid information.

    A Grid object holds all information about coordinates of the grid points
    and grid spacing, i.e. size of the grid box faces. For convenience there
    are following class methods:

    Arguments
    ---------
    (of the __init__ method)

    x: np.ndarray
      2D np.ndarray of x-coordinates on grid
    y: np.ndarray
      2D np.ndarray of y-coordinates on grid
    z : Array, default=None
      1D Array of z coordinates.
    mask: Array = None
      optional. Ocean mask, 1 where ocean is, 0 where land is. Default is a
      rectangular domain with closed boundaries.
    dx: Optional[np.ndarray] = None
      Initialization of dx
    dy: Optional[np.ndarray] = None
      Initialization of dy
    dz: Optional[np.ndarray] = None
      Initialization of dz

    Attributes
    ----------
    dx : Array
      Grid spacing in x.
    dy : Array
      Grid spacing in y.
    dz : Array
      Grid spacing in z.
    mask: Array
      optional. Ocean mask, 1 where ocean is, 0 where land is. Default is a
      rectangular domain with closed boundaries.
    shape: tuple[int]
      Shape of the grid data.
    ndim: int:
      Number of dimensions
    dim_x: int
      Axis index of x dimension.
    dim_y: int
      Axis index of y dimension.
    dim_z: int
      Axis index of z dimension.

    Raises
    ------
    ValueError
      Raised if the shape of `mask` does not fit the shape of the grid.
    """

    x: Array
    y: Array
    z: Array
    mask: Array
    dx: Array
    dy: Array
    dz: Array
    _id: int = field(init=False)

    def __init__(
        self,
        x: Array,
        y: Array,
        z: Optional[Array] = None,
        mask: Optional[Array] = None,
        dx: Optional[Array] = None,
        dy: Optional[Array] = None,
        dz: Optional[Array] = None,
    ):
        """Initialize self."""
        super().__setattr__("_id", id(self))
        super().__setattr__("x", x)
        super().__setattr__("y", y)
        if z is None:
            _z = np.array([], dtype=self.x.dtype)
        else:
            _z = z
        super().__setattr__("z", _z)
        if mask is None:
            mask = self._get_default_mask(self.shape)
        super().__setattr__("mask", mask)

        if dx is None:
            dx = self._compute_grid_spacing(coord=self.x, axis=self.dim_x)
        super().__setattr__("dx", dx)
        if dy is None:
            dy = self._compute_grid_spacing(coord=self.y, axis=self.dim_y)
        super().__setattr__("dy", dy)
        if dz is None:
            dz = self._compute_grid_spacing(coord=self.z, axis=0)
        super().__setattr__("dz", dz)

        # validate
        _check_shape(self.mask, self, "Mask shape not matching grid shape")
        _check_shape(self.dx, self.x, "dx shape not matching shape of x")
        _check_shape(self.dy, self.y, "dy shape not matching shape of y")
        assert self.x.ndim == 2
        assert self.y.ndim == 2
        if self.z is not None:
            assert self.z.ndim == 1

    @property
    def shape(self) -> Shape:
        """Return shape tuple of grid."""
        if len(self.z) == 0:
            return self.x.shape
        else:
            return self.z.shape + self.x.shape

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

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def split(self, splitter: SplitVisitorBase[np.ndarray]):
        """Split grid."""
        x, y, mask, dx, dy = (
            splitter.split_array(arr)
            for arr in (self.x, self.y, self.mask, self.dx, self.dy)
        )
        return tuple(
            self.__class__(
                **dict(x=x0, y=y0, z=self.z, mask=mask0, dx=dx0, dy=dy0, dz=self.dz)
            )
            for x0, y0, mask0, dx0, dy0 in zip(x, y, mask, dx, dy)
        )

    @classmethod
    @lru_cache(maxsize=config.lru_cache_maxsize)
    def merge(cls, others: tuple["Grid"], merger: MergeVisitorBase):
        """Merge grids."""
        x = merger.merge_array(tuple(o.x for o in others))
        y = merger.merge_array(tuple(o.y for o in others))
        mask = merger.merge_array(tuple(o.mask for o in others))
        dx = merger.merge_array(tuple(o.dx for o in others))
        dy = merger.merge_array(tuple(o.dy for o in others))
        return cls(x=x, y=y, z=others[0].z, mask=mask, dx=dx, dy=dy, dz=others[0].dz)

    @staticmethod
    def _compute_grid_spacing(coord: Array, axis: int) -> Array:
        """Compute the spatial differences of a coordinate along a given axis."""
        if coord is None or len(coord) <= 1:
            return np.array([], dtype=coord.dtype)
        dx = np.diff(coord, axis=axis)
        dx_0 = dx.take(indices=0, axis=axis)
        dx = np.append(dx, np.expand_dims(dx_0, axis=axis), axis=axis)
        return dx

    @classmethod
    def cartesian(
        cls: Any,
        x: Array,
        y: Array,
        z: Optional[Array] = None,
        mask: Optional[Array] = None,
        **kwargs,
    ) -> "Grid":
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
        assert x.ndim == y.ndim == 1
        x_2D, y_2D = np.meshgrid(x, y, indexing="xy")

        grid = cls(
            x=x_2D,
            y=y_2D,
            z=z,
            mask=mask,
            **kwargs,
        )

        return grid

    @classmethod
    def regular_lat_lon(
        cls: Type["Grid"],
        lon_start: float,
        lon_end: float,
        lat_start: float,
        lat_end: float,
        nx: int,
        ny: int,
        z: Optional[Array] = None,
        mask: Optional[Array] = None,
        radius: float = 6_371_000.0,
    ) -> "Grid":
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
        to_rad = np.pi / 180.0
        lon = np.linspace(lon_start, lon_end, nx)
        lat = np.linspace(lat_start, lat_end, ny)

        grid = cls.cartesian(
            x=lon,
            y=lat,
            z=z,
            mask=mask,
        )

        # compute grid spacing
        dx = radius * np.cos(grid.y * to_rad) * grid.dx * to_rad
        dy = radius * grid.dy * to_rad

        grid = cls.cartesian(
            x=lon, y=lat, z=z, dx=dx, dy=dy, dz=grid.dz, mask=grid.mask
        )
        return grid

    @staticmethod
    def _get_default_mask(shape: Shape) -> Array:
        mask = np.ones(shape, dtype=np.int8)
        mask[..., 0, :] = 0
        mask[..., -1, :] = 0
        mask[..., :, 0] = 0
        mask[..., :, -1] = 0
        return mask

    def __hash__(self):
        """Return object id as hashing."""
        return self._id

    def __eq__(self, other) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, Grid):
            return NotImplemented
        if self.__hash__() == other.__hash__():
            return True
        return self.__eq__grid__(other)

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def __eq__grid__(self, other) -> bool:
        """Return True if all fields are equal, except _id."""
        return all(
            (getattr(self, f.name) == getattr(other, f.name)).all()
            if f.name in ("x", "y", "mask", "dx", "dy", "z", "dz")
            else getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
            if f.name not in ("_id", "z", "dz")
        )


class StaggeredGrid(StaggeredGridBase[Grid]):
    """Staggered Grid.

    Subgrids are available as attributes `eta`, `u`, `v` and `q` where
    the grid box centers are located at `eta`, the faces are at `u` and `v`,
    and the vertices are located at `q`.

    Parameters
    ----------
    eta : Grid
      Grid of the box centeroids
    u : Grid
      Grid of the box faces perpendicular to the fist spatial dimension
    v : Grid
      Grid of the box faces perpendicular to the second spatial dimension
    q : Grid
      Grid of the box vertices
    """

    _gtype = Grid

    @classmethod
    def cartesian_c_grid(
        cls: Any,
        shift: GridShift = GridShift.LL,
        **grid_kwargs: dict[str, Any],
    ) -> "StaggeredGrid":
        """Generate a Cartesian Arakawa C-Grid.

        Arguments
        ---------
        shift : GridShift, default=GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See :py:class:`GridShift` for more details.
        **grid_kwargs : dict[str, Any]
          Keyword arguments are passed to :py:meth:`Grid.cartesian` to create
          the `eta` subgrid, i.e. the grid of the box centeroids.

        Returns
        -------
        StaggeredGrid
        """
        eta_grid = Grid.cartesian(**grid_kwargs)  # type: ignore

        u_x, u_y = (eta_grid.x + shift.value[0] * eta_grid.dx / 2, eta_grid.y)
        v_x, v_y = (eta_grid.x, eta_grid.y + shift.value[1] * eta_grid.dy / 2)
        q_x, q_y = (u_x, v_y)
        z = eta_grid.z if len(eta_grid.z != 0) else None
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
        return StaggeredGrid(eta=eta_grid, u=u_grid, v=v_grid, q=q_grid)

    @classmethod
    def regular_lat_lon_c_grid(
        cls,
        shift: GridShift = GridShift.LL,
        **kwargs,
    ) -> "StaggeredGrid":
        """Generate a Arakawa C-grid for a regular longitude/latitude grid.

        Arguments
        ---------
        shift : GridShift, default=GridShift.LL
          Direction of shift of staggered grids with respect to the eta-grid.
          See :py:class:`GridShift` for more details.
        **grid_kwargs : dict[str, Any]
          Keyword arguments are passed to :py:meth:`Grid.regular_lat_lon` to create
          the `eta` subgrid, i.e. the grid of the box centeroids.

        Returns
        -------
        StaggeredGrid
        """
        eta_grid = Grid.regular_lat_lon(**kwargs)
        dx = eta_grid._compute_grid_spacing(eta_grid.x, eta_grid.dim_x)
        dy = eta_grid._compute_grid_spacing(eta_grid.y, eta_grid.dim_y)

        u_x_start, u_x_end = (
            eta_grid.x.min() + shift.value[0] * dx.min() / 2,
            eta_grid.x.max() + shift.value[0] * dx.min() / 2,
        )
        u_kwargs = kwargs.copy()
        u_grid_mask = cls._compute_mask(
            cls._u_mask_from_eta,
            eta_grid,
            shift,
        )
        u_kwargs.update(dict(lon_start=u_x_start, lon_end=u_x_end, mask=u_grid_mask))
        u_grid = Grid.regular_lat_lon(**u_kwargs)

        v_y_start, v_y_end = (
            eta_grid.y.min() + shift.value[1] * dy.min() / 2,
            eta_grid.y.max() + shift.value[1] * dy.min() / 2,
        )
        v_kwargs = kwargs.copy()
        v_kwargs.update(
            dict(
                lat_start=v_y_start,
                lat_end=v_y_end,
                mask=cls._compute_mask(
                    cls._v_mask_from_eta,
                    eta_grid,
                    shift,
                ),
            )
        )
        v_grid = Grid.regular_lat_lon(**v_kwargs)

        q_kwargs = kwargs.copy()
        q_kwargs.update(
            dict(
                lon_start=u_x_start,
                lon_end=u_x_end,
                lat_start=v_y_start,
                lat_end=v_y_end,
                mask=cls._compute_mask(
                    cls._q_mask_from_eta,
                    eta_grid,
                    shift,
                ),
            )
        )
        q_grid = Grid.regular_lat_lon(**q_kwargs)

        return cls(eta_grid, u_grid, v_grid, q_grid)

    @staticmethod
    def _compute_mask(func: Callable[..., Array], from_grid: Grid, shift: GridShift):
        is_2D = from_grid.ndim < 3
        if is_2D:
            nk = 1
            mask = from_grid.mask[np.newaxis]
        else:
            nk = from_grid.shape[from_grid.dim_z]
            mask = from_grid.mask
        res = func(
            from_grid.shape[from_grid.dim_x],
            from_grid.shape[from_grid.dim_y],
            nk,
            mask,
            shift.value[0],
            shift.value[1],
        )
        if is_2D:
            return res[0, ...]
        else:
            return res

    @staticmethod
    @_numba_3D_grid_iterator_i8_parallel_over_k
    def _u_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: Array,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
        i_shift = (i + shift_x) % ni
        if (eta_mask[k, j, i] + eta_mask[k, j, i_shift]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_3D_grid_iterator_i8_parallel_over_k
    def _v_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: Array,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
        j_shift = (j + shift_y) % nj
        if (eta_mask[k, j, i] + eta_mask[k, j_shift, i]) == 2:
            return 1
        else:
            return 0

    @staticmethod
    @_numba_3D_grid_iterator_i8_parallel_over_k
    def _q_mask_from_eta(
        i: int,
        j: int,
        k: int,
        ni: int,
        nj: int,
        nk: int,
        eta_mask: Array,
        shift_x: int,
        shift_y: int,
    ) -> int:  # pragma: no cover
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
