# flake8: noqa
"""Test the behavior of the grid classes."""
import numpy as np
import pytest

from multimodemodel import (
    Grid,
    StaggeredGrid,
    GridShift,
)

try:
    import xarray as xr

    xr_version = xr.__version__
    has_xarray = True
except ModuleNotFoundError:
    has_xarray = False


grid_order = {
    GridShift.LL: "qveu",
    GridShift.LR: "vque",
    GridShift.UL: "uevq",
    GridShift.UR: "euqv",
}


def get_x_y(nx=10.0, ny=10.0, dx=1.0, dy=2.0):
    """Return 2D coordinate arrays."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    assert np.all(X[0, :] == x)
    assert np.all(Y[:, 0] == y)
    return X, Y


def get_x_y_z(nx=10.0, ny=10.0, nz=1, dx=1.0, dy=2.0):
    """Return 2D coordinate arrays."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.arange(nz)
    assert np.all(X[0, :] == x)
    assert np.all(Y[:, 0] == y)
    return X, Y, Z


def get_test_mask(shape):
    """Return a test ocean mask with the shape of the input coordinate array.

    The mask is zero at the outmost array elements, one elsewhere.
    """
    mask = np.ones(shape, dtype=int)
    mask[..., 0, :] = 0
    mask[..., -1, :] = 0
    mask[..., :, 0] = 0
    mask[..., :, -1] = 0
    return mask


class TestGrid:
    """Test Grid class."""

    def test_post_init(self):
        """Test post_init."""
        nx, ny, nz = 10, 5, 1
        dx, dy = 1.0, 2.0
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)

        g1 = Grid(x=x, y=y, mask=mask)
        assert np.all(g1.dx == dx * np.ones(x.shape))
        assert np.all(g1.dy == dy * np.ones(y.shape))
        assert len(g1.dz) == 0
        assert len(g1.z) == 0
        assert g1.shape[g1.dim_x] == nx
        assert g1.shape[g1.dim_y] == ny
        assert g1.ndim == 2

    def test_post_init_3D(self):
        """Test post_init."""
        nx, ny, nz = 10, 5, 5
        dx, dy, dz = 1.0, 2.0, 2.5
        x, y = get_x_y(nx, ny, dx, dy)
        z = np.arange(nz) * dz
        shape = z.shape + x.shape
        mask = get_test_mask(shape)

        g1 = Grid(x=x, y=y, z=z, mask=mask)
        assert np.all(g1.dx == dx * np.ones(x.shape))
        assert np.all(g1.dy == dy * np.ones(y.shape))
        assert g1.shape[g1.dim_x] == nx
        assert g1.shape[g1.dim_y] == ny
        assert g1.shape[g1.dim_z] == nz
        assert g1.ndim == 3

    def test_post_init_with_computed_grid_spacing(self):
        """Test post_init."""
        nx, ny, nz = 10, 5, 7
        dx, dy, dz = 1.0, 2.0, 23.0
        x, y = get_x_y(nx, ny, dx, dy)
        z = np.arange(nz) * dz
        mask = get_test_mask((nz, ny, nx))

        g1 = Grid(x=x, y=y, z=z, mask=mask)
        assert np.all(g1.dx == dx * np.ones(x.shape))
        assert np.all(g1.dy == dy * np.ones(y.shape))
        assert g1.shape == (nz, ny, nx)

    def test_post_init_with_initialized_grid_spacing(self):
        """Test post_init."""
        nx, ny = 10, 5
        dx, dy = 1.0, 2.0
        x, y = get_x_y(nx, ny, dx, dy)
        dx = np.ones_like(x) * 2.0
        dy = np.ones_like(y) * 1.0
        mask = get_test_mask(x.shape)

        g1 = Grid(x=x, y=y, mask=mask, dx=dx, dy=dy)
        assert (g1.dx == 2).all()
        assert (g1.dy == 1).all()

    def test_grid_default_mask(self):
        """Test default grid setting."""
        nx, ny, nz = 10, 5, 5
        dx, dy, dz = 1.0, 2.0, 2.5
        x, y = get_x_y(nx, ny, dx, dy)
        z = np.arange(nz) * dz
        g = Grid(x=x, y=y, z=z)
        mask = get_test_mask(g.shape)
        assert np.all(g.mask == mask)
        assert g.mask.dtype == np.int8  # type: ignore

    def test_grid_raises_on_mask_missmatch(self):
        """Test exception raises on mask shape missmatch."""
        x, y = get_x_y()
        mask = get_test_mask(y.shape)
        with pytest.raises(ValueError, match="Mask shape not matching grid shape"):
            _ = Grid(x=x, y=y, mask=mask[:, ::2])

    def test_dim_def(self):
        """Test dimension definition."""
        nx, ny, nz = 10, 5, 1
        dx, dy = 1.0, 2.0
        x, y, z = get_x_y_z(nx, ny, nz, dx, dy)
        g2 = Grid(x=x, y=y, z=z, mask=get_test_mask((nz, ny, nx)))
        assert g2.shape[g2.dim_x] == nx
        assert g2.shape[g2.dim_y] == ny
        assert g2.shape[g2.dim_z] == nz
        assert np.all(g2.dx == dx)
        assert np.all(g2.dy == dy)
        assert g2.x.shape == (ny, nx)
        assert g2.y.shape == (ny, nx)
        assert g2.dx.shape == (ny, nx)
        assert g2.dy.shape == (ny, nx)

    def test_comparison_of_identical_returns_true(self):
        """Test comparison of references."""
        nx, ny = 10, 20
        dx, dy = 1.0, 0.5

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)

        g = Grid.cartesian(x, y)
        g2 = g
        assert g2 == g

    def test_comparison_of_same_returns_true(self):
        """Test comparison of references."""
        nx, ny = 10, 20
        dx, dy = 1.0, 0.5

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)

        g = Grid.cartesian(x=x, y=y)
        g2 = Grid.cartesian(x=x, y=y)
        assert g2 == g

    def test_comparison_of_different_returns_false(self):
        """Test comparison of references."""
        nx, ny = 10, 20
        dx, dy = 1.0, 0.5

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)

        g = Grid.cartesian(x=x, y=y)
        g2 = Grid.cartesian(x=x, y=y + 1)
        assert g2 != g

    def test_comparison_with_other_returns_false(self):
        nx, ny = 10, 20
        dx, dy = 1.0, 0.5

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)

        g = Grid.cartesian(x=x, y=y)
        assert g != 5

    def test_cartesian_grid(self):
        """Test construction of cartesian grid."""
        nx, ny = 10, 20
        dx, dy = 1.0, 0.5

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)

        g = Grid.cartesian(x, y)
        assert g.x.shape == g.y.shape == (ny, nx)
        assert np.all(np.diff(g.x, 1, g.dim_x) == dx)
        assert np.all(np.diff(g.y, 1, g.dim_y) == dy)
        assert g.mask.ndim == 2
        assert g.ndim == 2

    def test_cartesian_grid_3D(self):
        """Test construction of cartesian grid."""
        nx, ny, nz = 10, 20, 10
        dx, dy, dz = 1.0, 0.5, 0.25

        x = np.arange(0, nx * dx, dx)
        y = np.arange(0, ny * dy, dy)
        z = np.arange(0, nz * dz, dz)

        g = Grid.cartesian(x, y, z)
        assert g.x.shape == g.y.shape == (ny, nx)
        assert g.z.shape == (nz,)
        assert np.all(np.diff(g.x, 1, g.dim_x) == dx)
        assert np.all(np.diff(g.y, 1, g.dim_y) == dy)
        assert g.mask.ndim == 3
        assert g.ndim == 3

    def test_regular_lat_lon(self):
        """Test lat_lon grid generating classmethod.

        Assert the grid spacing with six digits precision.
        """
        r = 6371000.0
        lon_start, lon_end = 0.0, 10.0
        lat_start, lat_end = 0.0, 5.0
        nx, ny = 11, 6
        d_lon, d_lat = 1.0, 1.0
        lon, lat = get_x_y(nx, ny, d_lon, d_lat)
        mask = get_test_mask(lon.shape)
        dx = d_lon * r * np.cos(lat * np.pi / 180) * np.pi / 180.0
        dy = d_lat * r * np.pi / 180.0

        grid = Grid.regular_lat_lon(lon_start, lon_end, lat_start, lat_end, nx, ny)

        assert np.all(grid.x == lon)
        assert np.all(grid.y == lat)
        assert np.all(np.round(grid.dx, 6) == np.round(dx, 6))
        assert np.all(np.round(grid.dy, 6) == np.round(dy, 6))
        assert np.all(grid.mask == mask)

    def test_regular_lat_lon_3D(self):
        """Test lat_lon grid generating classmethod.

        Assert the grid spacing with six digits precision.
        """
        lon_start, lon_end = 0.0, 10.0
        lat_start, lat_end = 0.0, 5.0
        nx, ny, nz = 11, 6, 10
        d_lon, d_lat, dz = 1.0, 1.0, 0.25
        lon, lat = get_x_y(nx, ny, d_lon, d_lat)
        z = np.arange(nz) * dz

        grid = Grid.regular_lat_lon(lon_start, lon_end, lat_start, lat_end, nx, ny, z)

        assert np.all(grid.x == lon)
        assert np.all(grid.y == lat)
        assert np.all(grid.z == z)


class TestStaggeredGrid:
    """Test StaggeredGrid class."""

    def get_regular_staggered_grids(
        self, xs=0.0, xe=10.0, ys=0.0, ye=5.0, nx=11, ny=11, z=None
    ):
        """Set up spherical test grids."""
        dx = (xe - xs) / (nx - 1)
        dy = (ye - ys) / (ny - 1)
        ll_grid = Grid.regular_lat_lon(
            lon_start=xs, lon_end=xe, lat_start=ys, lat_end=ye, nx=nx, ny=ny, z=z
        )
        lr_grid = Grid.regular_lat_lon(
            lon_start=xs + dx / 2,
            lon_end=xe + dx / 2,
            lat_start=ys,
            lat_end=ye,
            nx=nx,
            ny=ny,
            z=z,
        )
        ur_grid = Grid.regular_lat_lon(
            lon_start=xs + dx / 2,
            lon_end=xe + dx / 2,
            lat_start=ys + dy / 2,
            lat_end=ye + dy / 2,
            nx=nx,
            ny=ny,
            z=z,
        )
        ul_grid = Grid.regular_lat_lon(
            lon_start=xs,
            lon_end=xe,
            lat_start=ys + dy / 2,
            lat_end=ye + dy / 2,
            nx=nx,
            ny=ny,
            z=z,
        )
        return ll_grid, lr_grid, ur_grid, ul_grid

    def get_cartesian_staggered_grids(self, x, y, z, dx, dy):
        """Set up cartesian test grids."""
        ll_grid = Grid.cartesian(x=x, y=y, z=z)
        lr_grid = Grid.cartesian(x=x + dx / 2, y=y, z=z)
        ul_grid = Grid.cartesian(x, y + dy / 2, z)
        ur_grid = Grid.cartesian(x + dx / 2, y + dy / 2, z)
        return ll_grid, lr_grid, ur_grid, ul_grid

    def test__grid_type_is_Grid(self):
        """Test value of _stype attribute."""
        assert StaggeredGrid._grid_type() is Grid

    @pytest.mark.parametrize(
        "shift",
        [
            GridShift.LL,
            GridShift.LR,
            GridShift.UL,
            GridShift.UR,
        ],
    )
    def test_cartesian_c_grid(self, shift):
        """Test staggering of cartesian grid to Arakawa c-grid."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        grids = self.get_cartesian_staggered_grids(x, y, None, dx, dy)
        q_grid, u_grid, v_grid, eta_grid = [
            grids[grid_order[shift].index(i)] for i in "quve"
        ]
        kwargs = dict(x=eta_grid.x[0], y=eta_grid.y[:, 0])
        staggered_grid = StaggeredGrid.cartesian_c_grid(shift=shift, **kwargs)
        assert np.all(staggered_grid.q.x == q_grid.x)
        assert np.all(staggered_grid.q.y == q_grid.y)
        assert np.all(staggered_grid.u.x == u_grid.x)
        assert np.all(staggered_grid.u.y == u_grid.y)
        assert np.all(staggered_grid.v.x == v_grid.x)
        assert np.all(staggered_grid.v.y == v_grid.y)
        assert np.all(staggered_grid.eta.x == eta_grid.x)
        assert np.all(staggered_grid.eta.y == eta_grid.y)

    @pytest.mark.parametrize(
        "shift",
        [
            GridShift.LL,
            GridShift.LR,
            GridShift.UL,
            GridShift.UR,
        ],
    )
    def test_cartesian_c_grid_3D(self, shift):
        """Test staggering of cartesian grid to Arakawa c-grid."""
        nx, ny, nz = 20, 10, 5
        dx, dy, dz = 1.0, 0.25, 2.0
        x, y, z = (
            np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy), (nz, dz))
        )
        grids = self.get_cartesian_staggered_grids(x, y, z, dx, dy)
        q_grid, u_grid, v_grid, eta_grid = [
            grids[grid_order[shift].index(i)] for i in "quve"
        ]
        kwargs = dict(x=eta_grid.x[0, :], y=eta_grid.y[:, 0], z=z)
        staggered_grid = StaggeredGrid.cartesian_c_grid(shift=shift, **kwargs)
        assert np.all(staggered_grid.q.x == q_grid.x)
        assert np.all(staggered_grid.q.y == q_grid.y)
        assert np.all(staggered_grid.q.z == q_grid.z)
        assert np.all(staggered_grid.u.x == u_grid.x)
        assert np.all(staggered_grid.u.y == u_grid.y)
        assert np.all(staggered_grid.u.z == u_grid.z)
        assert np.all(staggered_grid.v.x == v_grid.x)
        assert np.all(staggered_grid.v.y == v_grid.y)
        assert np.all(staggered_grid.v.z == v_grid.z)
        assert np.all(staggered_grid.eta.x == eta_grid.x)
        assert np.all(staggered_grid.eta.y == eta_grid.y)
        assert np.all(staggered_grid.eta.z == eta_grid.z)

    @pytest.mark.parametrize(
        "shift",
        [
            GridShift.LL,
            GridShift.LR,
            GridShift.UL,
            GridShift.UR,
        ],
    )
    def test_regular_c_grid(self, shift):
        """Test staggering of regular lon/lat grid to Arakawa c-grid."""
        grids = self.get_regular_staggered_grids()
        q_grid, u_grid, v_grid, eta_grid = [
            grids[grid_order[shift].index(i)] for i in "quve"
        ]

        args = dict(
            lon_start=eta_grid.x.min(),
            lon_end=eta_grid.x.max(),
            lat_start=eta_grid.y.min(),
            lat_end=eta_grid.y.max(),
            nx=eta_grid.shape[eta_grid.dim_x],
            ny=eta_grid.shape[eta_grid.dim_y],
        )

        staggered_grid = StaggeredGrid.regular_lat_lon_c_grid(shift=shift, **args)

        assert np.all(staggered_grid.q.x == q_grid.x)
        assert np.all(staggered_grid.q.y == q_grid.y)
        assert np.all(staggered_grid.u.x == u_grid.x)
        assert np.all(staggered_grid.u.y == u_grid.y)
        assert np.all(staggered_grid.v.x == v_grid.x)
        assert np.all(staggered_grid.v.y == v_grid.y)
        assert np.all(staggered_grid.eta.x == eta_grid.x)
        assert np.all(staggered_grid.eta.y == eta_grid.y)

    @pytest.mark.parametrize(
        "shift",
        [
            GridShift.LL,
            GridShift.LR,
            GridShift.UL,
            GridShift.UR,
        ],
    )
    def test_regular_c_grid_3D(self, shift):
        """Test staggering of regular lon/lat grid to Arakawa c-grid."""
        nz, dz = 10, 0.25
        z = np.arange(nz) * dz
        grids = self.get_regular_staggered_grids(z=z)
        q_grid, u_grid, v_grid, eta_grid = [
            grids[grid_order[shift].index(i)] for i in "quve"
        ]

        args = dict(
            lon_start=eta_grid.x.min(),
            lon_end=eta_grid.x.max(),
            lat_start=eta_grid.y.min(),
            lat_end=eta_grid.y.max(),
            z=z,
            nx=eta_grid.shape[eta_grid.dim_x],
            ny=eta_grid.shape[eta_grid.dim_y],
        )

        staggered_grid = StaggeredGrid.regular_lat_lon_c_grid(shift=shift, **args)

        assert np.all(staggered_grid.q.x == q_grid.x)
        assert np.all(staggered_grid.q.y == q_grid.y)
        assert np.all(staggered_grid.q.z == q_grid.z)
        assert np.all(staggered_grid.u.x == u_grid.x)
        assert np.all(staggered_grid.u.y == u_grid.y)
        assert np.all(staggered_grid.u.z == u_grid.z)
        assert np.all(staggered_grid.v.x == v_grid.x)
        assert np.all(staggered_grid.v.y == v_grid.y)
        assert np.all(staggered_grid.v.z == v_grid.z)
        assert np.all(staggered_grid.eta.x == eta_grid.x)
        assert np.all(staggered_grid.eta.y == eta_grid.y)
        assert np.all(staggered_grid.eta.z == eta_grid.z)

    @pytest.mark.parametrize(
        "shift",
        [
            GridShift.LL,
            GridShift.LR,
            GridShift.UL,
            GridShift.UR,
        ],
    )
    def test_mask(self, shift):
        """Test derivation of land/ocean mask."""
        z = np.array([0])
        grids = self.get_regular_staggered_grids(z=z)
        eta_grid, _, _, _ = [grids[grid_order[shift].index(i)] for i in "equv"]
        mask = get_test_mask(eta_grid.shape)

        u_mask = (
            (mask == 1)
            & (np.roll(mask, axis=eta_grid.dim_x, shift=-1 * shift.value[0]) == 1)
        ).astype(int)
        v_mask = (
            (mask == 1)
            & (np.roll(mask, axis=eta_grid.dim_y, shift=-1 * shift.value[1]) == 1)
        ).astype(int)
        q_mask = (
            (mask == 1)
            & (np.roll(mask, axis=eta_grid.dim_x, shift=-1 * shift.value[0]) == 1)
            & (np.roll(mask, axis=eta_grid.dim_y, shift=-1 * shift.value[1]) == 1)
            & (
                np.roll(
                    np.roll(mask, axis=eta_grid.dim_y, shift=-1 * shift.value[1]),
                    axis=eta_grid.dim_x,
                    shift=-1 * shift.value[0],
                )
                == 1
            )
        ).astype(int)

        staggered_grid = StaggeredGrid.regular_lat_lon_c_grid(
            shift=shift,
            lon_start=eta_grid.x.min(),
            lon_end=eta_grid.x.max(),
            lat_start=eta_grid.y.min(),
            lat_end=eta_grid.y.max(),
            nx=eta_grid.shape[eta_grid.dim_x],  # type: ignore
            ny=eta_grid.shape[eta_grid.dim_y],  # type: ignore
            z=eta_grid.z,  # type: ignore
            mask=mask,  # type: ignore
        )

        assert np.all(staggered_grid.u.mask == u_mask)
        assert np.all(staggered_grid.v.mask == v_mask)
        assert np.all(staggered_grid.q.mask == q_mask)
        assert np.all(staggered_grid.eta.mask == mask)

    def test_comparison_of_identical_returns_true(self):
        grids = self.get_regular_staggered_grids()
        grids2 = grids
        assert grids == grids2

    def test_comparison_of_same_returns_true(self):
        grids = self.get_regular_staggered_grids()
        grids2 = self.get_regular_staggered_grids()
        assert grids == grids2

    def test_comparison_of_different_returns_false(self):
        grids = self.get_regular_staggered_grids(ys=1)
        grids2 = self.get_regular_staggered_grids(ys=2)
        assert grids != grids2

    def test_comparison_of_wrong_type_returns_false(self):
        grids = self.get_regular_staggered_grids()
        assert grids != 5
