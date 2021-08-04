"""Test the behaviour of the dataclasses."""
import numpy as np
import pytest

from multimodemodel import (
    Parameters,
    Grid,
    Variable,
    State,
    StaggeredGrid,
    GridShift,
)


def get_x_y(nx, ny, dx, dy):
    """Return 2D coordinate arrays."""
    return np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx)[::-1]


def get_test_mask(x):
    """Return a test ocean mask with the shape of the input coordinate array.

    The mask is zero at the outmost array elements, one elsewhere.
    """
    mask = np.ones(x.shape)
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    return mask


class TestParameters:
    """Test Parameters class."""

    def test_default_values(self):
        """Test default values."""
        p = Parameters()
        defaults = {
            "f": 0.0,
            "g": 9.81,
            "beta": 2.0 / (24 * 3600),
            "H": 1000.0,
            "dt": 1.0,
            "t_0": 0.0,
            "t_end": 3600.0,
            "write": 20.0,
            "r": 6_371_000.0,
        }
        for var, val in defaults.items():
            assert p.__getattribute__(var) == val


class TestGrid:
    """Test Grid class."""

    def test_post_init(self):
        """Test post_init."""
        nx, ny = 10, 5
        dx, dy = 1.0, 2.0
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)

        g1 = Grid(x=x, y=y, mask=mask)
        assert np.all(g1.dx == dx * np.ones(x.shape))
        assert np.all(g1.dy == dy * np.ones(y.shape))
        assert g1.len_x == nx
        assert g1.len_y == ny

    def test_dim_def(self):
        """Test dimension definition."""
        nx, ny = 10, 5
        dx, dy = 1.0, 2.0
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)

        g2 = Grid(x=x.T, y=y.T, mask=mask.T, dim_x=1, dim_y=0)
        assert g2.len_x == nx
        assert g2.len_y == ny
        assert np.all(g2.dx == dx)
        assert np.all(g2.dy == dy)
        assert g2.x.shape == (ny, nx)
        assert g2.y.shape == (ny, nx)
        assert g2.dx.shape == (ny, nx)
        assert g2.dy.shape == (ny, nx)

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
        mask = get_test_mask(lon)
        dx = d_lon * r * np.cos(lat * np.pi / 180) * np.pi / 180.0
        dy = d_lat * r * np.pi / 180.0

        grid = Grid.regular_lat_lon(lon_start, lon_end, lat_start, lat_end, nx, ny)

        assert np.all(grid.x == lon)
        assert np.all(grid.y == lat)
        assert np.all(np.round(grid.dx, 6) == np.round(dx, 6))
        assert np.all(np.round(grid.dy, 6) == np.round(dy, 6))
        assert np.all(grid.mask == mask)


class TestStaggeredGrid:
    """Test StaggeredGrid class."""

    def get_regular_staggered_grids(
        self, xs=0.0, xe=10.0, ys=0.0, ye=5.0, nx=11, ny=11
    ):
        """Set up test grids."""
        dx = (xe - xs) / (nx - 1)
        dy = (ye - ys) / (ny - 1)
        ll_grid = Grid.regular_lat_lon(xs, xe, ys, ye, nx, ny)
        lr_grid = Grid.regular_lat_lon(xs + dx / 2, xe + dx / 2, ys, ye, nx, ny)
        ur_grid = Grid.regular_lat_lon(
            xs + dx / 2, xe + dx / 2, ys + dy / 2, ye + dy / 2, nx, ny
        )
        ul_grid = Grid.regular_lat_lon(xs, xe, ys + dy / 2, ye + dy / 2, nx, ny)
        return ll_grid, lr_grid, ur_grid, ul_grid

    @pytest.mark.parametrize(
        "shift,order",
        [
            (GridShift.LL, "qveu"),
            (GridShift.LR, "vque"),
            (GridShift.UL, "uevq"),
            (GridShift.UR, "euqv"),
        ],
    )
    def test_regular_c_grid(self, shift, order):
        """Test staggering to Arakawa c-grid."""
        grids = self.get_regular_staggered_grids()
        q_grid, u_grid, v_grid, eta_grid = [grids[order.index(i)] for i in "quve"]

        staggered_grid = StaggeredGrid.regular_lat_lon_c_grid(
            shift=shift,
            lon_start=eta_grid.x.min(),
            lon_end=eta_grid.x.max(),
            lat_start=eta_grid.y.min(),
            lat_end=eta_grid.y.max(),
            nx=eta_grid.len_x,
            ny=eta_grid.len_y,
        )

        assert np.all(staggered_grid.q.x == q_grid.x)
        assert np.all(staggered_grid.q.y == q_grid.y)
        assert np.all(staggered_grid.u.x == u_grid.x)
        assert np.all(staggered_grid.u.y == u_grid.y)
        assert np.all(staggered_grid.v.x == v_grid.x)
        assert np.all(staggered_grid.v.y == v_grid.y)
        assert np.all(staggered_grid.eta.x == eta_grid.x)
        assert np.all(staggered_grid.eta.y == eta_grid.y)


class TestVariable:
    """Test Variable class."""

    def test_add_data(self):
        """Test variable summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)
        g1 = Grid(x, y, mask)

        d1 = np.zeros_like(g1.x) + 1.0
        d2 = np.zeros_like(g1.x) + 2.0
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g1)
        v3 = v1 + v2
        assert np.all(v3.data == 3.0)

    def test_grid_mismatch(self):
        """Test grid mismatch detection."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)
        g1 = Grid(x, y, mask)
        g2 = Grid(x, y, mask)
        d1 = np.zeros_like(g1.x) + 1.0
        d2 = np.zeros_like(g1.x) + 2.0
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g2)
        with pytest.raises(ValueError) as excinfo:
            _ = v1 + v2
        assert "Try to add variables defined on different grids." in str(excinfo.value)

    def test_not_implemented_add(self):
        """Test missing summation implementation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)
        g1 = Grid(x, y, mask)
        d1 = np.zeros_like(g1.x) + 1.0
        v1 = Variable(d1, g1)
        with pytest.raises(TypeError) as excinfo:
            _ = v1 + 1.0
        assert "unsupported operand type(s)" in str(excinfo.value)


class TestState:
    """Test State class."""

    def test_add(self):
        """Test state summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)
        g1 = Grid(x, y, mask)
        d1 = np.zeros_like(g1.x) + 1.0
        s1 = State(
            u=Variable(d1, g1),
            v=Variable(d1, g1),
            eta=Variable(d1, g1),
        )
        s2 = State(Variable(d1 * 2, g1), Variable(d1 * 2, g1), Variable(d1 * 2, g1))
        s3 = s1 + s2
        assert np.all(s3.u.data == 3.0)
        assert np.all(s3.v.data == 3.0)
        assert np.all(s3.eta.data == 3.0)
