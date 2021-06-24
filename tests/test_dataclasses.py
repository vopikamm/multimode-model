"""Test the behaviour of the dataclasses."""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shallow_water_eqs import Parameters, Grid, Variable, State  # noqa: E402


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
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)

        g1 = Grid(x=x, y=y, mask=mask, e_x=e_x, e_y=e_y)
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

        g2 = Grid(x=x.T, y=y.T, mask=mask.T, e_x=x.T, e_y=y.T, dim_x=1, dim_y=0)
        assert g2.len_x == nx
        assert g2.len_y == ny
        assert np.all(g2.dx == dx)
        assert np.all(g2.dy == dy)
        assert g2.x.shape == (ny, nx)
        assert g2.y.shape == (ny, nx)
        assert g2.dx.shape == (ny, nx)
        assert g2.dy.shape == (ny, nx)
        assert g2.e_x.shape == (ny, nx)
        assert g2.e_y.shape == (ny, nx)


class TestVariable:
    """Test Variable class."""

    def test_add_data(self):
        """Test variable summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x)
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        g1 = Grid(x, y, mask, e_x, e_y)

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
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        g1 = Grid(x, y, mask, e_x, e_y)
        g2 = Grid(x, y, mask, e_x, e_y)
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
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        g1 = Grid(x, y, mask, e_x, e_y)
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
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        g1 = Grid(x, y, mask, e_x, e_y)
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
