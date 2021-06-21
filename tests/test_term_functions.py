"""Test the behaviour of the term functions."""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shallow_water_eqs as swe  # noqa: E402


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


class TestTerms:
    """Test RHS terms."""

    def test__zonal_pressure_gradient(self):
        """Test _zonal_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, _ = get_x_y(ni, nj, dx, dy)
        eta = np.copy(x)

        assert np.all(
            swe._zonal_pressure_gradient(ni, nj, eta, g, dx)
            == -g * (np.roll(eta, -1, axis=0) - eta) / dx
        )

    def test_zonal_pressure_gradient(self):
        """Test zonal_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        eta = np.copy(x)
        result = -1 * np.ones(x.shape) * mask

        params = swe.Parameters(g=g)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(x.shape), grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(eta, grid),
        )

        inc = swe.zonal_pressure_gradient(state, params)

        assert np.all(inc.eta.data == np.zeros(x.shape))
        assert np.all(inc.v.data == np.zeros(x.shape))
        assert np.all(inc.u.data == result)

    def test_meridional_pressure_gradient(self):
        """Test meridional_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        eta = np.copy(y)
        result = -1 * np.ones(y.shape) * mask

        params = swe.Parameters(g=g)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(np.zeros(y.shape), grid),
            eta=swe.Variable(eta, grid),
        )

        inc = swe.meridional_pressure_gradient(state, params)

        assert np.all(inc.eta.data == 0.0)
        assert np.all(inc.u.data == 0.0)
        assert np.all(inc.v.data == result)

    def test_zonal_divergence(self):
        """Test zonal_divergence."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        u = np.copy(x)
        result = -1 * np.ones(x.shape) * mask

        params = swe.Parameters(H=H)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(np.zeros(x.shape), grid),
        )

        inc = swe.zonal_divergence(state, params)

        assert np.all(inc.u.data == 0.0)
        assert np.all(inc.v.data == 0.0)
        assert np.all(inc.eta.data == result)

    def test_meridional_divergence(self):
        """Test meridional_divergence."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        v = np.copy(y)
        result = -1 * np.ones(y.shape) * mask

        params = swe.Parameters(H=H)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid),
        )

        inc = swe.meridional_divergence(state, params)

        assert np.all(inc.u.data == np.zeros(y.shape))
        assert np.all(inc.v.data == np.zeros(y.shape))
        assert np.all(inc.eta.data == result)

    def test_coriolis_u(self):
        """Test coriolis_u."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        u = 4 * np.ones(x.shape)
        result = -4 * np.ones(x.shape)

        result[1, :] = -2
        result[:, -2] = -2
        result[1, -2] = -1

        result = result * mask

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(np.zeros(x.shape), grid),
        )

        inc = swe.coriolis_u(state, params)

        assert np.all(inc.u.data == np.zeros(x.shape))
        assert np.all(inc.eta.data == np.zeros(x.shape))
        assert np.all(inc.v.data == result)

    def test_coriolis_v(self):
        """Test coriolis_v."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        v = 4 * np.ones(y.shape)
        result = 4 * np.ones(y.shape)

        result[-2, :] = 2
        result[:, 1] = 2
        result[-2, 1] = 1

        result = result * mask

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid),
        )

        inc = swe.coriolis_v(state, params)

        assert np.all(inc.v.data == np.zeros(y.shape))
        assert np.all(inc.eta.data == np.zeros(y.shape))
        assert np.all(inc.u.data == result)
