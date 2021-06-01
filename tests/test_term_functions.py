"""Test the behaviour of the term functions."""
import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import shallow_water_eqs as swe  # noqa: E402


def get_x_y(nx, ny, dx, dy):
    """Return 2D coordinate arrays."""
    return np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx)[::-1]

def get_test_mask(x):
    """Returns a test ocean mask with the shape of the input coordinate array.
       The mask is zero at the outmost array elements, one elsewhere"""
    mask        = np.ones(x.shape)
    mask[0,:]   = 0.
    mask[-1,:]  = 0.
    mask[:,0]   = 0.
    mask[:,-1]  = 0.
    return mask


class TestTerms:
    """Test RHS terms."""

    def test_zonal_pressure_gradient_loopbody(self):
        """Test _zonal_pressure_gradient_loopbody."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        i, j = 5, 2
        x, y = get_x_y(ni, nj, dx, dy)
        eta = np.copy(x)

        assert swe._zonal_pressure_gradient_loop_body(
            eta, g, dx, i, j, ni, nj
        ) == -1
        assert swe._zonal_pressure_gradient_loop_body(
            eta, g, dx, ni - 1, j, ni, nj
        ) == 9

    def test_iteration(self):
        """Test _iterate_over_grid_2D."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        eta = np.copy(x)
        result = -1 * np.ones(x.shape)
        result[-1, :] = 9


        assert np.all(
            swe._iterate_over_grid_2D(
                swe._zonal_pressure_gradient_loop_body,
                ni, nj, args=(eta, g, dx)
            ) == result
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
            eta=swe.Variable(eta, grid)
        )

        assert np.all(
            swe.zonal_pressure_gradient(
                state, params
            ).eta.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.zonal_pressure_gradient(
                state, params
            ).v.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.zonal_pressure_gradient(
                state, params
            ).u.data == result
        )

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
            eta=swe.Variable(eta, grid)
        )

        assert np.all(
            swe.meridional_pressure_gradient(
                state, params
            ).eta.data == 0.
        )
        assert np.all(
            swe.meridional_pressure_gradient(
                state, params
            ).u.data == 0.
        )
        assert np.all(
            swe.meridional_pressure_gradient(
                state, params
            ).v.data == result
        )

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
            eta=swe.Variable(np.zeros(x.shape), grid)
        )

        assert np.all(
            swe.zonal_divergence(
                state, params
            ).u.data == 0.
        )
        assert np.all(
            swe.zonal_divergence(
                state, params
            ).v.data == 0.
        )
        assert np.all(
            swe.zonal_divergence(
                state, params
            ).eta.data == result
        )

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
            eta=swe.Variable(np.zeros(y.shape), grid)
        )

        assert np.all(
            swe.meridional_divergence(
                state, params
            ).u.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.meridional_divergence(
                state, params
            ).v.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.meridional_divergence(
                state, params
            ).eta.data == result
        )

    def test_coriolis_u(self):
        """Test coriolis_u."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        u    = 4 * np.ones(x.shape)
        result = -4 * np.ones(x.shape)

        result[1,:]   = -2
        result[:, -2] = -2
        result[1,-2]  = -1

        result = result * mask


        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(np.zeros(x.shape), grid)
        )

        assert np.all(
            swe.coriolis_u(
                state, params
            ).u.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.coriolis_u(
                state, params
            ).eta.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.coriolis_u(
                state, params
            ).v.data == result
        )

    def test_coriolis_v(self):
        """Test coriolis_v."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        v    = 4 * np.ones(y.shape)
        result = 4 * np.ones(y.shape)

        result[-2,:]   = 2
        result[:, 1]   = 2
        result[-2,1]   = 1

        result = result * mask

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid)
        )

        assert np.all(
            swe.coriolis_v(
                state, params
            ).v.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.coriolis_v(
                state, params
            ).eta.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.coriolis_v(
                state, params
            ).u.data == result
        )
