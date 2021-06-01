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


class TestTerms:
    """Test RHS terms."""

    def test__zonal_pressure_gradient(self):
        """Test _zonal_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, _ = get_x_y(ni, nj, dx, dy)
        eta = np.copy(x)

        assert np.all(swe._zonal_pressure_gradient(
            ni, nj, eta, g, dx
        ) == -g * (np.roll(eta, -1, axis=0) - eta) / dx)

    # def test_iteration(self):
    #     """Test _iterate_over_grid_2D."""
    #     g = 1
    #     dx, dy = 1, 2
    #     ni, nj = 10, 5
    #     x, y = get_x_y(ni, nj, dx, dy)
    #     eta = np.copy(x)
    #     result = -1 * np.ones(x.shape)
    #     result[-1, :] = 9

    #     assert np.all(
    #         swe._iterate_over_grid_2D(
    #             swe._zonal_pressure_gradient_loop_body,
    #             ni, nj, args=(eta, g, dx)
    #         ) == result
    #     )

    def test_zonal_pressure_gradient(self):
        """Test zonal_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        eta = np.copy(x)
        result = -1 * np.ones(x.shape)
        result[-1, :] = 9

        params = swe.Parameters(g=g)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(np.zeros(x.shape), grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(eta, grid)
        )

        assert np.all(
            swe.zonal_pressure_gradient(
                state, grid, params
            ).eta.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.zonal_pressure_gradient(
                state, grid, params
            ).v.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.zonal_pressure_gradient(
                state, grid, params
            ).u.data == result
        )

    def test_meridional_pressure_gradient(self):
        """Test meridional_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        eta = np.copy(y)
        result = -1 * np.ones(y.shape)
        result[:, -1] = 4

        params = swe.Parameters(g=g)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(np.zeros(y.shape), grid),
            eta=swe.Variable(eta, grid)
        )

        assert np.all(
            swe.meridional_pressure_gradient(
                state, grid, params
            ).eta.data == 0.
        )
        assert np.all(
            swe.meridional_pressure_gradient(
                state, grid, params
            ).u.data == 0.
        )
        assert np.all(
            swe.meridional_pressure_gradient(
                state, grid, params
            ).v.data == result
        )

    def test_zonal_divergence(self):
        """Test zonal_divergence."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        u = np.copy(x)
        result = -1 * np.ones(x.shape)
        result[0, :] = 9

        params = swe.Parameters(H=H)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(np.zeros(x.shape), grid)
        )

        assert np.all(
            swe.zonal_divergence(
                state, grid, params
            ).u.data == 0.
        )
        assert np.all(
            swe.zonal_divergence(
                state, grid, params
            ).v.data == 0.
        )
        assert np.all(
            swe.zonal_divergence(
                state, grid, params
            ).eta.data == result
        )

    def test_meridional_divergence(self):
        """Test meridional_divergence."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        v = np.copy(y)
        result = -1 * np.ones(y.shape)
        result[:, 0] = 4

        params = swe.Parameters(H=H)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid)
        )

        assert np.all(
            swe.meridional_divergence(
                state, grid, params
            ).u.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.meridional_divergence(
                state, grid, params
            ).v.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.meridional_divergence(
                state, grid, params
            ).eta.data == result
        )

    def test_coriolis_u(self):
        """Test coriolis_u."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        u = np.ones(x.shape)
        result = -1 * np.ones(x.shape)

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(np.zeros(x.shape), grid),
            eta=swe.Variable(np.zeros(x.shape), grid)
        )

        assert np.all(
            swe.coriolis_u(
                state, grid, params
            ).u.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.coriolis_u(
                state, grid, params
            ).eta.data == np.zeros(x.shape)
        )
        assert np.all(
            swe.coriolis_u(
                state, grid, params
            ).v.data == result
        )

    def test__coriolis_v(self):
        """Test _coriolis_v."""
        f = 10e-4
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        v = np.arange(y.shape[0] * y.shape[1]).reshape(y.shape)
        result = f * .25 * (
            np.roll(v, -1, axis=0)
            + np.roll(v, 1, axis=1)
            + np.roll(v, (-1, 1), axis=(0, 1))
            + v
        )

        assert np.all(
            swe._coriolis_v(ni, nj, v, f) == result
        )

    def test_coriolis_v(self):
        """Test coriolis_v."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        v = np.arange(y.shape[0] * y.shape[1]).reshape(y.shape)
        result = f * .25 * (
            np.roll(v, -1, axis=0)
            + np.roll(v, 1, axis=1)
            + np.roll(v, (-1, 1), axis=(0, 1))
            + v
        )

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid)
        )

        assert np.all(
            swe.coriolis_v(
                state, grid, params
            ).v.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.coriolis_v(
                state, grid, params
            ).eta.data == np.zeros(y.shape)
        )
        assert np.all(
            swe.coriolis_v(
                state, grid, params
            ).u.data == result
        )
