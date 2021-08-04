"""Test the behaviour of the term functions."""
import numpy as np
import multimodemodel as swe


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
        mask = get_test_mask(x)
        eta = np.copy(x)
        dx_a = dx * np.ones(x.shape)
        dy_a = dy * np.ones(x.shape)

        assert np.all(
            swe._zonal_pressure_gradient(
                ni, nj, eta, mask, g, dx_a, dy_a, dy_a  # type: ignore
            )
            == -g * (eta * mask - np.roll(eta * mask, 1, axis=0)) / dx
        )

    def test_zonal_pressure_gradient(self):
        """Test zonal_pressure_gradient."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        eta = np.copy(x) * mask
        result = -g * mask * (eta - np.roll(eta, 1, axis=0)) / dx

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
        result = -g * mask * (eta - np.roll(eta, 1, axis=1)) / dy

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
        u = np.copy(x) * mask
        result = -H * mask * (np.roll(u, -1, axis=0) - u) / dx

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
        v = np.copy(y) * mask
        result = -H * mask * (np.roll(v, -1, axis=1) - v) / dy

        params = swe.Parameters(H=H)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid),
        )

        inc = swe.meridional_divergence(state, params)
        assert np.all(inc.u.data == 0.0)
        assert np.all(inc.v.data == 0.0)
        assert np.all(inc.eta.data == result)

    def test_coriolis_u(self):
        """Test coriolis_u."""
        f = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        u = np.ones(x.shape) * mask
        result = (
            -mask
            * (
                np.roll(np.roll(u, -1, axis=0), 1, axis=1)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + u
            )
            / 4.0
        )

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
        v = np.ones(y.shape) * mask
        result = (
            mask
            * (
                np.roll(np.roll(v, 1, axis=0), -1, axis=1)
                + np.roll(v, 1, axis=0)
                + np.roll(v, -1, axis=1)
                + v
            )
            / 4.0
        )

        params = swe.Parameters(f=f)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(np.zeros(y.shape), grid),
        )

        inc = swe.coriolis_v(state, params)
        print(inc.u.data)
        print(result)
        assert np.all(inc.v.data == np.zeros(y.shape))
        assert np.all(inc.eta.data == np.zeros(y.shape))
        assert np.all(inc.u.data == result)
