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

    def test__pressure_gradient_i(self):
        """Test _pressure_gradient_i."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, _ = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        mask_u = mask * np.roll(mask, 1, axis=0)
        eta = np.copy(x)
        dx_a = dx * np.ones(x.shape)

        oracle = -g * (eta * mask - np.roll(eta * mask, 1, axis=0)) / dx_a * mask_u
        result = swe._pressure_gradient_i(ni, nj, eta, g, dx_a, mask_u)  # type: ignore

        assert np.all(result == oracle)

    def test_pressure_gradient_i(self):
        """Test pressure_gradient_i."""
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

        inc = swe.pressure_gradient_i(state, params)

        assert inc.eta.data is None
        assert inc.v.data is None
        assert np.all(inc.u.data == result)

    def test_pressure_gradient_j(self):
        """Test pressure_gradient_j."""
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

        inc = swe.pressure_gradient_j(state, params)

        assert inc.eta.data is None
        assert inc.u.data is None
        assert np.all(inc.v.data == result)

    def test_divergence_i(self):
        """Test divergence_i."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)
        mask_u = mask_eta * np.roll(mask_eta, 1, axis=0)
        u = np.copy(x) * mask_u
        result = -H * (np.roll(u, -1, axis=0) - u) / dx

        params = swe.Parameters(H=H)
        grid_u = swe.Grid(x, y, mask_u)
        grid_eta = swe.Grid(x, y, mask_eta)
        state = swe.State(
            u=swe.Variable(u, grid_u),
            v=swe.Variable(np.zeros(x.shape), grid_u),
            eta=swe.Variable(np.zeros(x.shape), grid_eta),
        )

        inc = swe.divergence_i(state, params)

        assert inc.u.data is None
        assert inc.v.data is None
        assert np.all(inc.eta.data == result)

    def test_divergence_j(self):
        """Test divergence_j."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)
        mask_v = mask_eta * np.roll(mask_eta, 1, axis=1)
        v = np.copy(y) * mask_v
        result = -H * (np.roll(v, -1, axis=1) - v) / dy

        params = swe.Parameters(H=H)
        grid_v = swe.Grid(x, y, mask_v)
        grid_eta = swe.Grid(x, y, mask_eta)
        state = swe.State(
            u=swe.Variable(np.zeros(y.shape), grid_v),
            v=swe.Variable(v, grid_v),
            eta=swe.Variable(np.zeros(y.shape), grid_eta),
        )

        inc = swe.divergence_j(state, params)
        assert inc.u.data is None
        assert inc.v.data is None
        assert np.all(inc.eta.data == result)

    def test_coriolis_j(self):
        """Test coriolis_j."""
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

        inc = swe.coriolis_j(state, params)
        assert inc.u.data is None
        assert inc.eta.data is None
        assert np.all(inc.v.data == result)

    def test_coriolis_i(self):
        """Test coriolis_i."""
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

        inc = swe.coriolis_i(state, params)
        print(inc.u.data)
        print(result)
        assert inc.v.data is None
        assert inc.eta.data is None
        assert np.all(inc.u.data == result)
