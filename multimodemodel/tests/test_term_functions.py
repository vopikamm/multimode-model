"""Test the behavior of the term functions."""
from multimodemodel.coriolis import f_constant
from multimodemodel.grid import StaggeredGrid
import numpy as np
import pytest
import multimodemodel as swe

some_datetime = np.datetime64("2001-01-01", "s")


def get_x_y(nx=10.0, ny=10.0, dx=1.0, dy=2.0):
    """Return 2D coordinate arrays."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    assert np.all(X[0, :] == x)
    assert np.all(Y[:, 0] == y)
    return X, Y


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
        mask_u = mask * np.roll(mask, 1, axis=-1)
        eta = np.copy(x)
        dx_a = dx * np.ones(x.shape)

        oracle = -g * (eta * mask - np.roll(eta * mask, 1, axis=-1)) / dx_a * mask_u
        result = swe._pressure_gradient_i(ni, nj, eta, g, dx_a, mask_u)  # type: ignore
        print(oracle, result, x.shape)

        assert np.all(result == oracle)

    def test_2D_on_3D(self):
        """Test mapping of 2D iterator on 3D data."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y, z = *get_x_y(ni, nj, dx, dy), np.arange(10.0)
        c_grid_2D = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0])
        c_grid_3D = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0], z=z)

        params_2D = swe.Parameters(coriolis_func=f_constant(1e-4), on_grid=c_grid_2D)
        params_3D = swe.Parameters(coriolis_func=f_constant(1e-4), on_grid=c_grid_3D)

        state_2D = swe.State(
            u=swe.Variable(np.ones(c_grid_2D.u.shape), c_grid_2D.u, some_datetime),
            v=swe.Variable(None, c_grid_2D.v, some_datetime),
        )
        inc_2D = swe.coriolis_j(state_2D, params_2D)

        state_3D = swe.State(
            u=swe.Variable(np.ones(c_grid_3D.u.shape), c_grid_3D.u, some_datetime),
            v=swe.Variable(None, c_grid_3D.v, some_datetime),
        )
        inc_3D = swe.coriolis_j(state_3D, params_3D)

        # exploit broadcasting before comparison
        assert np.all(inc_2D.variables["v"].data == inc_3D.variables["v"].data)

    def test_pressure_gradient_i(self):
        """Test pressure_gradient_i."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        eta = np.copy(x) * mask
        result = -g * mask * (eta - np.roll(eta, 1, axis=-1)) / dx

        params = swe.Parameters(g=g)
        grid = swe.Grid(x=x, y=y, mask=mask)
        state = swe.State(
            u=swe.Variable(None, grid, some_datetime),
            eta=swe.Variable(eta, grid, some_datetime),
        )

        inc = swe.pressure_gradient_i(state, params)

        assert "eta" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["u"].data == result)
        assert inc.variables["u"].time == some_datetime

    def test_pressure_gradient_j(self):
        """Test pressure_gradient_j."""
        g = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        eta = np.copy(y)
        result = -g * mask * (eta - np.roll(eta, 1, axis=-2)) / dy

        params = swe.Parameters(g=g)
        grid = swe.Grid(x=x, y=y, mask=mask)
        state = swe.State(
            v=swe.Variable(None, grid, some_datetime),
            eta=swe.Variable(eta, grid, some_datetime),
        )

        inc = swe.pressure_gradient_j(state, params)

        assert "eta" not in inc.variables
        assert "u" not in inc.variables
        assert np.all(inc.variables["v"].data == result)
        assert inc.variables["v"].time == some_datetime

    def test_divergence_i(self):
        """Test divergence_i."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)
        mask_u = mask_eta * np.roll(mask_eta, 1, axis=-1)
        u = np.copy(x) * mask_u
        result = -H * (np.roll(u, -1, axis=-1) - u) / dx

        params = swe.Parameters(H=H)
        grid_u = swe.Grid(x=x, y=y, mask=mask_u)
        grid_eta = swe.Grid(x=x, y=y, mask=mask_eta)
        state = swe.State(
            u=swe.Variable(u, grid_u, some_datetime),
            eta=swe.Variable(None, grid_eta, some_datetime),
        )

        inc = swe.divergence_i(state, params)

        assert "u" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["eta"].data == result)
        assert inc.variables["eta"].time == some_datetime

    def test_divergence_j(self):
        """Test divergence_j."""
        H = 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)
        mask_v = mask_eta * np.roll(mask_eta, 1, axis=-2)
        v = np.copy(y) * mask_v
        result = -H * (np.roll(v, -1, axis=-2) - v) / dy

        params = swe.Parameters(H=H)
        grid_v = swe.Grid(x=x, y=y, mask=mask_v)
        grid_eta = swe.Grid(x=x, y=y, mask=mask_eta)
        state = swe.State(
            v=swe.Variable(v, grid_v, some_datetime),
            eta=swe.Variable(None, grid_eta, some_datetime),
        )

        inc = swe.divergence_j(state, params)
        assert "u" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["eta"].data == result)
        assert inc.variables["eta"].time == some_datetime

    @pytest.mark.parametrize(
        "coriolis_func",
        [
            swe.f_constant(0.0),
            swe.f_constant(1.0),
            swe.beta_plane(0.0, 1.0, 5),
            swe.f_on_sphere(1.0),
        ],
    )
    def test_coriolis_j(self, coriolis_func):
        """Test coriolis_j."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            mask=mask,  # type: ignore
        )

        params = swe.Parameters(coriolis_func=coriolis_func, on_grid=c_grid)

        u = np.ones(x.shape) * c_grid.u.mask  # type: ignore

        result = (
            -c_grid.v.mask  # type: ignore
            * coriolis_func(c_grid.v.y)
            * (
                np.roll(np.roll(u, -1, axis=-1), 1, axis=-2)
                + np.roll(u, -1, axis=-1)
                + np.roll(u, 1, axis=-2)
                + u
            )
            / 4.0
        )

        state = swe.State(
            u=swe.Variable(u, c_grid.u, some_datetime),
            v=swe.Variable(None, c_grid.v, some_datetime),
        )

        inc = swe.coriolis_j(state, params)
        assert "u" not in inc.variables
        assert "eta" not in inc.variables
        assert np.all(inc.variables["v"].data == result)
        assert inc.variables["v"].time == some_datetime

    @pytest.mark.parametrize(
        "coriolis_func",
        [
            swe.f_constant(0.0),
            swe.f_constant(1.0),
            swe.beta_plane(0.0, 1.0, 5),
            swe.f_on_sphere(1.0),
        ],
    )
    def test_coriolis_i(self, coriolis_func):
        """Test coriolis_i."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            mask=mask,  # type: ignore
        )

        params = swe.Parameters(coriolis_func=coriolis_func, on_grid=c_grid)

        v = np.ones(y.shape) * c_grid.v.mask  # type: ignore

        result = (
            c_grid.u.mask
            * coriolis_func(c_grid.u.y)
            * (
                np.roll(np.roll(v, 1, axis=-1), -1, axis=-2)
                + np.roll(v, 1, axis=-1)
                + np.roll(v, -1, axis=-2)
                + v
            )
            / 4.0
        )

        state = swe.State(
            u=swe.Variable(None, c_grid.u, some_datetime),
            v=swe.Variable(v, c_grid.v, some_datetime),
            eta=swe.Variable(None, c_grid.eta, some_datetime),
        )

        inc = swe.coriolis_i(state, params)
        assert "v" not in inc.variables
        assert "eta" not in inc.variables
        assert np.all(inc.variables["u"].data == result)
        assert inc.variables["u"].time == some_datetime

    @pytest.mark.parametrize(
        "term",
        (
            swe.pressure_gradient_i,
            swe.pressure_gradient_j,
            swe.coriolis_i,
            swe.coriolis_j,
            swe.divergence_i,
            swe.divergence_j,
        ),
    )
    def test_none_input(self, term):
        """Test for None input compatibility."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        c_grid = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0])
        params = swe.Parameters(coriolis_func=f_constant(1.0), on_grid=c_grid)

        state = swe.State(
            u=swe.Variable(None, c_grid.u, some_datetime),
            v=swe.Variable(None, c_grid.v, some_datetime),
            eta=swe.Variable(None, c_grid.eta, some_datetime),
        )

        _ = term(state, params)
