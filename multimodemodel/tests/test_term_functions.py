"""Test the behavior of the term functions."""
import pytest
import numpy as np
from multimodemodel import (
    beta_plane,
    f_constant,
    f_on_sphere,
    Parameter,
    State,
    Variable,
    Grid,
    StaggeredGrid,
    coriolis_i,
    coriolis_j,
    divergence_i,
    divergence_j,
    pressure_gradient_i,
    pressure_gradient_j,
)
from multimodemodel.datastructure import MultimodeParameter
from multimodemodel.kernel import (
    _pressure_gradient_i_dispatch_table,
    _get_from_dispatch_table,
    advection_density,
    advection_momentum_u,
    advection_momentum_v,
    laplacian_mixing_u,
    laplacian_mixing_v,
    linear_damping_eta,
    linear_damping_u,
    linear_damping_v,
)

some_datetime = np.datetime64("2001-01-01", "s")


def get_x_y(nx=10.0, ny=10.0, dx=1.0, dy=2.0) -> tuple[np.ndarray, np.ndarray]:
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
    mask = np.ones(shape)
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
        ni, nj, nz = 10, 5, 1
        x, _, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        mask_u = mask * np.roll(mask, 1, axis=-1)
        eta = np.expand_dims(x, axis=0)
        dx_a = dx * np.ones(x.shape)

        oracle = -g * (eta * mask - np.roll(eta * mask, 1, axis=-1)) / dx_a * mask_u
        pg_i = _get_from_dispatch_table(eta, _pressure_gradient_i_dispatch_table)
        result = pg_i(ni, nj, nz, eta, mask_u, dx_a, g)  # type: ignore
        print(oracle, result, eta.shape)

        assert np.all(result == oracle)

    def test_2D_on_3D(self):
        """Test mapping of 2D iterator on 3D data."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        z = np.arange(10.0)
        c_grid_2D = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0])
        c_grid_3D = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0], z=z)

        params_2D = Parameter(coriolis_func=f_constant(1e-4), on_grid=c_grid_2D)
        params_3D = Parameter(coriolis_func=f_constant(1e-4), on_grid=c_grid_3D)

        state_2D = State(
            u=Variable(np.ones(c_grid_2D.u.shape), c_grid_2D.u, some_datetime),
            v=Variable(None, c_grid_2D.v, some_datetime),
        )
        inc_2D = coriolis_j(state_2D, params_2D)

        state_3D = State(
            u=Variable(np.ones(c_grid_3D.u.shape), c_grid_3D.u, some_datetime),
            v=Variable(None, c_grid_3D.v, some_datetime),
        )
        inc_3D = coriolis_j(state_3D, params_3D)

        # exploit broadcasting before comparison
        assert np.all(inc_2D.variables["v"].data == inc_3D.variables["v"].data)

    def test_pressure_gradient_i(self):
        """Test pressure_gradient_i."""
        g = 1
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        eta = np.expand_dims(x, axis=0) * mask
        result = -g * mask * (eta - np.roll(eta, 1, axis=-1)) / dx

        params = Parameter(g=g)
        grid = Grid(x=x, y=y, z=z, mask=mask)
        state = State(
            u=Variable(None, grid, some_datetime),
            eta=Variable(eta, grid, some_datetime),
        )

        inc = pressure_gradient_i(state, params)

        assert "eta" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["u"].data == result)
        assert inc.variables["u"].time == some_datetime

    def test_pressure_gradient_j(self):
        """Test pressure_gradient_j."""
        g = 1
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        eta = np.expand_dims(x, axis=0) * mask
        result = -g * mask * (eta - np.roll(eta, 1, axis=-2)) / dy

        params = Parameter(g=g)
        grid = Grid(x=x, y=y, z=z, mask=mask)
        state = State(
            v=Variable(None, grid, some_datetime),
            eta=Variable(eta, grid, some_datetime),
        )

        inc = pressure_gradient_j(state, params)

        assert "eta" not in inc.variables
        assert "u" not in inc.variables
        assert np.all(inc.variables["v"].data == result)
        assert inc.variables["v"].time == some_datetime

    def test_divergence_i(self):
        """Test divergence_i."""
        H = np.array([1])
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask_eta = get_test_mask(z.shape + x.shape)
        mask_u = mask_eta * np.roll(mask_eta, 1, axis=-1)
        u = np.expand_dims(x, axis=0) * mask_u
        result = -H * (np.roll(u, -1, axis=-1) - u) / dx

        params = Parameter(H=H)
        grid_u = Grid(x=x, y=y, z=z, mask=mask_u)
        grid_eta = Grid(x=x, y=y, z=z, mask=mask_eta)
        state = State(
            u=Variable(u, grid_u, some_datetime),
            eta=Variable(None, grid_eta, some_datetime),
        )

        inc = divergence_i(state, params)

        assert "u" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["eta"].data == result)
        assert inc.variables["eta"].time == some_datetime

    def test_divergence_j(self):
        """Test divergence_j."""
        H = np.array([1])
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask_eta = get_test_mask(z.shape + x.shape)
        mask_v = mask_eta * np.roll(mask_eta, 1, axis=-2)
        v = np.expand_dims(y, axis=0) * mask_v
        result = -H * (np.roll(v, -1, axis=-2) - v) / dy

        params = Parameter(H=H)
        grid_v = Grid(x=x, y=y, z=z, mask=mask_v)
        grid_eta = Grid(x=x, y=y, z=z, mask=mask_eta)
        state = State(
            v=Variable(v, grid_v, some_datetime),
            eta=Variable(None, grid_eta, some_datetime),
        )

        inc = divergence_j(state, params)
        assert "u" not in inc.variables
        assert "v" not in inc.variables
        assert np.all(inc.variables["eta"].data == result)
        assert inc.variables["eta"].time == some_datetime

    @pytest.mark.parametrize(
        "coriolis_func",
        [
            f_constant(0.0),
            f_constant(1.0),
            beta_plane(0.0, 1.0, 5),
            f_on_sphere(1.0),
        ],
    )
    def test_coriolis_j(self, coriolis_func):
        """Test coriolis_j."""
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            z=z,
            mask=mask,  # type: ignore
        )

        params = Parameter(coriolis_func=coriolis_func, on_grid=c_grid)

        u = np.ones(mask.shape) * c_grid.u.mask  # type: ignore

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

        state = State(
            u=Variable(u, c_grid.u, some_datetime),
            v=Variable(None, c_grid.v, some_datetime),
        )

        inc = coriolis_j(state, params)
        assert "u" not in inc.variables
        assert "eta" not in inc.variables
        assert np.all(inc.variables["v"].data == result)
        assert inc.variables["v"].time == some_datetime

    @pytest.mark.parametrize(
        "coriolis_func",
        [
            f_constant(0.0),
            f_constant(1.0),
            beta_plane(0.0, 1.0, 5),
            f_on_sphere(1.0),
        ],
    )
    def test_coriolis_i(self, coriolis_func):
        """Test coriolis_i."""
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            z=z,
            mask=mask,  # type: ignore
        )

        params = Parameter(coriolis_func=coriolis_func, on_grid=c_grid)

        v = np.ones(mask.shape) * c_grid.v.mask  # type: ignore

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

        state = State(
            u=Variable(None, c_grid.u, some_datetime),
            v=Variable(v, c_grid.v, some_datetime),
            eta=Variable(None, c_grid.eta, some_datetime),
        )

        inc = coriolis_i(state, params)
        assert "v" not in inc.variables
        assert "eta" not in inc.variables
        assert np.all(inc.variables["u"].data == result)
        assert inc.variables["u"].time == some_datetime

    @pytest.mark.parametrize(
        "term",
        (
            pressure_gradient_i,
            pressure_gradient_j,
            coriolis_i,
            coriolis_j,
            divergence_i,
            divergence_j,
            laplacian_mixing_u,
            laplacian_mixing_v,
            linear_damping_u,
            linear_damping_v,
            linear_damping_eta,
            advection_momentum_u,
            advection_momentum_v,
            advection_density,
        ),
    )
    def test_none_input(self, term):
        """Test for None input compatibility."""
        dx, dy = 1, 2
        ni, nj, nk = 10, 5, 1
        x, y, z = get_x_y_z(ni, nj, nk, dx, dy)
        kwargs = dict(x=x[0, :], y=y[:, 0], z=z)
        c_grid = StaggeredGrid.cartesian_c_grid(**kwargs)
        triple_tensors = np.zeros((nk, nk, nk))
        params = MultimodeParameter(
            coriolis_func=f_constant(1.0),
            on_grid=c_grid,
            z=z,
            psi=z,
            dpsi_dz=z,
            c=z,
            Q=triple_tensors,
            R=triple_tensors,
            S=triple_tensors,
            T=triple_tensors,
        )

        state = State(
            u=Variable(None, c_grid.u, some_datetime),
            v=Variable(None, c_grid.v, some_datetime),
            eta=Variable(None, c_grid.eta, some_datetime),
            q=Variable(None, c_grid.q, some_datetime),
        )
        state.set_diagnostic_variable(w=Variable(None, c_grid.eta, some_datetime))

        _ = term(state, params)

    @pytest.mark.parametrize(
        "term",
        (
            pressure_gradient_i,
            pressure_gradient_j,
            coriolis_i,
            coriolis_j,
            divergence_i,
            divergence_j,
            laplacian_mixing_u,
            laplacian_mixing_v,
            linear_damping_u,
            linear_damping_v,
            linear_damping_eta,
        ),
    )
    def test_on_2D_grid(self, term):
        """Test for None input compatibility."""
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        c_grid = StaggeredGrid.cartesian_c_grid(x=x[0, :], y=y[:, 0])
        params = Parameter(coriolis_func=f_constant(1.0), on_grid=c_grid)

        state = State(
            u=Variable(None, c_grid.u, some_datetime),
            v=Variable(None, c_grid.v, some_datetime),
            eta=Variable(None, c_grid.eta, some_datetime),
            q=Variable(None, c_grid.q, some_datetime),
        )
        state.set_diagnostic_variable(w=Variable(None, c_grid.eta, some_datetime))

        _ = term(state, params)
