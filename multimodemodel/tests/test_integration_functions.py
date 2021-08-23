"""Test the behavior of the term functions."""
from multimodemodel.grid import StaggeredGrid
from multimodemodel.coriolis import f_constant
import numpy as np
from collections import deque
import multimodemodel as swe
from typing import Tuple
import pytest


def get_x_y(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D coordinate arrays."""
    return np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx)[::-1]


def get_test_mask(x: np.ndarray) -> np.ndarray:
    """Return a test ocean mask with the shape of the input coordinate array.

    The mask is zero at the outmost array elements, one elsewhere.
    """
    mask = np.ones(x.shape, dtype=int)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    return mask


class TestRHS:
    """Test RHS of linear shallow water equations."""

    def test_linearised_SWE(self):
        """Test LSWE."""
        H, g, f0 = 1.0, 2.0, 4.0
        dx, dy = 1.0, 2.0
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)

        c_grid = StaggeredGrid.cartesian_c_grid(x[:, 0], y[0], mask_eta)

        eta = x * y * c_grid.eta.mask
        u = 2.0 * x * y * c_grid.u.mask
        v = 3.0 * x * y * c_grid.v.mask

        s = swe.State(
            u=swe.Variable(u, c_grid.u),
            v=swe.Variable(v, c_grid.v),
            eta=swe.Variable(eta, c_grid.eta),
        )

        d_u = c_grid.u.mask * (
            f_constant(f0)(c_grid.u.y)
            * (
                np.roll(np.roll(v, 1, axis=0), -1, axis=1)
                + np.roll(v, 1, axis=0)
                + np.roll(v, -1, axis=1)
                + v
            )
            / 4.0
            - g * (eta - np.roll(eta, 1, axis=0)) / dx
        )
        d_v = c_grid.v.mask * (
            -f_constant(f0)(c_grid.v.y)
            * (
                np.roll(np.roll(u, -1, axis=0), 1, axis=1)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + u
            )
            / 4.0
            - g * (eta - np.roll(eta, 1, axis=1)) / dy
        )
        d_eta = (
            -H
            * mask_eta
            * ((np.roll(u, -1, axis=0) - u) / dx + (np.roll(v, -1, axis=1) - v) / dy)
        )

        params = swe.Parameters(H=H, g=g, coriolis_func=f_constant(f0), on_grid=c_grid)

        ds = swe.linearised_SWE(s, params)
        assert np.all(ds.u.data == d_u)
        assert np.all(ds.v.data == d_v)
        assert np.all(ds.eta.data == d_eta)


class TestIntegration:
    """Test time integration schemes."""

    def test_time_stepping_function_raises_on_zero_or_less(self):
        """Test if time_stepping_function raises on invalid input."""
        with pytest.raises(
            ValueError, match="n_rhs and n_state both needs to be larger than 0."
        ):

            @swe.time_stepping_function(0, 1)
            def test():  # pragma: no cover
                pass

        with pytest.raises(
            ValueError, match="n_rhs and n_state both needs to be larger than 0."
        ):

            @swe.time_stepping_function(1, 0)
            def test2():  # pragma: no cover
                pass

    def test_euler_forward(self):
        """Test euler_forward."""
        dt = 5.0
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(x[:, 0], y[0], mask)
        params = swe.Parameters()

        ds = swe.State(
            u=swe.Variable(2 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(3 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(1 * np.ones(x.shape), c_grid.eta),
        )

        ds_test = swe.euler_forward(deque([ds], maxlen=1), params, dt)
        assert np.all(ds_test.u.data == dt * ds.u.data)
        assert np.all(ds_test.v.data == dt * ds.v.data)
        assert np.all(ds_test.eta.data == dt * ds.eta.data)

    def test_adams_bashforth2_euler_forward_dropin(self):
        """Test adams_bashforth2 computational initial condition."""
        params = swe.Parameters()
        dx, dy = 1, 2
        dt = 2.0
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        grid = swe.Grid(x, y, mask)

        state1 = swe.State(
            u=swe.Variable(np.ones(x.shape), grid),
            v=swe.Variable(np.ones(x.shape), grid),
            eta=swe.Variable(np.ones(x.shape), grid),
        )

        d_u = dt * np.ones(x.shape)

        rhs = deque([state1], maxlen=3)

        d_state = swe.adams_bashforth2(rhs, params, step=dt)

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_u)
        assert np.all(d_state.eta.data == d_u)

    def test_adams_bashforth2(self):
        """Test adams_bashforth2."""
        dt = 5.0
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        c_grid = StaggeredGrid.cartesian_c_grid(x[:, 0], y[0], mask)
        params = swe.Parameters()

        ds1 = swe.State(
            u=swe.Variable(1 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(2 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(3 * np.ones(x.shape), c_grid.eta),
        )
        ds2 = swe.State(
            u=swe.Variable(4.0 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(5 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(6 * np.ones(x.shape), c_grid.eta),
        )

        ds3 = swe.State(
            u=swe.Variable(
                dt * (3 / 2 * ds2.u.data - 1 / 2 * ds1.u.data),
                c_grid.u,
            ),
            v=swe.Variable(
                dt * (3 / 2 * ds2.v.data - 1 / 2 * ds1.v.data),
                c_grid.v,
            ),
            eta=swe.Variable(
                dt * (3 / 2 * ds2.eta.data - 1 / 2 * ds1.eta.data),
                c_grid.eta,
            ),
        )

        rhs = deque([ds1, ds2], maxlen=3)

        d_state = swe.adams_bashforth2(rhs, params, step=dt)

        assert np.all(d_state.u.data == ds3.u.data)
        assert np.all(d_state.v.data == ds3.v.data)
        assert np.all(d_state.eta.data == ds3.eta.data)

    def test_adams_bashforth3(self):
        """Test adams_bashforth3."""
        dt = 5.0
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        c_grid = StaggeredGrid.cartesian_c_grid(x[:, 0], y[0], mask)
        params = swe.Parameters()

        ds1 = swe.State(
            u=swe.Variable(1 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(2 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(3 * np.ones(x.shape), c_grid.eta),
        )
        ds2 = swe.State(
            u=swe.Variable(4.0 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(5 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(6 * np.ones(x.shape), c_grid.eta),
        )

        ds3 = swe.State(
            u=swe.Variable(7 * np.ones(x.shape), c_grid.u),
            v=swe.Variable(8 * np.ones(x.shape), c_grid.v),
            eta=swe.Variable(9 * np.ones(x.shape), c_grid.eta),
        )

        ds4 = swe.State(
            u=swe.Variable(
                dt
                * (23 / 12 * ds3.u.data - 16 / 12 * ds2.u.data + 5 / 12 * ds1.u.data),
                c_grid.u,
            ),
            v=swe.Variable(
                dt
                * (23 / 12 * ds3.v.data - 16 / 12 * ds2.v.data + 5 / 12 * ds1.v.data),
                c_grid.v,
            ),
            eta=swe.Variable(
                dt
                * (
                    23 / 12 * ds3.eta.data
                    - 16 / 12 * ds2.eta.data
                    + 5 / 12 * ds1.eta.data
                ),
                c_grid.eta,
            ),
        )

        rhs = deque([ds1, ds2, ds3], maxlen=3)

        d_state = swe.adams_bashforth3(rhs, params, step=dt)

        assert np.allclose(d_state.u.data, ds4.u.data)
        assert np.allclose(d_state.v.data, ds4.v.data)
        assert np.allclose(d_state.eta.data, ds4.eta.data)

    def test_adams_bashforth3_adams_bashforth2_dropin(self):
        """Test adams_bashforth2."""
        params = swe.Parameters()
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        grid = swe.Grid(x, y, mask)

        state1 = swe.State(
            u=swe.Variable(3 * np.ones(x.shape), grid),
            v=swe.Variable(3 * np.ones(x.shape), grid),
            eta=swe.Variable(3 * np.ones(x.shape), grid),
        )
        state2 = swe.State(
            u=swe.Variable(np.ones(x.shape), grid),
            v=swe.Variable(np.ones(x.shape), grid),
            eta=swe.Variable(np.ones(x.shape), grid),
        )

        d_u = np.zeros(x.shape)
        d_v = np.zeros(x.shape)
        d_eta = np.zeros(x.shape)

        rhs = deque([state1, state2], maxlen=3)

        d_state = swe.adams_bashforth3(rhs, params, step=2.0)

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_v)
        assert np.all(d_state.eta.data == d_eta)

    def test_integrator(self):
        """Test integrate."""
        H, g, f = 1, 1, 1
        t_end, dt = 1, 1
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones_like(x)
        c_grid = StaggeredGrid.cartesian_c_grid(x[:, 0], y[0], mask)

        eta_0 = 1 * np.ones(x.shape)
        u_0 = 1 * np.ones(x.shape)
        v_0 = 1 * np.ones(x.shape)

        eta_1 = 1 * np.ones(x.shape)
        u_1 = 2 * np.ones(x.shape)
        v_1 = 0 * np.ones(x.shape)

        params = swe.Parameters(
            H=H,
            g=g,
            coriolis_func=f_constant(f),
            on_grid=c_grid,
        )
        state_0 = swe.State(
            u=swe.Variable(u_0, c_grid.u),
            v=swe.Variable(v_0, c_grid.v),
            eta=swe.Variable(eta_0, c_grid.eta),
        )
        for state_1 in swe.integrate(
            state_0,
            params,
            scheme=swe.euler_forward,
            RHS=swe.linearised_SWE,
            step=dt,
            time=t_end,
        ):
            pass
        assert np.all(state_1.u.data == u_1)
        assert np.all(state_1.v.data == v_1)
        assert np.all(state_1.eta.data == eta_1)

    def test_integrate_raises_on_missing_scheme_attr(self):
        """Test integrate raises on unknown scheme."""
        p = swe.Parameters()
        state_0 = swe.State(u=None, v=None, eta=None)

        def rhs(state, params):  # pragma: no cover
            return state

        with pytest.raises(
            AttributeError, match="declare the function with time_stepping_function"
        ):
            for _ in swe.integrate(
                state_0,
                p,
                scheme=(lambda x: x),
                RHS=rhs,
                step=1.0,
                time=1.0,
            ):
                pass
