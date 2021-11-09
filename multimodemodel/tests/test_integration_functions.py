"""Test the behavior of the term functions."""
from multimodemodel.grid import StaggeredGrid
from multimodemodel.coriolis import f_constant
from multimodemodel.integrate import seconds_to_timedelta64
import numpy as np
from collections import deque
import multimodemodel as swe
from typing import Tuple
import pytest


def get_x_y(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
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


def get_test_mask(shape) -> np.ndarray:
    """Return a test ocean mask with the shape of the input coordinate array.

    The mask is zero at the outmost array elements, one elsewhere.
    """
    mask = np.ones(shape, dtype=int)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    return mask


class TestRHS:
    """Test RHS of linear shallow water equations."""

    def test_linearised_SWE(self):
        """Test LSWE."""
        H, g, f0 = np.array([1.0]), 2.0, 4.0
        dx, dy = 1.0, 2.0
        ni, nj, nz = 10, 5, 1
        t = np.datetime64("2000-01-01", "s")

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask_eta = get_test_mask(z.shape + x.shape)

        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :], y=y[:, 0], mask=mask_eta  # type: ignore
        )

        xy = np.expand_dims(x * y, axis=0)
        eta = xy * c_grid.eta.mask
        u = 2.0 * xy * c_grid.u.mask
        v = 3.0 * xy * c_grid.v.mask

        s = swe.State(
            u=swe.Variable(u, c_grid.u, t),
            v=swe.Variable(v, c_grid.v, t),
            eta=swe.Variable(eta, c_grid.eta, t),
        )

        d_u = c_grid.u.mask * (
            f_constant(f0)(c_grid.u.y)
            * (
                np.roll(np.roll(v, 1, axis=-1), -1, axis=-2)
                + np.roll(v, 1, axis=-1)
                + np.roll(v, -1, axis=-2)
                + v
            )
            / 4.0
            - g * (eta - np.roll(eta, 1, axis=-1)) / dx
        )
        d_v = c_grid.v.mask * (
            -f_constant(f0)(c_grid.v.y)
            * (
                np.roll(np.roll(u, -1, axis=-1), 1, axis=-2)
                + np.roll(u, -1, axis=-1)
                + np.roll(u, 1, axis=-2)
                + u
            )
            / 4.0
            - g * (eta - np.roll(eta, 1, axis=-2)) / dy
        )
        d_eta = (
            -H
            * mask_eta
            * ((np.roll(u, -1, axis=-1) - u) / dx + (np.roll(v, -1, axis=-2) - v) / dy)
        )

        params = swe.Parameters(H=H, g=g, coriolis_func=f_constant(f0), on_grid=c_grid)

        ds = swe.linearised_SWE(s, params)
        assert np.all(ds.variables["u"].data == d_u)
        assert np.all(ds.variables["v"].data == d_v)
        assert np.all(ds.variables["eta"].data == d_eta)


class TestIntegration:
    """Test time integration schemes."""

    def test_time_stepping_function_raises_on_zero_or_less(self):
        """Test if time_stepping_function raises on invalid input."""
        with pytest.raises(
            ValueError, match="n_rhs and n_state both needs to be larger than 0."
        ):

            @swe.time_stepping_function(0, 1)  # type: ignore
            def test():  # pragma: no cover
                pass

        with pytest.raises(
            ValueError, match="n_rhs and n_state both needs to be larger than 0."
        ):

            @swe.time_stepping_function(1, 0)  # type: ignore
            def test2():  # pragma: no cover
                pass

    def test_euler_forward(self):
        """Test euler_forward."""
        dt = 5.0
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        t = np.datetime64("2000-01-01", "s")

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = np.ones(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            mask=mask,  # type: ignore
        )

        ds = swe.State(
            u=swe.Variable(2 * np.ones(mask.shape), c_grid.u, t),
            v=swe.Variable(3 * np.ones(mask.shape), c_grid.v, t),
            eta=swe.Variable(1 * np.ones(mask.shape), c_grid.eta, t),
        )

        ds_test = swe.euler_forward(deque([ds], maxlen=1), dt)
        assert np.all(ds_test.variables["u"].data == dt * ds.variables["u"].data)
        assert np.all(ds_test.variables["v"].data == dt * ds.variables["v"].data)
        assert np.all(ds_test.variables["eta"].data == dt * ds.variables["eta"].data)
        assert np.all(ds_test.variables["eta"].time == t)

    def test_adams_bashforth2_euler_forward_dropin(self):
        """Test adams_bashforth2 computational initial condition."""
        dx, dy = 1, 2
        dt = 2.0
        ni, nj, nz = 10, 5, 1
        t = np.datetime64("2000-01-01", "s")

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        grid = swe.Grid(x=x, y=y, z=z, mask=mask)

        state1 = swe.State(
            u=swe.Variable(np.ones(mask.shape), grid, t),
            v=swe.Variable(np.ones(mask.shape), grid, t),
            eta=swe.Variable(np.ones(mask.shape), grid, t),
        )

        d_u = dt * np.ones(mask.shape)

        rhs = deque([state1], maxlen=3)

        d_state = swe.adams_bashforth2(rhs, step=dt)  # type: ignore

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_u)
        assert np.all(d_state.eta.data == d_u)
        assert d_state.u.time == d_state.eta.time == d_state.v.time == t

    def test_adams_bashforth2(self):
        """Test adams_bashforth2."""
        dt = 5
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        t1 = np.datetime64("2000-01-01", "s")
        t2 = t1 + seconds_to_timedelta64(dt)
        t3 = t2 + seconds_to_timedelta64(dt / 2)

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            z=z,
            mask=mask,  # type: ignore
        )

        ds1 = swe.State(
            u=swe.Variable(1 * np.ones(mask.shape), c_grid.u, t1),
            v=swe.Variable(2 * np.ones(mask.shape), c_grid.v, t1),
            eta=swe.Variable(3 * np.ones(mask.shape), c_grid.eta, t1),
        )
        ds2 = swe.State(
            u=swe.Variable(4.0 * np.ones(mask.shape), c_grid.u, t2),
            v=swe.Variable(5 * np.ones(mask.shape), c_grid.v, t2),
            eta=swe.Variable(6 * np.ones(mask.shape), c_grid.eta, t2),
        )

        ds3 = swe.State(
            u=swe.Variable(
                dt
                * (3 / 2 * ds2.variables["u"].data - 1 / 2 * ds1.variables["u"].data),
                c_grid.u,
                t3,
            ),
            v=swe.Variable(
                dt
                * (3 / 2 * ds2.variables["v"].data - 1 / 2 * ds1.variables["v"].data),
                c_grid.v,
                t3,
            ),
            eta=swe.Variable(
                dt
                * (
                    3 / 2 * ds2.variables["eta"].data
                    - 1 / 2 * ds1.variables["eta"].data
                ),
                c_grid.eta,
                t3,
            ),
        )

        rhs = deque([ds1, ds2], maxlen=3)

        d_state = swe.adams_bashforth2(rhs, step=dt)  # type: ignore

        assert np.all(d_state.variables["u"].data == ds3.variables["u"].data)
        assert np.all(d_state.variables["v"].data == ds3.variables["v"].data)
        assert np.all(d_state.variables["eta"].data == ds3.variables["eta"].data)
        assert d_state.u.time == d_state.eta.time == d_state.v.time == t3

    def test_adams_bashforth3(self):
        """Test adams_bashforth3."""
        dt = 5
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        t1 = np.datetime64("2000-01-01", "s")
        t2 = t1 + seconds_to_timedelta64(dt)
        t3 = t2 + seconds_to_timedelta64(dt)
        t4 = t3 + seconds_to_timedelta64(dt / 2)

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = get_test_mask(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            z=z,
            mask=mask,  # type: ignore
        )

        ds1 = swe.State(
            u=swe.Variable(1 * np.ones(mask.shape), c_grid.u, t1),
            v=swe.Variable(2 * np.ones(mask.shape), c_grid.v, t1),
            eta=swe.Variable(3 * np.ones(mask.shape), c_grid.eta, t1),
        )
        ds2 = swe.State(
            u=swe.Variable(4.0 * np.ones(mask.shape), c_grid.u, t2),
            v=swe.Variable(5 * np.ones(mask.shape), c_grid.v, t2),
            eta=swe.Variable(6 * np.ones(mask.shape), c_grid.eta, t2),
        )

        ds3 = swe.State(
            u=swe.Variable(7 * np.ones(mask.shape), c_grid.u, t3),
            v=swe.Variable(8 * np.ones(mask.shape), c_grid.v, t3),
            eta=swe.Variable(9 * np.ones(mask.shape), c_grid.eta, t3),
        )

        ds4 = swe.State(
            u=swe.Variable(
                dt
                * (
                    23 / 12 * ds3.variables["u"].safe_data
                    - 16 / 12 * ds2.variables["u"].safe_data
                    + 5 / 12 * ds1.variables["u"].safe_data
                ),
                c_grid.u,
                t4,
            ),
            v=swe.Variable(
                dt
                * (
                    23 / 12 * ds3.variables["v"].safe_data
                    - 16 / 12 * ds2.variables["v"].safe_data
                    + 5 / 12 * ds1.variables["v"].safe_data
                ),
                c_grid.v,
                t4,
            ),
            eta=swe.Variable(
                dt
                * (
                    23 / 12 * ds3.variables["eta"].safe_data
                    - 16 / 12 * ds2.variables["eta"].safe_data
                    + 5 / 12 * ds1.variables["eta"].safe_data
                ),
                c_grid.eta,
                t4,
            ),
        )

        rhs = deque([ds1, ds2, ds3], maxlen=3)

        d_state = swe.adams_bashforth3(rhs, step=dt)  # type: ignore

        assert np.allclose(d_state.variables["u"].data, ds4.variables["u"].data)
        assert np.allclose(d_state.variables["v"].data, ds4.variables["v"].data)
        assert np.allclose(d_state.variables["eta"].data, ds4.variables["eta"].data)
        assert d_state.u.time == d_state.eta.time == d_state.v.time == t4

    def test_adams_bashforth3_adams_bashforth2_dropin(self):
        """Test adams_bashforth2."""
        dt = 2.0
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        t1 = np.datetime64("2000-01-01", "s")
        t2 = t1 + seconds_to_timedelta64(dt)
        t3 = t2 + seconds_to_timedelta64(dt / 2)

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = np.ones(z.shape + x.shape)
        grid = swe.Grid(x=x, y=y, z=z, mask=mask)

        state1 = swe.State(
            u=swe.Variable(3 * np.ones(mask.shape), grid, t1),
            v=swe.Variable(3 * np.ones(mask.shape), grid, t1),
            eta=swe.Variable(3 * np.ones(mask.shape), grid, t1),
        )
        state2 = swe.State(
            u=swe.Variable(np.ones(mask.shape), grid, t2),
            v=swe.Variable(np.ones(mask.shape), grid, t2),
            eta=swe.Variable(np.ones(mask.shape), grid, t2),
        )

        d_u = np.zeros(mask.shape)
        d_v = np.zeros(mask.shape)
        d_eta = np.zeros(mask.shape)

        rhs = deque([state1, state2], maxlen=3)

        d_state = swe.adams_bashforth3(rhs, step=dt)  # type: ignore

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_v)
        assert np.all(d_state.eta.data == d_eta)
        assert d_state.u.time == d_state.eta.time == d_state.v.time == t3

    def test_integrator(self):
        """Test integrate."""
        H, g, f = np.array([1]), 1, 1
        t_end, dt = 1, 1
        dx, dy = 1, 2
        ni, nj, nz = 10, 5, 1
        t = np.datetime64("2000-01-01", "s")

        x, y, z = get_x_y_z(ni, nj, nz, dx, dy)
        mask = np.ones(z.shape + x.shape)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            z=z,
            mask=mask,  # type: ignore
        )

        eta_0 = 1 * np.ones(mask.shape)
        u_0 = 1 * np.ones(mask.shape)
        v_0 = 1 * np.ones(mask.shape)

        eta_1 = 1 * np.ones(mask.shape)
        u_1 = 2 * np.ones(mask.shape)
        v_1 = 0 * np.ones(mask.shape)

        params = swe.Parameters(
            H=H,
            g=g,
            coriolis_func=f_constant(f),
            on_grid=c_grid,
        )
        state_0 = swe.State(
            u=swe.Variable(u_0, c_grid.u, t),
            v=swe.Variable(v_0, c_grid.v, t),
            eta=swe.Variable(eta_0, c_grid.eta, t),
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
        assert np.all(state_1.u.data == u_1)  # type: ignore
        assert np.all(state_1.v.data == v_1)  # type: ignore
        assert np.all(state_1.eta.data == eta_1)  # type: ignore

    def test_integrate_raises_on_missing_scheme_attr(self):
        """Test integrate raises on unknown scheme."""
        p = swe.Parameters()
        state_0 = swe.State()

        def rhs(state, params):  # pragma: no cover
            return state

        with pytest.raises(
            AttributeError, match="declare the function with time_stepping_function"
        ):
            for _ in swe.integrate(
                state_0,
                p,
                scheme=(lambda x: x),  # type: ignore
                RHS=rhs,
                step=1.0,
                time=1.0,
            ):
                pass
