"""Test the behavior of the term functions."""
import pytest
import numpy as np
from collections import deque
from multimodemodel import (
    Grid,
    StaggeredGrid,
    Variable,
    State,
    StateDeque,
    Parameter,
    linearised_SWE,
    euler_forward,
    adams_bashforth2,
    adams_bashforth3,
    integrate,
    f_constant,
    time_stepping_function,
)

from multimodemodel.util import add_time


def get_x_y(nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D coordinate arrays."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    assert np.all(X[0, :] == x)
    assert np.all(Y[:, 0] == y)
    return X, Y


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
        t = np.datetime64("2000-01-01", "s")

        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)

        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            mask=mask_eta,  # type: ignore
        )

        eta = 1.0 * x * y * c_grid.eta.mask
        u = 2.0 * x * y * c_grid.u.mask
        v = 3.0 * x * y * c_grid.v.mask

        s = State(
            u=Variable(u, c_grid.u, t),
            v=Variable(v, c_grid.v, t),
            eta=Variable(eta, c_grid.eta, t),
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

        params = Parameter(H=H, g=g, coriolis_func=f_constant(f0), on_grid=c_grid)

        ds = linearised_SWE(s, params)
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

            @time_stepping_function(0, 1)  # type: ignore
            def test():  # pragma: no cover
                ...

        with pytest.raises(
            ValueError, match="n_rhs and n_state both needs to be larger than 0."
        ):

            @time_stepping_function(1, 0)  # type: ignore
            def test2():  # pragma: no cover
                pass

    def test_euler_forward(self):
        """Test euler_forward."""
        dt = 5.0
        dx, dy = 1, 2
        ni, nj = 10, 5
        t = np.datetime64("2000-01-01", "s")

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        kwargs = dict(x=x[0], y=y[:, 0], mask=mask)
        c_grid = StaggeredGrid.cartesian_c_grid(**kwargs)

        ds = State(
            u=Variable(2 * np.ones(x.shape), c_grid.u, t),
            v=Variable(3 * np.ones(x.shape), c_grid.v, t),
            eta=Variable(1 * np.ones(x.shape), c_grid.eta, t),
        )

        ds_test = euler_forward(StateDeque([ds], maxlen=1), dt)
        assert np.all(ds_test.variables["u"].data == dt * ds.variables["u"].safe_data)
        assert np.all(ds_test.variables["v"].data == dt * ds.variables["v"].safe_data)
        assert np.all(
            ds_test.variables["eta"].data == dt * ds.variables["eta"].safe_data
        )
        assert np.all(ds_test.variables["eta"].time == t)

    def test_adams_bashforth2_euler_forward_dropin(self):
        """Test adams_bashforth2 computational initial condition."""
        dx, dy = 1, 2
        dt = 2.0
        ni, nj = 10, 5
        t = np.datetime64("2000-01-01", "s")

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        grid = Grid(x=x, y=y, mask=mask)

        state1 = State(
            u=Variable(np.ones(x.shape), grid, t),
            v=Variable(np.ones(x.shape), grid, t),
            eta=Variable(np.ones(x.shape), grid, t),
        )

        d_u = dt * np.ones(x.shape)

        rhs = deque([state1], maxlen=3)

        d_state = adams_bashforth2(rhs, step=dt)  # type: ignore

        assert np.all(d_state.variables["u"].data == d_u)
        assert np.all(d_state.variables["v"].data == d_u)
        assert np.all(d_state.variables["eta"].data == d_u)
        assert (
            d_state.variables["u"].time
            == d_state.variables["eta"].time
            == d_state.variables["v"].time
            == t
        )

    def test_adams_bashforth2(self):
        """Test adams_bashforth2."""
        dt = 5
        dx, dy = 1, 2
        ni, nj = 10, 5
        t1 = np.datetime64("2000-01-01", "s")
        t2 = add_time(t1, dt)
        t3 = add_time(t2, dt / 2)

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        kwargs = dict(x=x[0], y=y[:, 0], mask=mask)
        c_grid = StaggeredGrid.cartesian_c_grid(**kwargs)

        ds1 = State(
            u=Variable(1 * np.ones(x.shape), c_grid.u, t1),
            v=Variable(2 * np.ones(x.shape), c_grid.v, t1),
            eta=Variable(3 * np.ones(x.shape), c_grid.eta, t1),
        )
        ds2 = State(
            u=Variable(4.0 * np.ones(x.shape), c_grid.u, t2),
            v=Variable(5 * np.ones(x.shape), c_grid.v, t2),
            eta=Variable(6 * np.ones(x.shape), c_grid.eta, t2),
        )

        ds3 = State(
            u=Variable(
                dt
                * (
                    3 / 2 * ds2.variables["u"].safe_data
                    - 1 / 2 * ds1.variables["u"].safe_data
                ),
                c_grid.u,
                t3,
            ),
            v=Variable(
                dt
                * (
                    3 / 2 * ds2.variables["v"].safe_data
                    - 1 / 2 * ds1.variables["v"].safe_data
                ),
                c_grid.v,
                t3,
            ),
            eta=Variable(
                dt
                * (
                    3 / 2 * ds2.variables["eta"].safe_data
                    - 1 / 2 * ds1.variables["eta"].safe_data
                ),
                c_grid.eta,
                t3,
            ),
        )

        rhs = deque([ds1, ds2], maxlen=3)

        d_state = adams_bashforth2(rhs, step=dt)  # type: ignore

        assert np.all(d_state.variables["u"].data == ds3.variables["u"].data)
        assert np.all(d_state.variables["v"].data == ds3.variables["v"].data)
        assert np.all(d_state.variables["eta"].data == ds3.variables["eta"].data)
        assert (
            d_state.variables["u"].time
            == d_state.variables["eta"].time
            == d_state.variables["v"].time
            == t3
        )

    def test_adams_bashforth3(self):
        """Test adams_bashforth3."""
        dt = 5
        dx, dy = 1, 2
        ni, nj = 10, 5
        t1 = np.datetime64("2000-01-01", "s")
        t2 = add_time(t1, dt)
        t3 = add_time(t2, dt)
        t4 = add_time(t3, dt / 2)

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        kwargs = dict(x=x[0], y=y[:, 0], mask=mask)
        c_grid = StaggeredGrid.cartesian_c_grid(**kwargs)

        ds1 = State(
            u=Variable(1 * np.ones(x.shape), c_grid.u, t1),
            v=Variable(2 * np.ones(x.shape), c_grid.v, t1),
            eta=Variable(3 * np.ones(x.shape), c_grid.eta, t1),
        )
        ds2 = State(
            u=Variable(4.0 * np.ones(x.shape), c_grid.u, t2),
            v=Variable(5 * np.ones(x.shape), c_grid.v, t2),
            eta=Variable(6 * np.ones(x.shape), c_grid.eta, t2),
        )

        ds3 = State(
            u=Variable(7 * np.ones(x.shape), c_grid.u, t3),
            v=Variable(8 * np.ones(x.shape), c_grid.v, t3),
            eta=Variable(9 * np.ones(x.shape), c_grid.eta, t3),
        )

        ds4 = State(
            u=Variable(
                dt
                * (
                    23 / 12 * ds3.variables["u"].safe_data
                    - 16 / 12 * ds2.variables["u"].safe_data
                    + 5 / 12 * ds1.variables["u"].safe_data
                ),
                c_grid.u,
                t4,
            ),
            v=Variable(
                dt
                * (
                    23 / 12 * ds3.variables["v"].safe_data
                    - 16 / 12 * ds2.variables["v"].safe_data
                    + 5 / 12 * ds1.variables["v"].safe_data
                ),
                c_grid.v,
                t4,
            ),
            eta=Variable(
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

        d_state = adams_bashforth3(rhs, step=dt)  # type: ignore

        assert np.allclose(
            d_state.variables["u"].safe_data, ds4.variables["u"].safe_data
        )
        assert np.allclose(
            d_state.variables["v"].safe_data, ds4.variables["v"].safe_data
        )
        assert np.allclose(
            d_state.variables["eta"].safe_data, ds4.variables["eta"].safe_data
        )
        assert (
            d_state.variables["u"].time
            == d_state.variables["eta"].time
            == d_state.variables["v"].time
            == t4
        )

    def test_adams_bashforth3_adams_bashforth2_dropin(self):
        """Test adams_bashforth2."""
        dt = 2.0
        dx, dy = 1, 2
        ni, nj = 10, 5
        t1 = np.datetime64("2000-01-01", "s")
        t2 = add_time(t1, dt)
        t3 = add_time(t2, dt / 2)

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        grid = Grid(x=x, y=y, mask=mask)

        state1 = State(
            u=Variable(3 * np.ones(x.shape), grid, t1),
            v=Variable(3 * np.ones(x.shape), grid, t1),
            eta=Variable(3 * np.ones(x.shape), grid, t1),
        )
        state2 = State(
            u=Variable(np.ones(x.shape), grid, t2),
            v=Variable(np.ones(x.shape), grid, t2),
            eta=Variable(np.ones(x.shape), grid, t2),
        )

        d_u = np.zeros(x.shape)
        d_v = np.zeros(x.shape)
        d_eta = np.zeros(x.shape)

        rhs = deque([state1, state2], maxlen=3)

        d_state = adams_bashforth3(rhs, step=dt)  # type: ignore

        assert np.all(d_state.variables["u"].data == d_u)
        assert np.all(d_state.variables["v"].data == d_v)
        assert np.all(d_state.variables["eta"].data == d_eta)
        assert (
            d_state.variables["u"].time
            == d_state.variables["eta"].time
            == d_state.variables["v"].time
            == t3
        )

    def test_integrator(self):
        """Test integrate."""
        H, g, f = 1, 1, 1
        t_end, dt = 1, 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        t = np.datetime64("2000-01-01", "s")

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones_like(x)
        c_grid = StaggeredGrid.cartesian_c_grid(
            x=x[0, :],
            y=y[:, 0],
            mask=mask,  # type: ignore
        )

        eta_0 = 1 * np.ones(x.shape)
        u_0 = 1 * np.ones(x.shape)
        v_0 = 1 * np.ones(x.shape)

        eta_1 = 1 * np.ones(x.shape)
        u_1 = 2 * np.ones(x.shape)
        v_1 = 0 * np.ones(x.shape)

        params = Parameter(
            H=H,
            g=g,
            coriolis_func=f_constant(f),
            on_grid=c_grid,
        )
        state_0 = State(
            u=Variable(u_0, c_grid.u, t),
            v=Variable(v_0, c_grid.v, t),
            eta=Variable(eta_0, c_grid.eta, t),
        )
        for state_1 in integrate(
            state_0,
            params,
            scheme=euler_forward,
            RHS=linearised_SWE,
            step=dt,
            time=t_end,
        ):
            pass
        assert np.all(state_1.u.data == u_1)  # type: ignore
        assert np.all(state_1.v.data == v_1)  # type: ignore
        assert np.all(state_1.eta.data == eta_1)  # type: ignore

    def test_integrate_raises_on_missing_scheme_attr(self):
        """Test integrate raises on unknown scheme."""
        p = Parameter()
        state_0 = State()

        def rhs(state, p):  # pragma: no cover
            return state

        with pytest.raises(
            AttributeError, match="declare the function with time_stepping_function"
        ):
            for _ in integrate(
                state_0,
                p,
                scheme=(lambda x: x),  # type: ignore
                RHS=rhs,
                step=1.0,
                time=1.0,
            ):
                pass
