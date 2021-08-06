"""Test the behaviour of the term functions."""
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
        H, g, f = 1.0, 2.0, 4.0
        dx, dy = 1.0, 2.0
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask_eta = get_test_mask(x)
        mask_v = mask_eta * np.roll(mask_eta, 1, axis=1)
        mask_u = mask_eta * np.roll(mask_eta, 1, axis=0)
        eta = np.zeros(x.shape)
        u = np.zeros(x.shape)
        v = mask_v.copy()

        d_u = (
            mask_u
            * f
            * (
                np.roll(np.roll(v, 1, axis=0), -1, axis=1)
                + np.roll(v, 1, axis=0)
                + np.roll(v, -1, axis=1)
                + v
            )
            / 4.0
        )
        d_eta = -H * mask_eta * (np.roll(v, -1, axis=1) + (-1) * v) / dy
        d_v = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f)
        state = swe.State(
            u=swe.Variable(u, swe.Grid(x, y, mask_u)),
            v=swe.Variable(v, swe.Grid(x, y, mask_v)),
            eta=swe.Variable(eta, swe.Grid(x, y, mask_eta)),
        )

        assert np.all(swe.linearised_SWE(state, params).u.data == d_u)
        assert np.all(swe.linearised_SWE(state, params).v.data == d_v)
        assert np.all(swe.linearised_SWE(state, params).eta.data == d_eta)


class TestIntegration:
    """Test time integration schemes."""

    def test_euler_forward(self):
        """Test euler_forward."""
        H, g, f = 1, 2, 3
        dt = 1
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        eta = 1 * np.ones(x.shape)
        u = 2 * np.ones(x.shape)
        v = 3 * np.ones(x.shape)

        d_u = 9 * np.ones(v.shape)
        d_v = -6 * np.ones(u.shape)
        d_eta = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f, dt=dt)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid),
        )
        rhs = deque([swe.linearised_SWE(state, params)], maxlen=1)

        assert np.all(swe.euler_forward(rhs, params).u.data == d_u)
        assert np.all(swe.euler_forward(rhs, params).v.data == d_v)
        assert np.all(swe.euler_forward(rhs, params).eta.data == d_eta)

    def test_adams_bashforth2_euler_forward_dropin(self):
        """Test adams_bashforth2 mcomputational initial condition."""
        params = swe.Parameters(dt=2.0)
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        grid = swe.Grid(x, y, mask)

        state1 = swe.State(
            u=swe.Variable(np.ones(x.shape), grid),
            v=swe.Variable(np.ones(x.shape), grid),
            eta=swe.Variable(np.ones(x.shape), grid),
        )

        d_u = params.dt * np.ones(x.shape)

        rhs = deque([state1], maxlen=3)

        d_state = swe.adams_bashforth2(rhs, params)

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_u)
        assert np.all(d_state.eta.data == d_u)

    def test_adams_bashforth2(self):
        """Test adams_bashforth2."""
        params = swe.Parameters(dt=2.0)
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

        d_state = swe.adams_bashforth2(rhs, params)

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_v)
        assert np.all(d_state.eta.data == d_eta)

    def test_adams_bashforth3(self):
        """Test adams_bashforth3."""
        H, g, f = 1, 1, 1
        dt = 1
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        eta = 1 * np.ones(x.shape)
        u = 1 * np.ones(x.shape)
        v = 1 * np.ones(x.shape)

        d_u = 1 * np.ones(v.shape)
        d_v = -1 * np.ones(u.shape)
        d_eta = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f, dt=dt)
        grid = swe.Grid(x, y, mask)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid),
        )

        rhs = deque(
            [
                swe.linearised_SWE(state, params),
                swe.linearised_SWE(state, params),
                swe.linearised_SWE(state, params),
            ],
            maxlen=3,
        )

        assert np.all(swe.adams_bashforth3(rhs, params).u.data == d_u)
        assert np.all(swe.adams_bashforth3(rhs, params).v.data == d_v)
        assert np.all(swe.adams_bashforth3(rhs, params).eta.data == d_eta)

    def test_adams_bashforth3_adams_bashforth2_dropin(self):
        """Test adams_bashforth2."""
        params = swe.Parameters(dt=2.0)
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

        d_state = swe.adams_bashforth3(rhs, params)

        assert np.all(d_state.u.data == d_u)
        assert np.all(d_state.v.data == d_v)
        assert np.all(d_state.eta.data == d_eta)

    def test_integrator(self):
        """Test integrator."""
        H, g, f = 1, 1, 1
        t_0, t_end, dt = 0, 1, 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)

        eta_0 = 1 * np.ones(x.shape)
        u_0 = 1 * np.ones(x.shape)
        v_0 = 1 * np.ones(x.shape)

        eta_1 = 1 * np.ones(x.shape)
        u_1 = 2 * np.ones(x.shape)
        v_1 = 0 * np.ones(x.shape)

        params = swe.Parameters(H=H, g=g, f=f, t_0=t_0, t_end=t_end, dt=dt)
        grid = swe.Grid(x, y, mask)
        state_0 = swe.State(
            u=swe.Variable(u_0, grid),
            v=swe.Variable(v_0, grid),
            eta=swe.Variable(eta_0, grid),
        )
        state_1 = swe.integrator(
            state_0, params, scheme=swe.euler_forward, RHS=swe.linearised_SWE
        )
        assert np.all(state_1.u.data == u_1)
        assert np.all(state_1.v.data == v_1)
        assert np.all(state_1.eta.data == eta_1)

    def test_integrator_raises_on_unknown_scheme(self):
        """Test integrator raises on unknown scheme."""
        H, g, f = 1, 1, 1
        t_0, t_end, dt = 0, 1, 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)

        eta_0 = 1 * np.ones(x.shape)
        u_0 = 1 * np.ones(x.shape)
        v_0 = 1 * np.ones(x.shape)

        params = swe.Parameters(H=H, g=g, f=f, t_0=t_0, t_end=t_end, dt=dt)
        grid = swe.Grid(x, y, mask)
        state_0 = swe.State(
            u=swe.Variable(u_0, grid),
            v=swe.Variable(v_0, grid),
            eta=swe.Variable(eta_0, grid),
        )

        with pytest.raises(ValueError, match="Unsupported scheme"):
            _ = swe.integrator(
                state_0, params, scheme=(lambda x: state_0), RHS=swe.linearised_SWE
            )
