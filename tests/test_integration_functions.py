"""Test the behaviour of the term functions."""
import sys
import os
import numpy as np
from collections import deque
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



class TestRHS:
    """Test RHS of linear shallow water equations."""

    def test_linearised_SWE(self):
        """Test LSWE."""
        H, g, f = 1, 2, 4
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = get_test_mask(x)
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        eta = np.zeros(x.shape)
        u = np.zeros(x.shape)
        v = 1 * np.ones(x.shape)

        d_u = 4 * np.ones(v.shape)
        d_u[-2,:]   = 2
        d_u[:, 1]   = 2
        d_u[-2,1]   = 1
        d_u = d_u * mask

        d_v = np.zeros_like(u)
        d_eta = np.zeros_like(u)
        d_eta[1:-1,1] = -0.5

        params = swe.Parameters(H=H, g=g, f=f)
        grid = swe.Grid(x, y, mask, e_x, e_y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid)
        )

        assert np.all(swe.linearised_SWE(state, params).u.data == d_u)
        assert np.all(swe.linearised_SWE(state, params).v.data == d_v)
        assert np.all(swe.linearised_SWE(
            state, params).eta.data == d_eta
        )


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
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        eta = 1 * np.ones(x.shape)
        u = 2 * np.ones(x.shape)
        v = 3 * np.ones(x.shape)

        d_u = 9 * np.ones(v.shape)
        d_v = -6 * np.ones(u.shape)
        d_eta = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f, dt=dt)
        grid = swe.Grid(x, y, mask, e_x, e_y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid)
        )
        rhs = deque([swe.linearised_SWE(state, params)], maxlen=1)

        assert np.all(swe.euler_forward(rhs, params).u.data == d_u)
        assert np.all(swe.euler_forward(rhs, params).v.data == d_v)
        assert np.all(swe.euler_forward(rhs, params).eta.data == d_eta)

    def test_adams_bashforth2(self):
        """Test adams_bashforth2."""
        H, g, f = 1, 1, 1
        dt = 1
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        eta = 1 * np.ones(x.shape)
        u = 1 * np.ones(x.shape)
        v = 1 * np.ones(x.shape)

        d_u = 2.5 * np.ones(v.shape)
        d_v = -2.5 * np.ones(u.shape)
        d_eta = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f, dt=dt)
        grid = swe.Grid(x, y, mask, e_x, e_y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid)
        )
        rhs = deque([swe.linearised_SWE(state, params)], maxlen=2)

        assert np.all(swe.adams_bashforth2(rhs, params).u.data == d_u)
        assert np.all(swe.adams_bashforth2(rhs, params).v.data == d_v)
        assert np.all(swe.adams_bashforth2(rhs, params).eta.data == d_eta)

    def test_adams_bashforth3(self):
        """Test adams_bashforth3."""
        H, g, f = 1, 1, 1
        dt = 1
        dx, dy = 1, 2
        ni, nj = 10, 5

        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)
        eta = 1 * np.ones(x.shape)
        u = 1 * np.ones(x.shape)
        v = 1 * np.ones(x.shape)

        d_u = 1 * np.ones(v.shape)
        d_v = -1 * np.ones(u.shape)
        d_eta = np.zeros_like(u)

        params = swe.Parameters(H=H, g=g, f=f, dt=dt)
        grid = swe.Grid(x, y, mask, e_x, e_y)
        state = swe.State(
            u=swe.Variable(u, grid),
            v=swe.Variable(v, grid),
            eta=swe.Variable(eta, grid)
        )

        rhs = deque(
            [
                swe.linearised_SWE(state, params),
                swe.linearised_SWE(state, params),
                swe.linearised_SWE(state, params)
            ],
            maxlen=3
        )

        assert np.all(swe.adams_bashforth3(rhs, params).u.data == d_u)
        assert np.all(swe.adams_bashforth3(rhs, params).v.data == d_v)
        assert np.all(swe.adams_bashforth3(rhs, params).eta.data == d_eta)

    def test_integrator(self):
        """Test integrator."""
        H, g, f = 1, 1, 1
        t_0, t_end, dt = 0, 1, 1
        dx, dy = 1, 2
        ni, nj = 10, 5
        x, y = get_x_y(ni, nj, dx, dy)
        mask = np.ones(x.shape)
        e_x = np.ones(x.shape)
        e_y = np.ones(y.shape)

        eta_0 = 1 * np.ones(x.shape)
        u_0 = 1 * np.ones(x.shape)
        v_0 = 1 * np.ones(x.shape)

        eta_1 = 1 * np.ones(x.shape)
        u_1 = 2 * np.ones(x.shape)
        v_1 = 0 * np.ones(x.shape)

        params = swe.Parameters(H=H, g=g, f=f, t_0=t_0, t_end=t_end, dt=dt)
        grid = swe.Grid(x, y, mask, e_x, e_y)
        state_0 = swe.State(
            u=swe.Variable(u_0, grid),
            v=swe.Variable(v_0, grid),
            eta=swe.Variable(eta_0, grid)
        )
        state_1 = swe.integrator(
            state_0, params,
            scheme=swe.euler_forward,
            RHS=swe.linearised_SWE
        )
        assert np.all(state_1.u.data == u_1)
        assert np.all(state_1.v.data == v_1)
        assert np.all(state_1.eta.data == eta_1)
