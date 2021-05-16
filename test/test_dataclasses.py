""" Test the behaviour of the dataclasses.
"""
import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
import numpy as np
import pytest

from shallow_water_eqs import Parameters, Grid, Variable, State

def get_x_y(nx, ny, dx, dy):
    return np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx)[::-1]


class TestParameters:

    def test_default_values(self):
        p = Parameters()
        defaults = {
            'f': 0.,
            'g': 9.81,
            'beta': 2./(24*3600),
            'H': 1000.,
            'dt': 8.,
            't_0': 0.,
            't_end': 3600.,
            'write': 20.,
        }
        for var, val in defaults.items():
            assert p.__getattribute__(var) == val


class TestGrid:

    def test_post_init(self):
        nx, ny = 10, 5
        dx, dy = 1., 2.
        x, y = get_x_y(nx, ny, dx, dy)
        
        g1 = Grid(x=x, y=y)
        assert g1.dx == dx
        assert g1.dy == dy
        assert g1.len_x == nx
        assert g1.len_y == ny

    def test_dim_def(self):
        nx, ny = 10, 5
        dx, dy = 1., 2.
        x, y = get_x_y(nx, ny, dx, dy)

        g2 = Grid(x=x.T, y=y.T, dim_x=1, dim_y=0)
        assert g2.len_x == nx
        assert g2.len_y == ny
        assert g2.dx == dx
        assert g2.dy == dy


class TestVariable:

    def test_add_data(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        g1 = Grid(*get_x_y(nx, ny, dx, dy))
        d1 = np.zeros_like(g1.x) + 1.
        d2 = np.zeros_like(g1.x) + 2.
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g1)
        v3 = v1 + v2
        assert np.all(v3.data == 3.)

    def test_grid_mismatch(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        g1 = Grid(*get_x_y(nx, ny, dx, dy))
        g2 = Grid(*get_x_y(nx, ny, dx, dy))
        d1 = np.zeros_like(g1.x) + 1.
        d2 = np.zeros_like(g1.x) + 2.
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g2)
        with pytest.raises(ValueError) as excinfo:
            v3 = v1 + v2
        assert "Try to add variables defined on different grids." in str(excinfo.value)

    def test_not_implemented_add(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        g1 = Grid(*get_x_y(nx, ny, dx, dy))
        d1 = np.zeros_like(g1.x) + 1.
        v1 = Variable(d1, g1)
        with pytest.raises(TypeError) as excinfo:
            v3 = v1 + 1.
        assert "unsupported operand type(s)" in str(excinfo.value)


class TestState:

    def test_add(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        g1 = Grid(*get_x_y(nx, ny, dx, dy))
        d1 = np.zeros_like(g1.x) + 1.
        s1 = State(
            u=Variable(d1, g1),
            v=Variable(d1, g1),
            eta=Variable(d1, g1),
        )
        s2 = State(
            Variable(d1 * 2, g1),
            Variable(d1 * 2, g1),
            Variable(d1 * 2, g1)
        )
        s3 = s1 + s2
        assert np.all(s3.u.data == 3.)
        assert np.all(s3.v.data == 3.)
        assert np.all(s3.eta.data == 3.)
