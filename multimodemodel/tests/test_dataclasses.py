"""Test the behavior of the dataclasses."""
import numpy as np
import pytest

from multimodemodel import (
    Parameters,
    Grid,
    Variable,
    State,
    StaggeredGrid,
    GridShift,
    f_constant,
)

try:
    import xarray as xr

    xr_version = xr.__version__
    has_xarray = True
except ModuleNotFoundError:
    has_xarray = False


grid_order = {
    GridShift.LL: "qveu",
    GridShift.LR: "vque",
    GridShift.UL: "uevq",
    GridShift.UR: "euqv",
}


def get_x_y(nx=10.0, ny=10.0, dx=1.0, dy=2.0):
    """Return 2D coordinate arrays."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    assert np.all(X[0, :] == x)
    assert np.all(Y[:, 0] == y)
    return X, Y


def get_test_mask(shape):
    """Return a test ocean mask with the shape of the input coordinate array.

    The mask is zero at the outmost array elements, one elsewhere.
    """
    mask = np.ones(shape, dtype=int)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    return mask


class TestParameters:
    """Test Parameters class."""

    @pytest.mark.parametrize("f0", [0.0, 1.0])
    def test_coriolis_computation(self, f0):
        """Test coriolis parameter computation."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameters(coriolis_func=f_constant(f=f0), on_grid=staggered_grid)

        for var in ("u", "v", "eta"):
            assert np.all(p.f[var] == f0)

    @pytest.mark.parametrize("f", (f_constant(1.0), None))
    @pytest.mark.parametrize("g", (True, False))
    def test_raise_on_missing_coriolis_argument(self, f, g):
        """Test error thrown on missing argument."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))

        if g:
            staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)
        else:
            staggered_grid = None

        p = Parameters(coriolis_func=f, on_grid=staggered_grid)

        if f is None or g is None:
            with pytest.raises(
                RuntimeError, match="Coriolis parameter not available.*"
            ):
                _ = p.f


class TestVariable:
    """Test Variable class."""

    def test_add_data(self):
        """Test variable summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)

        d1 = np.zeros_like(g1.x) + 1.0
        d2 = np.zeros_like(g1.x) + 2.0
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g1)
        v3 = v1 + v2
        assert np.all(v3.data == 3.0)

    def test_add_data_with_none(self):
        """Test summing with None data."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)

        v1 = Variable(np.ones_like(g1.x), g1)
        v2 = Variable(None, g1)
        v3 = v1 + v2
        assert np.all(v3.data == v1.data)
        v3 = v2 + v1
        assert np.all(v3.data == v1.data)

    def test_add_none_with_none(self):
        """Test summing with None data."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)

        v1 = Variable(None, g1)
        v2 = Variable(None, g1)
        v3 = v1 + v2
        assert np.all(v3.data == v1.data)

    def test_grid_mismatch(self):
        """Test grid mismatch detection."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)
        g2 = Grid(x=x, y=y, mask=mask)
        d1 = np.zeros_like(g1.x) + 1.0
        d2 = np.zeros_like(g1.x) + 2.0
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g2)
        with pytest.raises(ValueError) as excinfo:
            _ = v1 + v2
        assert "Try to add variables defined on different grids." in str(excinfo.value)

    def test_not_implemented_add(self):
        """Test missing summation implementation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)
        d1 = np.zeros_like(g1.x) + 1.0
        v1 = Variable(d1, g1)
        with pytest.raises(TypeError) as excinfo:
            _ = v1 + 1.0
        assert "unsupported operand type(s)" in str(excinfo.value)


# @pytest.mark.xarray
@pytest.mark.skipif(not has_xarray, reason="Xarray not available.")
class TestVariableAsDataArray:
    """Test Variable to xarray.DataArray conversion."""

    nx, ny, dx, dy = 10, 5, 1, 2

    def _gen_var(self, data=None, mask=None):
        x, y = get_x_y(self.nx, self.ny, self.dx, self.dy)
        g1 = Grid(x=x, y=y, mask=mask)
        return Variable(data, g1)

    def test_None_data_is_zero(self):
        """Test handling of None as data."""
        v = self._gen_var(data=None)
        assert (v.as_dataarray == 0).where(v.grid.mask).all()

    def test_attribute_is_read_only(self):
        """Test read-only property."""
        v = self._gen_var()
        with pytest.raises(AttributeError, match="can't set attribute"):
            v.as_dataarray = None  # type: ignore

    def test_mask_applied(self):
        """Test if mask is applied properly."""
        v = self._gen_var()
        assert np.isnan(v.as_dataarray.where(v.grid.mask == 0)).all()

    def test_data_assignment(self):
        """Test assignment of data."""
        v = self._gen_var(
            data=np.random.randn(self.ny, self.nx), mask=np.ones((self.ny, self.nx))
        )
        assert (v.as_dataarray == v.data).all()

    def test_coords_and_dims(self):
        """Test coordinate and dimension definition."""
        v = self._gen_var()
        v_da = v.as_dataarray

        assert (v_da.x == v.grid.x).all()
        assert (v_da.y == v.grid.y).all()
        assert v_da.dims == v_da.x.dims == v_da.y.dims == ("i", "j")

    def test_masking_has_no_side_effects(self):
        """Test coordinate and dimension definition."""
        data = np.ones((self.ny, self.nx))
        v = self._gen_var(data=data)
        _ = v.as_dataarray

        assert np.all(v.data == data)


class TestState:
    """Test State class."""

    def test_add(self):
        """Test state summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)
        d1 = np.zeros_like(g1.x) + 1.0
        s1 = State(
            u=Variable(d1, g1),
            v=Variable(d1, g1),
            eta=Variable(d1, g1),
        )
        s2 = State(Variable(d1 * 2, g1), Variable(d1 * 2, g1), Variable(d1 * 2, g1))
        s3 = s1 + s2
        assert np.all(s3.u.data == 3.0)
        assert np.all(s3.v.data == 3.0)
        assert np.all(s3.eta.data == 3.0)
