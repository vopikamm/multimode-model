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
    mask[..., 0, :] = 0
    mask[..., -1, :] = 0
    mask[..., :, 0] = 0
    mask[..., :, -1] = 0
    return mask


class TestParameters:
    """Test Parameters class."""

    @pytest.mark.parametrize("f0", [0.0, 1.0])
    def test_coriolis_computation(self, f0):
        """Test coriolis parameter computation."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))

        # 2D
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)
        p = Parameters(coriolis_func=f_constant(f=f0), on_grid=staggered_grid)
        for var in ("u", "v", "eta"):
            assert np.all(p.f[var] == f0)
            assert p.f[var].ndim == 2

        # 3D
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y, z=np.arange(10.0))
        p = Parameters(coriolis_func=f_constant(f=f0), on_grid=staggered_grid)
        for var in ("u", "v", "eta"):
            assert np.all(p.f[var] == f0)
            assert p.f[var].ndim == 2

    @pytest.mark.parametrize("f", (f_constant(1.0), None))
    @pytest.mark.parametrize("g", (True, False))
    def test_raise_on_missing_coriolis_argument(self, f, g):
        """Test error thrown on missing argument."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))

        if g:
            staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y, z=np.arange(10.0))
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

    @pytest.mark.parametrize("data", (np.zeros((5, 10)), np.zeros((10, 5, 10))))
    @pytest.mark.parametrize(
        "grid",
        (Grid(*get_x_y(10, 5, 1, 1)), Grid(*get_x_y(10, 5, 1, 1), z=np.arange(10))),
    )
    def test_data_grid_shape_missmatch(self, data, grid):
        """Test detection of shape missmatch."""
        if data.shape != grid.shape:
            with pytest.raises(ValueError, match="Shape of data and grid missmatch.*"):
                _ = Variable(data, grid)
        else:
            _ = Variable(data, grid)

    def test_copy(self):
        """Test that deepcopy of data and reference to grid is returned."""
        shape = (10, 5)
        var = Variable(np.ones(shape[::-1]), Grid(*get_x_y(*shape, 1, 1)))
        var_copy = var.copy()
        assert not np.may_share_memory(var.data, var_copy.data)
        assert var.grid is var_copy.grid

    def test_copy_none(self):
        """Test copying None data."""
        shape = (10, 5)
        var_copy = Variable(None, Grid(*get_x_y(*shape, 1, 1))).copy()
        assert var_copy.data is None

    def test_add_data(self):
        """Test variable summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        g1 = Grid(x=x, y=y)
        g1.mask = get_test_mask(g1.shape)

        d1 = np.zeros_like(g1.x) + 1.0
        d2 = np.zeros_like(g1.x) + 2.0
        v1 = Variable(d1, g1)
        v2 = Variable(d2, g1)
        v3 = v1 + v2
        assert np.all(v3.data == 3.0)
        assert np.all(v3.grid.mask == v1.grid.mask)

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


@pytest.mark.skipif(not has_xarray, reason="Xarray not available.")
class TestVariableAsDataArray:
    """Test Variable to xarray.DataArray conversion."""

    nx, ny, dx, dy = 10, 5, 1, 2

    def _gen_var(self, data=None, mask=None, has_z=False):
        x, y = get_x_y(self.nx, self.ny, self.dx, self.dy)
        z = np.arange(10.0) if has_z else None
        g1 = Grid(x=x, y=y, z=z, mask=mask)
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
        assert np.isnan(v.as_dataarray.where(v.grid.mask == 0)).all()  # type: ignore

    def test_data_assignment(self):
        """Test assignment of data."""
        v = self._gen_var(
            data=np.random.randn(self.ny, self.nx), mask=np.ones((self.ny, self.nx))
        )
        assert (v.as_dataarray == v.data).all()

    @pytest.mark.parametrize("has_z", (True, False))
    def test_coords_and_dims(self, has_z):
        """Test coordinate and dimension definition."""
        v = self._gen_var(has_z=has_z)
        v_da = v.as_dataarray

        assert (v_da.x == v.grid.x).all()  # type: ignore
        assert (v_da.y == v.grid.y).all()  # type: ignore
        assert (
            v_da.dims[-2:] == v_da.x.dims == v_da.y.dims == ("j", "i")  # type: ignore
        )
        if has_z:
            assert (v_da.z == v.grid.z).all()  # type: ignore
            assert v_da.dims[v.grid.dim_z] == "z"  # type: ignore
        else:
            assert "z" not in v_da.coords  # type: ignore
            assert "z" not in v_da.dims  # type: ignore

    def test_masking_has_no_side_effects(self):
        """Test if masking has side effects."""
        data = np.ones((self.ny, self.nx))
        v = self._gen_var(data=data)
        _ = v.as_dataarray

        assert np.all(v.data == data)


class TestState:
    """Test State class."""

    def test_init_raise_on_non_variable_keyword_argument(self):
        """Test if keyword argument of type other than Variable raises error."""
        with pytest.raises(ValueError, match="Keyword argument"):
            _ = State(u=None)

    def test_add(self):
        """Test state summation."""
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x=x, y=y, mask=mask)
        d1 = np.ones_like(g1.x)
        s1 = State(
            u=Variable(d1, g1),
            v=Variable(d1, g1),
        )
        s2 = State(
            u=Variable(d1 * 2, g1),
            eta=Variable(d1 * 2, g1),
        )
        s3 = s1 + s2
        assert np.all(s3.variables["u"].data == 3.0)
        assert np.all(s3.variables["v"].data == 1.0)
        assert np.all(s3.variables["eta"].data == 2.0)

    def test_attribute_is_reference_to_dict_value(self):
        """Test if variable exposed by attribute is a reference to dictionary value."""
        g1 = Grid(*get_x_y(10, 10, 1, 1))
        s = State(var=Variable(np.ones(g1.shape), g1))
        assert s.var is s.variables["var"]
