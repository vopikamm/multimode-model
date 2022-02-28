"""Test the behavior of the dataclasses."""
# flake8: noqa
import numpy as np
import pytest

from multimodemodel import (
    GridShift,
    Grid,
    StaggeredGrid,
    Parameter,
    Variable,
    State,
    StateDeque,
    f_constant,
    beta_plane,
    f_on_sphere,
    sum_states,
    Domain,
)

from multimodemodel.kernel import sum_vars
from multimodemodel.util import str_to_date, add_time

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

some_datetime = str_to_date("2000-01-01")


def _not_share_memory(a1, a2):
    if a1 is not None and a2 is not None:
        assert not np.may_share_memory(a1, a2)


@pytest.fixture(
    params=(
        # (100, 50),
        (20, 15),
    )
)
def staggered_grid(request):
    nx, ny = request.param

    args = dict(
        x=np.arange(nx),
        y=np.arange(ny),
        mask=None,
    )

    return StaggeredGrid.cartesian_c_grid(**args)


@pytest.fixture(
    params=[
        (f_constant, (1.0,)),
        (beta_plane, (10, 0.1, 0.0)),
        (f_on_sphere, ()),
    ]
)
def coriolis_func(request):
    return request.param[0](*request.param[1])


@pytest.fixture
def param(staggered_grid, coriolis_func):
    return Parameter(
        coriolis_func=coriolis_func,
        on_grid=staggered_grid,
    )


@pytest.fixture(params=[True, False])
def state_param(staggered_grid, coriolis_func, request):
    param = Parameter(coriolis_func=coriolis_func, on_grid=staggered_grid)
    u = Variable(
        np.arange(staggered_grid.u.x.size).reshape(staggered_grid.u.x.shape) + 0.0,
        staggered_grid.u,
        time=some_datetime,
    )
    v = Variable(
        np.arange(staggered_grid.v.x.size).reshape(staggered_grid.v.x.shape) + 1.0,
        staggered_grid.v,
        time=some_datetime,
    )
    eta = Variable(
        np.arange(staggered_grid.eta.x.size).reshape(staggered_grid.eta.x.shape) + 2.0,
        staggered_grid.eta,
        time=some_datetime,
    )
    if request.param:
        eta.data = None
    return State(u=u, v=v, eta=eta), param


@pytest.fixture(params=[0, 99])
def ident(request):
    return request.param


@pytest.fixture
def domain_state(state_param):
    state, param = state_param
    return Domain(state, history=StateDeque(), parameter=param, iteration=0, id=0)


@pytest.fixture(params=[-1, 0, 99])
def it(request):
    return request.param


@pytest.fixture(params=(1, 2, 3))
def dt(request):
    return request.param


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
    mask = np.ones(shape, dtype=int)
    mask[..., 0, :] = 0
    mask[..., -1, :] = 0
    mask[..., :, 0] = 0
    mask[..., :, -1] = 0
    return mask


class TestParameter:
    """Test Parameters class."""

    @pytest.mark.parametrize("f0", [0.0, 1.0])
    def test_coriolis_computation(self, f0):
        """Test coriolis parameter computation."""
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))

        # 2D
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)
        p = Parameter(coriolis_func=f_constant(f=f0), on_grid=staggered_grid)
        for var in ("u", "v", "eta"):
            assert np.all(p.f[var] == f0)
            assert p.f[var].ndim == 2

        # 3D
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y, z=np.arange(10.0))
        p = Parameter(coriolis_func=f_constant(f=f0), on_grid=staggered_grid)
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

        p = Parameter(coriolis_func=f, on_grid=staggered_grid)

        if f is None or g is None:
            with pytest.raises(
                RuntimeError, match="Coriolis parameter not available.*"
            ):
                _ = p.f

    @pytest.mark.parametrize("f", (f_constant(1.0), None))
    def test_hash_of_identical_objects_is_same(self, f):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f, on_grid=staggered_grid)
        p2 = p
        assert hash(p) == hash(p2)

    @pytest.mark.parametrize("f", (f_constant(1.0), None))
    def test_hash_of_same_objects_is_false(self, f):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f, on_grid=staggered_grid)
        p2 = Parameter(coriolis_func=f, on_grid=staggered_grid)
        assert hash(p) != hash(p2)

    @pytest.mark.parametrize("f", (f_constant(1.0), None))
    def test_hash_of_different_objects_is_false(self, f):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f, on_grid=staggered_grid)
        p2 = Parameter(coriolis_func=f_constant(0.0), on_grid=staggered_grid)
        assert hash(p) != hash(p2)

    def test_comparison_with_identical_returns_true(self):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f_constant(f=1.0), on_grid=staggered_grid)
        p2 = p
        assert p == p2

    def test_comparison_with_same_returns_true(self):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f_constant(f=1.0), on_grid=staggered_grid)
        p2 = Parameter(coriolis_func=f_constant(f=1.0), on_grid=staggered_grid)
        assert p == p2

    def test_comparison_with_different_returns_false(self):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f_constant(f=1.0), on_grid=staggered_grid)
        p2 = Parameter(coriolis_func=f_constant(f=2.0), on_grid=staggered_grid)
        assert p != p2

    def test_comparison_with_wrong_type_returns_false(self):
        nx, ny = 20, 10
        dx, dy = 1.0, 0.25
        x, y = (np.arange(0.0, n * d, d) for (n, d) in ((nx, dx), (ny, dy)))
        staggered_grid = StaggeredGrid.cartesian_c_grid(x=x, y=y)

        p = Parameter(coriolis_func=f_constant(f=1.0), on_grid=staggered_grid)
        assert p != 5


class TestVariable:
    """Test Variable class."""

    def _get_grid(self, nx=10, ny=5, nz=2, dx=1, dy=2):
        x, y, z = get_x_y_z(nx, ny, nz, dx, dy)
        return Grid(x=x, y=y, z=z, mask=get_test_mask((nz, ny, nx)))

    @pytest.mark.parametrize("data", (np.zeros((5, 10)), np.zeros((10, 5, 10))))
    @pytest.mark.parametrize(
        "grid",
        (Grid(*get_x_y(10, 5, 1, 1)), Grid(*get_x_y(10, 5, 1, 1), z=np.arange(10))),
    )
    def test_data_grid_shape_missmatch(self, data, grid):
        """Test detection of shape missmatch."""
        if data.shape != grid.shape:
            print(data.shape, grid.shape)
            with pytest.raises(ValueError, match="Shape of data and grid missmatch.*"):
                _ = Variable(data, grid, some_datetime)
        else:
            _ = Variable(data, grid, some_datetime)

    def test_copy(self):
        """Test that deepcopy of data and reference to grid is returned."""
        shape = (10, 5, 2)
        var = Variable(
            np.ones(shape[::-1]),
            self._get_grid(*shape, 1, 1),
            time=np.datetime64("2000-01-01", "s"),
        )
        var_copy = var.copy()
        assert not np.may_share_memory(var.data, var_copy.data)
        assert var.grid is var_copy.grid
        assert var.time is not var_copy.time
        assert var.time == var_copy.time

    def test_copy_none(self):
        """Test copying None data."""
        shape = (10, 5)
        var_copy = Variable(None, Grid(*get_x_y(*shape, 1, 1)), some_datetime).copy()
        assert var_copy.data is None

    def test__grid_type_is_Grid(self):
        """Test value of _stype attribute."""
        assert Variable._grid_type() is Grid

    def test_add_data(self):
        """Test variable summation."""
        t1 = str_to_date("2000-01-01")
        t2 = add_time(t1, 2)
        g1 = self._get_grid()

        d1 = np.zeros(g1.shape) + 1.0
        d2 = np.zeros(g1.shape) + 2.0
        v1 = Variable(d1, g1, t1)
        v2 = Variable(d2, g1, t2)
        v3 = v1 + v2
        assert np.all(v3.data == 3.0)
        print(v3.data == sum_vars((v1, v2)).data)
        print(v3.grid == sum_vars((v1, v2)).grid)
        print(v3.time == sum_vars((v1, v2)).time)
        assert v3 == sum_vars((v1, v2))
        assert np.all(v3.grid.mask == v1.grid.mask)
        assert v3.time == t1 + np.timedelta64(1, "s")

    def test_add_data_with_none(self):
        """Test summing with None data."""
        t1 = str_to_date("2000-01-01")
        t2 = add_time(t1, 2)
        g1 = self._get_grid()

        v1 = Variable(np.ones(g1.shape), g1, t1)
        v2 = Variable(None, g1, t2)
        v3 = v1 + v2
        assert np.all(v3.data == v1.data)
        v3 = v2 + v1
        assert np.all(v3.data == v1.data)
        assert v3 == sum_vars((v1, v2))

    def test_add_none_with_none(self):
        """Test summing with None data."""
        t1 = str_to_date("2000-01-01")
        t2 = add_time(t1, 2)
        g1 = self._get_grid()

        v1 = Variable(None, g1, t1)
        v2 = Variable(None, g1, t2)
        v3 = v1 + v2
        assert np.all(v3.data == v1.data)
        assert v3 == sum_vars((v1, v2))

    def test_add_grid_mismatch(self):
        """Test grid mismatch detection."""
        t1 = str_to_date("2000-01-01")
        t2 = add_time(t1, 2)
        g1 = self._get_grid()
        g2 = self._get_grid()
        g2.x[:] = g2.x + 1.0
        d1 = np.zeros(g1.shape) + 1.0
        d2 = np.zeros(g2.shape) + 2.0
        v1 = Variable(d1, g1, t1)
        v2 = Variable(d2, g2, t2)
        with pytest.raises(ValueError) as excinfo:
            _ = v1 + v2
        assert "Try to add variables defined on different grids." in str(excinfo.value)

    def test_not_implemented_add(self):
        """Test missing summation implementation."""
        t1 = str_to_date("2000-01-01")
        g1 = self._get_grid()
        d1 = np.zeros(g1.shape) + 1.0
        v1 = Variable(d1, g1, t1)
        with pytest.raises(TypeError) as excinfo:
            _ = v1 + 1.0  # type: ignore
        assert "unsupported operand type(s)" in str(excinfo.value)

    def test_comparison_with_identical_returns_true(self):
        g1 = self._get_grid()

        d1 = np.zeros(g1.shape) + 1.0
        v1 = Variable(d1, g1, some_datetime)
        v2 = v1
        assert v1 == v2

    def test_comparison_with_same_returns_true(self):
        g1 = self._get_grid()

        d1 = np.zeros(g1.shape) + 1.0
        v1 = Variable(d1, g1, some_datetime)
        v2 = Variable(d1.copy(), g1, some_datetime)
        assert v1 == v2

    def test_comparison_with_both_none_data_returns_true(self):
        g1 = self._get_grid()

        v1 = Variable(None, g1, some_datetime)
        v2 = Variable(None, g1, some_datetime)
        assert v1 == v2

    def test_comparison_with_different_returns_false(self):
        g1 = self._get_grid()

        v1 = Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime)
        v2 = Variable(np.zeros(g1.shape) + 2.0, g1, some_datetime)
        assert v1 != v2

    def test_comparison_with_wrong_type_returns_false(self):
        g1 = self._get_grid()

        v1 = Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime)
        assert v1 != 5


@pytest.mark.skipif(not has_xarray, reason="Xarray not available.")
class TestVariableAsDataArray:
    """Test Variable to xarray.DataArray conversion."""

    nx, ny, nz, dx, dy = 10, 5, 1, 1, 2

    def _gen_var(self, data=None, mask=None):
        x, y, z = get_x_y_z(self.nx, self.ny, self.nz, self.dx, self.dy)
        g1 = Grid(x=x, y=y, z=z, mask=mask)
        return Variable(data, g1, some_datetime)

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
            data=np.random.randn(self.nz, self.ny, self.nx),
            mask=np.ones((self.nz, self.ny, self.nx)),
        )
        assert (v.as_dataarray == v.data).all()

    @pytest.mark.parametrize("has_z", (True, False))
    def test_coords_and_dims(self, has_z):
        """Test coordinate and dimension definition."""
        v = self._gen_var()
        v_da = v.as_dataarray

        assert (v_da.x == v.grid.x).all()  # type: ignore
        assert (v_da.y == v.grid.y).all()  # type: ignore
        assert (
            v_da.dims[-2:] == v_da.x.dims == v_da.y.dims == ("j", "i")  # type: ignore
        )
        assert v_da.dims[0] == "time"  # type: ignore

        assert (v_da.z == v.grid.z).all()  # type: ignore
        assert v_da.dims[v.grid.dim_z] == "z"  # type: ignore

    def test_masking_has_no_side_effects(self):
        """Test if masking has side effects."""
        data = np.ones((self.nz, self.ny, self.nx))
        v = self._gen_var(data=data)
        _ = v.as_dataarray

        assert np.all(v.data == data)


class TestState:
    """Test State class."""

    def _get_grid(self, nx=10, ny=5, nz=1, dx=1, dy=2):
        x, y, z = get_x_y_z(nx, ny, nz, dx, dy)
        return Grid(x=x, y=y, z=z, mask=get_test_mask((nz, ny, nx)))

    def test__variable_type_is_Variable(self):
        """Test value of _stype attribute."""
        assert State._variable_type() is Variable

    def test_variables_dict_set_correctly(self):
        """Test state summation."""
        g1 = self._get_grid()
        d1 = np.zeros(g1.shape) + 1.0
        s1 = State(
            u=Variable(d1, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        assert s1.u is s1.variables["u"]
        assert s1.v is s1.variables["v"]
        assert s1.eta is s1.variables["eta"]

    def test_init_raise_on_non_variable_keyword_argument(self):
        """Test if keyword argument of type other than Variable raises error."""
        with pytest.raises(ValueError, match="Keyword argument"):
            _ = State(u=None)

    def test_add(self):
        """Test state summation."""
        t1 = str_to_date("2000-01-01")
        t2 = add_time(t1, 2)
        g1 = self._get_grid()
        d1 = np.ones(g1.shape)
        s1 = State(
            u=Variable(d1, g1, t1),
            v=Variable(d1, g1, t1),
        )
        s2 = State(
            u=Variable(d1 * 2, g1, t2),
            eta=Variable(d1 * 2, g1, t2),
        )
        s3 = s1 + s2
        assert np.all(s3.variables["u"].data == 3.0)
        assert np.all(s3.variables["v"].data == 1.0)
        assert np.all(s3.variables["eta"].data == 2.0)
        assert s3.variables["u"].time == t1 + np.timedelta64(1, "s")
        assert s3.variables["v"].time == t1
        assert s3.variables["eta"].time == t2

    def test_attribute_is_reference_to_dict_value(self):
        """Test if variable exposed by attribute is a reference to dictionary value."""
        g1 = Grid(*get_x_y(10, 10, 1, 1))
        s = State(var=Variable(np.ones(g1.shape), g1, some_datetime))
        assert s.var is s.variables["var"]  # type: ignore

    def test_comparison_with_identical_returns_true(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x, y, mask=mask)
        d1 = np.zeros(g1.shape) + 1.0
        s1 = State(
            u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        s2 = s1
        assert s1 == s2

    def test_comparison_with_same_returns_true(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x, y, mask=mask)
        d1 = np.zeros(g1.shape) + 1.0
        s1 = State(
            u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        s2 = State(
            u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        assert s1 == s2

    def test_comparison_with_different_returns_false(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x, y, mask=mask)
        d1 = np.zeros(g1.shape) + 1.0
        s1 = State(
            u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        s2 = State(
            u=Variable(np.zeros(g1.shape) + 2.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        assert s1 != s2

    def test_comparison_with_wrong_type_returns_false(self):
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x, y, mask=mask)
        d1 = np.zeros(g1.shape) + 1.0
        s1 = State(
            u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
            v=Variable(d1, g1, some_datetime),
            eta=Variable(d1, g1, some_datetime),
        )
        assert s1 != 5

    @pytest.mark.parametrize("n_states", range(2, 11))
    def test_sum_states_sums_correctly(self, n_states):
        nx, ny, dx, dy = 10, 5, 1, 2
        x, y = get_x_y(nx, ny, dx, dy)
        mask = get_test_mask(x.shape)
        g1 = Grid(x, y, mask=mask)
        d1 = np.zeros(g1.shape) + 1.0
        states = n_states * (
            State(
                u=Variable(np.zeros(g1.shape) + 1.0, g1, some_datetime),
                v=Variable(d1, g1, some_datetime),
                eta=Variable(d1, g1, some_datetime),
            ),
        )
        states[-1].eta.data = None
        start_state = State(
            u=Variable(np.zeros(d1.shape), g1, some_datetime),
            v=Variable(np.zeros(d1.shape), g1, some_datetime),
            eta=Variable(np.zeros(d1.shape), g1, some_datetime),
        )
        assert sum(states, start=start_state) == sum_states(states)


class TestStateDeque:
    def test__state_type_is_State(self):
        """Test value of _stype attribute."""
        assert StateDeque._state_type() is State


class TestDomain:
    def test_type_class_variables(self, domain_state):
        assert domain_state._stype is State
        assert domain_state._ptype is Parameter

    def test_init_None_history_to_empty_StateDeque(self, state_param):
        state, param = state_param
        ds = Domain(state, None, param)
        assert isinstance(ds.history, StateDeque)
        assert len(ds.history) == 0
        assert ds.history.maxlen == 3

    def test_set_id(self, domain_state, ident):
        domain_state.id = ident
        assert domain_state.id == ident

    def test_get_id(self, domain_state, ident):
        domain_state.id = ident
        assert ident == domain_state.id

    def test_get_iteration(self, domain_state, it):
        domain_state.iteration = it
        assert domain_state.iteration == it

    def test_increment_iteration(self, domain_state, it):
        domain_state.iteration = it
        assert domain_state.increment_iteration() == it + 1

    def test_copy(self, domain_state):
        ds_copy = domain_state.copy()
        assert ds_copy == domain_state
        assert id(ds_copy) is not id(domain_state)
        assert ds_copy == domain_state
        tuple(
            _not_share_memory(
                getattr(ds_copy.state, v).data, getattr(domain_state.state, v).data
            )
            for v in ("u", "v", "eta")
        )
        assert all(
            id(getattr(ds_copy.state, v).grid)
            is not id(getattr(domain_state.state, v).grid)
            for v in ("u", "v", "eta")
        )
        tuple(
            (
                _not_share_memory(
                    getattr(getattr(ds_copy.state, v).grid, c),
                    getattr(getattr(domain_state.state, v).grid, c),
                )
                for c in ("x", "y", "mask")
            )
            for v in ("u", "v", "eta")
        )

    def test_comparison_with_identical_returns_true(self, domain_state):
        d2 = domain_state
        assert domain_state == d2

    def test_comparison_with_same_returns_true(self, domain_state):
        d2 = domain_state.copy()
        assert domain_state == d2

    def test_comparison_with_different_returns_false(self, domain_state):
        d2 = domain_state.copy()
        d2.id = 100
        assert domain_state != d2

    def test_comparison_with_wrong_type_returns_false(self, domain_state):
        assert domain_state != 5
