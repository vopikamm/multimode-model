"""Linear shallow water model.

This script is the initial attempt to formulate the linearised shallow
water equation as python functions.
"""
import numpy as np
import timeit
from typing import Callable, Tuple, Any
from numba import njit
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from collections import deque
from inspect import signature


"""
Dataclasses building instances to hold parameters, dynamic variables
and their associated grids.
"""


@dataclass
class Parameters:
    """Class to organise all constant parameters."""

    f: float = 0.0
    g: float = 9.81  # gravitational force m/s^2
    beta: float = 2.0 / (24 * 3600)  # beta parameter 1/ms with f=f_0+beta *y
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    dt: float = 1.0  # time stepping in s
    t_0: float = 0.0  # starting time
    t_end: float = 3600.0  # end time
    write: int = 20  # how many states should be output
    r: float = 6371000.0  # radius of the earth


@dataclass
class Grid:
    """Grid informtation."""

    x: np.array  # longitude on grid
    y: np.array  # latitude on grid
    mask: np.array  # ocean mask, 1 where ocean is, 0 where land is
    dim_x: int = 0  # x dimension in numpy array
    dim_y: int = 1  # y dimension in numpy array
    dx: int = field(init=False)  # grid spacing in x
    dy: int = field(init=False)  # grid spacing in y
    len_x: int = field(init=False)  # length of array in x dimension
    len_y: int = field(init=False)  # length of array in y dimension

    def _compute_grid_spacing(self):
        """Compute the spatial differences along x and y."""
        dx = np.diff(self.x, axis=self.dim_x)
        dy = np.diff(self.y, axis=self.dim_y)
        if self.dim_x == 0:
            dx_0 = dx[0, :]
            dy_0 = dy[:, 0]
        else:
            dx_0 = dx[:, 0]
            dy_0 = dy[0, :]
        dx = np.append(dx, np.expand_dims(dx_0, axis=self.dim_x), axis=self.dim_x)
        dy = np.append(dy, np.expand_dims(dy_0, axis=self.dim_y), axis=self.dim_y)
        return dx, dy

    def _get_u_mask(self) -> np.array:
        """U mask. Land when meridional neighbouring q-points are land."""
        return np.maximum(np.roll(self.mask, axis=self.dim_y, shift=1), self.mask)

    def _get_v_mask(self) -> np.array:
        """V mask. Land when zonal neighbouring q-points are land."""
        return np.maximum(np.roll(self.mask, axis=self.dim_x, shift=1), self.mask)

    def _get_eta_mask(self) -> np.array:
        """Eta mask. Land when all four surrounding q-points are land."""
        along_0 = np.maximum(np.roll(self.mask, axis=0, shift=1), self.mask)
        return np.maximum(np.roll(along_0, axis=1, shift=1), along_0)

    @classmethod
    def regular_lat_lon(
        cls: Any,
        lon_start: float = 0.0,
        lon_end: float = 60.0,
        lat_start: float = 0.0,
        lat_end: float = 60.0,
        nx: int = 61,
        ny: int = 61,
        dim_x: int = 0,
        r: float = 6371000.0,
    ):
        """Generate a regular lat/lon grid."""
        indexing = ["ij", "xy"]
        to_rad = np.pi / 180.0
        lon = np.linspace(lon_start, lon_end, nx)
        lat = np.linspace(lat_start, lat_end, ny)
        longitude, latitude = np.meshgrid(lon, lat, indexing=indexing[dim_x])
        mask = np.ones(longitude.shape)
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0

        grid = cls(
            x=longitude,
            y=latitude,
            mask=mask,
            dim_x=dim_x,
            dim_y=1 - dim_x,
        )

        grid.dx = r * np.cos(grid.y * to_rad) * grid.dx * to_rad
        grid.dy = r * grid.dy * to_rad

        return grid

    def __post_init__(self) -> None:
        """Set derived attributes of the grid."""
        self.dx, self.dy = self._compute_grid_spacing()
        self.len_x = self.x.shape[self.dim_x]
        self.len_y = self.x.shape[self.dim_y]


@dataclass
class StaggeredGrid:
    """The u, v, and eta grids are evaluated on staggered gridpoints."""

    q_grid: Grid
    u_grid: Grid
    v_grid: Grid
    eta_grid: Grid

    @classmethod
    def c_grid(
        cls: Any,
        func: Callable[..., Grid],
        **kwargs_to_callable: Tuple[Any],
    ):
        """Generate an Arakawa C-grid based on a given Grid() classmethod."""
        q_grid = func(**kwargs_to_callable)
        list_of_grid = [q_grid]
        u_grid = list_of_grid.copy()[0]
        v_grid = list_of_grid.copy()[0]
        eta_grid = list_of_grid.copy()[0]
        dx, dy = q_grid._compute_grid_spacing()

        u_grid.y = u_grid.y + dy / 2
        u_grid.mask = u_grid._get_u_mask()

        v_grid.x = v_grid.x + dx / 2
        v_grid.mask = v_grid._get_v_mask()

        eta_grid.x = eta_grid.x + dx / 2
        eta_grid.y = eta_grid.y + dy / 2
        eta_grid.mask = u_grid._get_eta_mask()

        return StaggeredGrid(q_grid, u_grid, v_grid, eta_grid)


@dataclass
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: np.array
    grid: Grid

    def __add__(self, other):
        """Add data of to variables."""
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")
        try:
            new_data = self.data + other.data
        except (TypeError, AttributeError):
            return NotImplemented
        return self.__class__(data=new_data, grid=self.grid)


@dataclass
class State:
    """State class.

    Combines the dynamical variables u,v, eta into one state object.
    """

    u: Variable
    v: Variable
    eta: Variable

    def __add__(self, other):
        """Add all variables of two states."""
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented
        try:
            u_new = self.u + other.u
            v_new = self.v + other.v
            eta_new = self.eta + other.eta
        except (AttributeError, TypeError):
            return NotImplemented
        return self.__class__(u=u_new, v=v_new, eta=eta_new)


"""
Jit-able functions. Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""


@njit(inline="always")
def _expand_10_arguments(func, i, j, ni, nj, args):
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5])


@njit(inline="always")
def _expand_9_arguments(func, i, j, ni, nj, args):
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4])


@njit(inline="always")
def _expand_8_arguments(func, i, j, ni, nj, args):
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3])


@njit(inline="always")
def _expand_7_arguments(func, i, j, ni, nj, args):
    return func(i, j, ni, nj, args[0], args[1], args[2])


@njit(inline="always")
def _expand_6_arguments(func, i, j, ni, nj, args):
    return func(i, j, ni, nj, args[0], args[1])


_arg_expand_map = {
    6: _expand_6_arguments,
    7: _expand_7_arguments,
    8: _expand_8_arguments,
    9: _expand_9_arguments,
    10: _expand_10_arguments,
}


def _numba_2D_grid_iterator(func):
    jitted_func = njit(inline="always")(func)
    exp_args = _arg_expand_map[len(signature(func).parameters)]

    @njit
    def _interate_over_grid_2D(ni: int, nj: int, *args: Tuple[Any]):
        result = np.empty((ni, nj))
        for i in range(ni):
            for j in range(nj):
                result[i, j] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _interate_over_grid_2D


@_numba_2D_grid_iterator
def _zonal_pressure_gradient(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.array,
    mask: np.array,
    g: float,
    dx_u: np.array,
    dy_u: np.array,
    dy_eta: np.array,
) -> float:
    return (
        -g
        * (
            mask[i, j] * dy_eta[i, j] * eta[i, j]
            - mask[i - 1, j] * dy_eta[i - 1, j] * eta[i - 1, j]
        )
        / dx_u[i, j]
        / dy_u[i, j]
    )


@_numba_2D_grid_iterator
def _meridional_pressure_gradient(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.array,
    mask: np.array,
    g: float,
    dx_v: np.array,
    dy_v: np.array,
    dx_eta: np.array,
) -> float:
    return (
        -g
        * (
            mask[i, j] * dx_eta[i, j] * eta[i, j]
            - mask[i, j - 1] * dx_eta[i, j - 1] * eta[i, j - 1]
        )
        / dx_v[i, j]
        / dy_v[i, j]
    )


@_numba_2D_grid_iterator
def _zonal_divergence(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: np.array,
    mask: np.array,
    H: float,
    dx_eta: np.array,
    dy_eta: np.array,
    dy_u: np.array,
) -> float:
    ip1 = (i + 1) % ni
    return (
        -H
        * (mask[ip1, j] * dy_u[ip1, j] * u[ip1, j] - mask[i, j] * dy_u[i, j] * u[i, j])
        / dx_eta[i, j]
        / dy_eta[i, j]
    )


@_numba_2D_grid_iterator
def _meridional_divergence(
    i: int,
    j: int,
    ni: int,
    nj: int,
    v: np.array,
    mask: np.array,
    H: float,
    dx_eta: np.array,
    dy_eta: np.array,
    dx_v: np.array,
) -> float:
    jp1 = (j + 1) % nj
    return (
        -H
        * (mask[i, jp1] * dx_v[i, jp1] * v[i, jp1] - mask[i, j] * dx_v[i, j] * v[i, j])
        / dx_eta[i, j]
        / dy_eta[i, j]
    )


@_numba_2D_grid_iterator
def _coriolis_u(
    i: int, j: int, ni: int, nj: int, u: np.array, mask: np.array, f: float
) -> float:
    ip1 = (i + 1) % ni
    return (
        -f
        * (
            mask[i, j - 1] * u[i, j - 1]
            + mask[i, j] * u[i, j]
            + mask[ip1, j] * u[ip1, j]
            + mask[ip1, j - 1] * u[ip1, j - 1]
        )
        / 4.0
    )


@_numba_2D_grid_iterator
def _coriolis_v(
    i: int, j: int, ni: int, nj: int, v: np.array, mask: np.array, f: float
) -> float:
    jp1 = (j + 1) % nj
    return (
        f
        * (
            mask[i - 1, j] * v[i - 1, j]
            + mask[i, j] * v[i, j]
            + mask[i, jp1] * v[i, jp1]
            + mask[i - 1, jp1] * v[i - 1, jp1]
        )
        / 4.0
    )


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses. Periodic boundary conditions are applied.
"""


def zonal_pressure_gradient(state: State, params: Parameters) -> State:
    """Compute the zonal pressure gradient.

    Using centered differences in space.
    """
    result = _zonal_pressure_gradient(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,
        state.eta.grid.mask,
        params.g,
        state.u.grid.dx,
        state.u.grid.dy,
        state.eta.grid.dy,
    )
    return State(
        u=Variable(state.u.grid.mask * result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def meridional_pressure_gradient(state: State, params: Parameters) -> State:
    """Compute the meridional pressure gradient.

    Using centered differences in space.
    """
    result = _meridional_pressure_gradient(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,
        state.eta.grid.mask,
        params.g,
        state.v.grid.dx,
        state.v.grid.dy,
        state.eta.grid.dx,
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(state.u.grid.mask * result, state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def zonal_divergence(state: State, params: Parameters) -> State:
    """Compute the zonal divergence with centered differences in space."""
    result = _zonal_divergence(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.data,
        state.u.grid.mask,
        params.H,
        state.eta.grid.dx,
        state.eta.grid.dy,
        state.u.grid.dy,
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(state.eta.grid.mask * result, state.eta.grid),
    )


def meridional_divergence(state: State, params: Parameters) -> State:
    """Compute the meridional divergence with centered differences in space."""
    result = _meridional_divergence(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.data,
        state.v.grid.mask,
        params.H,
        state.eta.grid.dx,
        state.eta.grid.dy,
        state.v.grid.dx,
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(state.eta.grid.mask * result, state.eta.grid),
    )


def coriolis_u(state: State, params: Parameters) -> State:
    """Compute meridional acceleration due to Coriolis force.

    An arithmetic four point average of u onto the v-grid is performed.
    """
    result = _coriolis_u(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.data,
        state.u.grid.mask,
        params.f,
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(state.v.grid.mask * result, state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )


def coriolis_v(state: State, params: Parameters) -> State:
    """Compute the zonal acceleration due to the Coriolis force.

    An arithmetic four point average of v onto the u-grid is performed.
    """
    result = _coriolis_v(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.data,
        state.v.grid.mask,
        params.f,
    )
    return State(
        u=Variable(state.u.grid.mask * result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )


"""
Time integration schemes. To be used optional in integrator function.
"""


def euler_forward(rhs: deque, params: Parameters) -> State:
    """Compute increment using Euler forward scheme.

    Used for the time integration. One previous is
    state necessary. The function evaluation is performed before the
    state is passed to this function. Returns the increment
    dstate = next_state - current_state.
    """
    du = params.dt * rhs[-1].u.data
    dv = params.dt * rhs[-1].v.data
    deta = params.dt * rhs[-1].eta.data

    return State(
        u=Variable(du, rhs[-1].u.grid),
        v=Variable(dv, rhs[-1].v.grid),
        eta=Variable(deta, rhs[-1].eta.grid),
    )


def adams_bashforth2(rhs: deque, params: Parameters) -> State:
    """Compute increment using two-level Adams-Bashforth scheme.

    Used for the time integration.
    Two previous states necessary. If not provided, computational
    initial conditions are produced by forward euler. The function
    evaluations are performed before the state is passed to this function.
    Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 2:
        rhs.append(rhs[-1] + euler_forward(rhs, params))

    du = (params.dt / 2) * (3 * rhs[-1].u.data - rhs[-2].u.data)
    dv = (params.dt / 2) * (3 * rhs[-1].v.data - rhs[-2].v.data)
    deta = (params.dt / 2) * (3 * rhs[-1].eta.data - rhs[-2].eta.data)

    return State(
        u=Variable(du, rhs[-1].u.grid),
        v=Variable(dv, rhs[-1].v.grid),
        eta=Variable(deta, rhs[-1].eta.grid),
    )


def adams_bashforth3(rhs: deque, params: Parameters) -> State:
    """Compute increment using three-level Adams-Bashforth scheme.

    Used for the time integration.
    Three previous states necessary. If not provided, computational
    initial conditions are produced by adams_bashforth2. The function
    evaluations are performed before the state is passed to this
    function. Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 3:
        rhs.append(rhs[-1] + adams_bashforth2(rhs, params))

    du = (params.dt / 12) * (
        23 * rhs[-1].u.data - 16 * rhs[-2].u.data + 5 * rhs[-3].u.data
    )
    dv = (params.dt / 12) * (
        23 * rhs[-1].v.data - 16 * rhs[-2].v.data + 5 * rhs[-3].v.data
    )
    deta = (params.dt / 12) * (
        23 * rhs[-1].eta.data - 16 * rhs[-2].eta.data + 5 * rhs[-3].eta.data
    )

    return State(
        u=Variable(du, rhs[-1].u.grid),
        v=Variable(dv, rhs[-1].v.grid),
        eta=Variable(deta, rhs[-1].eta.grid),
    )


"""
Outmost functions defining the problem and what output should be computed.
"""


def linearised_SWE(state: State, params: Parameters) -> State:
    """Compute RHS of the linearised shallow water equations.

    The equations are evaluated on a C-grid. Output is a state type variable
    forming the right-hand-side needed for any time stepping scheme.
    """
    RHS_state = (
        (zonal_pressure_gradient(state, params) + coriolis_v(state, params))  # u_t
        + (  # v_t
            meridional_pressure_gradient(state, params) + coriolis_u(state, params)
        )
        + (  # eta_t
            zonal_divergence(state, params) + meridional_divergence(state, params)
        )
    )
    return RHS_state


# @jit
def integrator(
    state_0: State,
    params: Parameters,
    scheme: Callable[..., State] = adams_bashforth3,
    RHS: Callable[..., State] = linearised_SWE,
) -> State:
    """Integrate a system of differential equations.

    Only the last time step is returned.
    """
    if scheme == euler_forward:
        level = 1
    if scheme == adams_bashforth2:
        level = 2
    if scheme == adams_bashforth3:
        level = 3

    N = round((params.t_end - params.t_0) / params.dt)
    state = deque([state_0], maxlen=1)
    rhs = deque([], maxlen=level)

    for _ in range(N):
        rhs.append(RHS(state[-1], params))
        state.append(state[-1] + scheme(rhs, params))

    return state[-1]


"""
Very basic setup with only zonal flow for testing the functionality.
"""

if __name__ == "__main__":
    params = Parameters(t_end=7200.0)

    c_grid = StaggeredGrid.c_grid(
        func=Grid.regular_lat_lon,
        lon_start=0.0,
        lon_end=50.0,
        lat_start=0.0,
        lat_end=50.0,
        nx=51,
        ny=51,
    )

    u_0 = np.zeros(c_grid.u_grid.x.shape)
    v_0 = np.zeros(c_grid.v_grid.x.shape)
    eta_0 = c_grid.eta_grid.mask * (c_grid.eta_grid.x / 50) - 0.5

    init = State(
        u=Variable(u_0, c_grid.u_grid),
        v=Variable(v_0, c_grid.v_grid),
        eta=Variable(eta_0, c_grid.eta_grid),
    )

    start = timeit.default_timer()
    solution = integrator(init, params, scheme=adams_bashforth3)
    stop = timeit.default_timer()

    print("Runtime: ", stop - start, " s ")

    plt.figure()
    plt.pcolor(solution.eta.data)
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.pcolor(solution.u.data)
    plt.colorbar()
    plt.show()
