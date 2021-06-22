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
    e_x: np.array  # weights transforming to curvilinear
    e_y: np.array  # weights transforming to curvilinear
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
        dx = np.append(
            dx, np.expand_dims(dx[0, :], axis=0), axis=self.dim_x
        )  # * self.ex
        dy = np.append(
            dy, np.expand_dims(dy[:, 0], axis=1), axis=self.dim_y
        )  # * self.e_y
        return dx, dy

    def __post_init__(self) -> None:
        """Set derived attributes of the grid."""
        self.dx, self.dy = self._compute_grid_spacing()
        self.len_x = self.x.shape[self.dim_x]
        self.len_y = self.x.shape[self.dim_y]


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
    g: float,
    dx: np.array,
    e_x: np.array,
) -> float:
    ip1 = (i + 1) % ni
    return -g * (eta[ip1, j] - eta[i, j]) / dx[i, j] / e_x[i, j]


@_numba_2D_grid_iterator
def _meridional_pressure_gradient(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.array,
    g: float,
    dy: np.array,
    e_y: np.array,
) -> float:
    jp1 = (j + 1) % nj
    return -g * (eta[i, jp1] - eta[i, j]) / dy[i, j] / e_y[i, j]


@_numba_2D_grid_iterator
def _zonal_divergence(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: np.array,
    mask: np.array,
    H: float,
    dx: np.array,
    e_x: np.array,
) -> float:
    return (
        -H
        * (mask[i, j] * u[i, j] - mask[i - 1, j] * u[i - 1, j])
        / dx[i - 1, j]
        / e_x[i - 1, j]
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
    dy: np.array,
    e_y: np.array,
) -> float:
    return (
        -H
        * (mask[i, j] * v[i, j] - mask[i, j - 1] * v[i, j - 1])
        / dy[i, j - 1]
        / e_y[i, j - 1]
    )


@_numba_2D_grid_iterator
def _coriolis_u(
    i: int, j: int, ni: int, nj: int, u: np.array, mask: np.array, f: float
) -> float:
    jp1 = (j + 1) % nj
    return (
        -f
        * (
            mask[i - 1, j] * u[i - 1, j]
            + mask[i, j] * u[i, j]
            + mask[i, jp1] * u[i, jp1]
            + mask[i - 1, jp1] * u[i - 1, jp1]
        )
        / 4.0
    )


@_numba_2D_grid_iterator
def _coriolis_v(
    i: int, j: int, ni: int, nj: int, v: np.array, mask: np.array, f: float
) -> float:
    ip1 = (i + 1) % ni
    return (
        f
        * 0.25
        * (
            mask[i, j - 1] * v[i, j - 1]
            + mask[i, j] * v[i, j]
            + mask[ip1, j] * v[ip1, j]
            + mask[ip1, j - 1] * v[ip1, j - 1]
        )
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
        params.g,
        state.eta.grid.dx,
        state.eta.grid.e_x,
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
        params.g,
        state.eta.grid.dy,
        state.eta.grid.e_y,
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
        state.u.grid.dx,
        state.eta.grid.e_x,
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
        state.u.grid.dy,
        state.eta.grid.e_y,
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(state.eta.grid.mask * result, state.eta.grid),
    )


def coriolis_u(state: State, params: Parameters) -> State:
    """Compute meridional acceleration due to Coriolis force.

    A arithmetic four point average of u onto the v-grid is performed.
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

    A arithmetic four point average of v onto the u-grid is performed.
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
    lat, lon = np.meshgrid(np.linspace(0, 50, 51), np.linspace(0, 50, 51))
    mask = np.ones(lon.shape)

    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0

    u_0 = np.zeros(lon.shape)
    v_0 = np.zeros(lon.shape)

    e_x = np.pi * params.r * np.cos(lat * np.pi / 180) / 180
    e_y = np.pi * np.ones(lon.shape) * params.r / 180
    eta_0 = mask * (np.copy(lon) / 50) - 0.5
    grid = Grid(lon, lat, mask, e_x, e_y)
    init = State(
        u=Variable(u_0, grid), v=Variable(v_0, grid), eta=Variable(eta_0, grid)
    )

    start = timeit.default_timer()
    solution = integrator(init, params, scheme=adams_bashforth3)
    stop = timeit.default_timer()

    print("Runtime: ", stop - start, " s ")
    """
    !!! without numba: ~5s, with numba: ~46s, numba gets confused,
    because it doesn't know the Dataclasses !!!

    --> The 2D grid loops are now jit-able, decreasing the measured
    (not tested) runtime.
    !!! without numba: ~8s, with numba: ~2s
    """

    plt.figure()
    plt.pcolor(solution.eta.data)
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.pcolor(solution.u.data)
    plt.colorbar()
    plt.show()
