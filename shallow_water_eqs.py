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


"""
Dataclasses building instances to hold parameters, dynamic variables
and their associated grids.
"""


@dataclass
class Parameters:
    """Class to organise all constant parameters."""

    f: float = 0.
    g: float = 9.81                 # gravitational force m/s^2
    beta: float = 2. / (24 * 3600)  # beta parameter 1/ms with f=f_0+beta *y
    H: float = 1000.                # reference depth in m
    rho_0:float = 1024.             # reference density in kg / m^3
    dt: float = 1.                  # time stepping in s
    t_0: float = 0.                 # starting time
    t_end: float = 3600.            # end time
    write: int = 20                 # how many states should be output


@dataclass
class Grid:
    """Grid informtation."""

    x: np.array                     # longitude on grid
    y: np.array                     # latitude on grid
    mask: np.array                  # ocean mask, 1 where ocean is, 0 where land is
    dim_x: int = 0                  # x dimension in numpy array
    dim_y: int = 1                  # y dimension in numpy array
    dx: int = field(init=False)     # grid spacing in x
    dy: int = field(init=False)     # grid spacing in y
    len_x: int = field(init=False)  # length of array in x dimension
    len_y: int = field(init=False)  # length of array in y dimension

    def __post_init__(self) -> None:
        """Set derived attributes of the grid."""
        self.dx = np.diff(self.x, axis=self.dim_x)[0, 0]
        self.dy = np.diff(self.y, axis=self.dim_y)[0, 0]
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
            raise ValueError(
                "Try to add variables defined on different grids."
            )
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
        if (
            not isinstance(other, type(self))
            or not isinstance(self, type(other))
        ):
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


@njit
def _iterate_over_grid_2D(
    loop_body: Callable[..., float], ni: int, nj: int, args: Tuple[Any]
) -> np.array:
    result = np.empty((ni, nj))
    for i in range(ni):
        for j in range(nj):
            result[i, j] = loop_body(*args, i, j, ni, nj)
    return result


@njit
def _zonal_pressure_gradient_loop_body(
    eta: np.array, g: float, dx: float, i: int, j: int, ni: int, nj: int
) -> float:
    ip1 = (i + 1) % ni
    return -g * (eta[ip1, j] - eta[i, j]) / dx


@njit
def _meridional_pressure_gradient_loop_body(
    eta: np.array, g: float, dy: float, i: int, j: int, ni: int, nj: int
) -> float:
    jp1 = (j + 1) % nj
    return -g * (eta[i, jp1] - eta[i, j]) / dy


@njit
def _zonal_divergence_loop_body(
    u: np.array, mask: np.array, H: float, dx: float, i: int, j: int, ni: int, nj: int
) -> float:
    return -H * (mask[i, j] * u[i, j] - mask[i - 1, j] * u[i - 1, j]) / dx


@njit
def _meridional_divergence_loop_body(
    v: np.array, mask: np.array, H: float, dy: float, i: int, j: int, ni: int, nj: int
) -> float:
    return -H * (mask[i, j] * v[i, j] - mask[i, j-1] * v[i, j - 1]) / dy


@njit
def _coriolis_u_loop_body(
    u: np.array, mask: np.array, f: float, i: int, j: int, ni: int, nj: int
) -> float:
    jp1 = (j + 1) % nj
    return -f * (mask[i - 1,j] * u[i - 1, j] + mask[i, j] * u[i, j] +
                 mask[i, jp1] * u[i, jp1] + mask[i - 1, jp1] * u[i - 1, jp1]) / 4.


@njit
def _coriolis_v_loop_body(
    v: np.array, mask: np.array, f: float, i: int, j: int, ni: int, nj: int
) -> float:
    ip1 = (i + 1) % ni
    return f * (mask[i, j - 1] * v[i, j - 1] + mask[i, j] * v[i, j] +
                mask[ip1, j] * v[ip1, j] + mask[ip1, j - 1] * v[ip1, j - 1]) / 4.


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses. Periodic boundary conditions are applied.
"""


def zonal_pressure_gradient(
    state: State, grid: Grid, params: Parameters
) -> State:
    """Compute the zonal pressure gradient.

    Using centered differences in space.
    """
    result = _iterate_over_grid_2D(
        loop_body=_zonal_pressure_gradient_loop_body,
        ni=state.eta.grid.len_x,
        nj=state.eta.grid.len_y,
        args=(state.eta.data, params.g, state.eta.grid.dx)
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid)
    )


def meridional_pressure_gradient(
    state: State, grid: Grid, params: Parameters
) -> State:
    """Compute the meridional pressure gradient.

    Using centered differences in space.
    """
    result = _iterate_over_grid_2D(
        loop_body=_meridional_pressure_gradient_loop_body,
        ni=state.eta.grid.len_x,
        nj=state.eta.grid.len_y,
        args=(state.eta.data, params.g, state.eta.grid.dy)
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid)
    )


def zonal_divergence(state: State, grid: Grid, params: Parameters) -> State:
    """Compute the zonal divergence with centered differences in space."""
    result = _iterate_over_grid_2D(
        loop_body=_zonal_divergence_loop_body,
        ni=state.u.grid.len_x,
        nj=state.u.grid.len_y,
        args=(state.u.data, state.u.grid.mask, params.H, state.u.grid.dx)
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(result, state.eta.grid)
    )


def meridional_divergence(
    state: State, grid: Grid, params: Parameters
) -> State:
    """Compute the meridional divergence with centered differences in space."""
    result = _iterate_over_grid_2D(
        loop_body=_meridional_divergence_loop_body,
        ni=state.v.grid.len_x,
        nj=state.v.grid.len_y,
        args=(state.v.data, state.v.grid.mask, params.H, state.u.grid.dy)
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(result, state.eta.grid)
    )


def coriolis_u(state: State, grid: Grid, params: Parameters) -> State:
    """Compute meridional acceleration due to Coriolis force.

    A arithmetic four point average of u onto the v-grid is performed.
    """
    result = _iterate_over_grid_2D(
        loop_body=_coriolis_u_loop_body,
        ni=state.u.grid.len_x,
        nj=state.u.grid.len_y,
        args=(state.u.data, state.u.grid.mask, params.f)
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid)
    )


def coriolis_v(state: State, grid: Grid, params: Parameters) -> State:
    """Compute the zonal acceleration due to the Coriolis force.

    A arithmetic four point average of v onto the u-grid is performed.
    """
    result = _iterate_over_grid_2D(
        loop_body=_coriolis_v_loop_body,
        ni=state.v.grid.len_x,
        nj=state.v.grid.len_y,
        args=(state.v.data, state.v.grid.mask, params.f)
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid)
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
        eta=Variable(deta, rhs[-1].eta.grid)
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
        eta=Variable(deta, rhs[-1].eta.grid)
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

    du = (
        (params.dt / 12)
        * (23 * rhs[-1].u.data - 16 * rhs[-2].u.data + 5 * rhs[-3].u.data)
    )
    dv = (
        (params.dt / 12)
        * (23 * rhs[-1].v.data - 16 * rhs[-2].v.data + 5 * rhs[-3].v.data)
    )
    deta = (
        (params.dt / 12)
        * (
            23 * rhs[-1].eta.data
            - 16 * rhs[-2].eta.data
            + 5 * rhs[-3].eta.data
        )
    )

    return State(
        u=Variable(du, rhs[-1].u.grid),
        v=Variable(dv, rhs[-1].v.grid),
        eta=Variable(deta, rhs[-1].eta.grid)
    )


"""
Outmost functions defining the problem and what output should be computed.
"""


def linearised_SWE(state: State, grid: Grid, params: Parameters) -> State:
    """Compute RHS of the linearised shallow water equations.

    The equations are evaluated on a C-grid. Output is a state type variable
    forming the right-hand-side needed for any time stepping scheme.
    """
    RHS_state = (
        (  # u_t
            zonal_pressure_gradient(state, grid, params)
            + coriolis_v(state, grid, params)
        )
        + (  # v_t
            meridional_pressure_gradient(state, grid, params)
            + coriolis_u(state, grid, params)
        )
        + (  # eta_t
            zonal_divergence(state, grid, params)
            + meridional_divergence(state, grid, params)
        )
    )
    return(RHS_state)


# @jit
def integrator(
    state_0: State, grid: Grid, params: Parameters,
    scheme: Callable[..., State] = adams_bashforth3,
    RHS: Callable[..., State] = linearised_SWE
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
        rhs.append(RHS(state[-1], grid, params))
        state.append(state[-1] + scheme(rhs, params))

    return(state[-1])


"""
Very basic setup with only zonal flow for testing the functionality.
"""

if __name__ == "__main__":
    params       = Parameters()
    y, x         = np.meshgrid(np.linspace(0, 50000, 51), np.linspace(0, 50000, 51))
    mask       = np.ones(x.shape)

    mask[0,:]  = 0.
    mask[-1,:] = 0.
    mask[:,0]  = 0.
    mask[:,-1] = 0.

    u_0 = np.zeros(x.shape)
    v_0 = np.zeros(x.shape)
    eta_0    = (np.copy(x)/ 50000) - 0.5
    grid   = Grid(x, y, mask)
    init = State(
        u=Variable(u_0, grid),
        v=Variable(v_0, grid),
        eta=Variable(eta_0, grid)
    )

    start = timeit.default_timer()
    solution = integrator(init, grid, params, scheme=adams_bashforth3)
    stop = timeit.default_timer()

    print('Runtime: ', stop - start, ' s ')
    '''
    !!! without numba: ~5s, with numba: ~46s, numba gets confused,
    because it doesn't know the Dataclasses !!!

    --> The 2D grid loops are now jit-able, decreasing the measured
    (not tested) runtime.
    !!! without numba: ~8s, with numba: ~2s
    '''

    plt.figure()
    plt.pcolor(solution.eta.data)
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.pcolor(solution.v.data)
    plt.colorbar()
    plt.show()
