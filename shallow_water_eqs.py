import numpy as np
import timeit
from typing import Callable, Tuple
from numba import jit
from dataclasses import dataclass, field
from matplotlib import pyplot as plt




"""
This script is the initial attempt to formulate the linearised shallow water equation as python functions.
"""

@dataclass
class Parameters:
    """
    Class to organise all constant parameters.
    """
    f: float     = 0.
    g: float     = 9.81          #gravitational force m/s^2
    beta: float  = 2./(24*3600)  #beta parameter 1/ms with f = f_0 + beta *y
    H: float     = 1000.         #reference depth in m
    dt: float    = 8.           #time stepping in s
    t_0: float   = 0.            #starting time
    t_end: float = 3600.         #end time
    write: int   = 20            #how many states should be written to output


@dataclass
class Grid:
    x: np.array                         #longitude on grid
    y: np.array                         #latitude on grid
    dim_x: int   = 0                    #x dimension in numpy array
    dim_y: int   = 1                    #y dimension in numpy array
    dx: int      = field(init = False)  #grid spacing in x
    dy: int      = field(init = False)  #grid spacing in y
    len_x: int   = field(init = False)  #length of array in x dimension
    len_y: int   = field(init = False)  #length of array in y dimension
    index_x: int = field(init = False)  #list of indices in x dimension
    index_y: int = field(init = False)  #list of indices in y dimension
    wrap_x: int  = field(init = False)  #wrapper list to produce periodic boundary conditions
    wrap_y: int  = field(init = False)  #wrapper list to produce periodic boundary conditions

    def __post_init__(self) -> None:
        """
        Second __init__ function deriving parameters of the grid,
        necessary for looping with periodic boundary conditions.
        """
        self.dx      = abs(self.x[1,0]-self.x[0,0])
        self.dy      = abs(self.y[0,1]-self.y[0,0])
        self.len_x   = self.x.shape[self.dim_x]
        self.len_y   = self.x.shape[self.dim_y]
        self.index_x = list(range(0,self.len_x))
        self.index_y = list(range(0,self.len_y))
        self.wrap_x  = self.index_x + [0]
        self.wrap_y  = self.index_y + [0]



@dataclass
class State:
    """
    Class that combines the dynamical variables u,v, eta into one state object.
    """
    u: np.array                    #zonal velocity
    v: np.array                    #meridional velocity
    eta: np.array                  #surface displacement

#@jit
def zonal_pressure_gradient(state: State, grid: Grid) -> np.array:
    """
    Computes the zonal pressure gradient with centered differencs in the longitude.
    Periodic boundary conditions are applied with a wrapper index.
    """
    d_eta_dx = np.zeros(state.eta.shape)

    for i in grid.index_x:
        for j in grid.index_y:
            d_eta_dx[i,j] = (state.eta[grid.wrap_x[i+1],j] - state.eta[i,j]) / grid.dx

    return(d_eta_dx)

#@jit
def meridional_pressure_gradient(state: State, grid: Grid) -> np.array:
    """
    Computes the meridional pressure gradient with centered differencs in the latitude.
    Periodic boundary conditions are applied with a wrapper index.
    """
    d_eta_dy = np.zeros(state.eta.shape)

    for i in grid.index_x:
        for j in grid.index_y:
            d_eta_dy[i,j] = (state.eta[i,grid.wrap_y[j+1]] - state.eta[i,j]) / grid.dy

    return(d_eta_dy)

#@jit
def horizontal_divergence(state: State, grid: Grid) -> np.array:
    """
    Computes the horizontal divergence with centered differences in space.
    """
    divergence = np.zeros(state.v.shape)

    for i in grid.index_x:
        for j in grid.index_y:
            divergence[i,j] = params.H*((state.v[i,j] - state.v[i,j-1]) / grid.dy +
                             (state.u[i,j] - state.u[i-1,j]) / grid.dx )

    return(divergence)

#@jit
def v_on_u_grid(state: State, grid: Grid) -> np.array:
    """
    Computes the four-point averaged v used for evaluations on the u grid.
    Periodic boundary conditions are applied with a wrapper index.
    """
    v = np.zeros(state.v.shape)

    for i in grid.index_x:
        for j in grid.index_y:
            v[i,j] = (state.v[i,j-1]+state.v[i,j]+state.v[grid.wrap_x[i+1],j] +
                      state.v[grid.wrap_x[i+1],j-1])/4

    return(v)

#@jit
def u_on_v_grid(state: State, grid: Grid) -> np.array:
    """
    Computes the four-point averaged u used for evaluations on the v grid.
    Periodic boundary conditions are applied with a wrapper index.
    """
    u = np.zeros(state.u.shape)

    for i in grid.index_x:
        for j in grid.index_y:
            u[i,j] = (state.u[i-1,j]+state.u[i,j]+state.u[i,grid.wrap_y[j+1]] +
                      state.u[i-1,grid.wrap_y[j+1]])/4
    return(u)

#@jit
def adams_bashford3(RHS_state_0: State, RHS_state_1: State, state_2: State, RHS: Callable[[State, Parameters],State], grid: Grid, params: Parameters) -> Tuple[State,State]:
    """
    Three-level Adams-Bashford scheme used for the time integration. Three previous states necessary.
    The function evaluation of the first two states are already known and are passed directly.
    """
    RHS_state_2 = RHS(state_2, grid, params)
    u_3         = state_2.u + (params.dt/12)*(23*RHS_state_2.u - 16*RHS_state_1.u + 5*RHS_state_0.u)
    v_3         = state_2.v + (params.dt/12)*(23*RHS_state_2.v - 16*RHS_state_1.v + 5*RHS_state_0.v)
    eta_3       = state_2.eta + (params.dt/12)*(23*RHS_state_2.eta - 16*RHS_state_1.eta + 5*RHS_state_0.eta)
    state_3     = State(u = u_3, v = v_3, eta = eta_3)
    return(state_3, RHS_state_2)

#@jit
def computational_initial_states(state_0: State, RHS: Callable[[State, Parameters],State], grid: Grid, params: Parameters) -> Tuple[State,State,State,State]:
    """
    The initial state is used to get two computational initial states necessary for the three-level AB.
    The two initial states are passed together with the two function evaluations to reduce computational
    effort.
    """
    #Euler scheme for the first computational initial state
    RHS_state_0 = RHS(state_0, grid, params)
    u_1         = state_0.u + params.dt*RHS_state_0.u
    v_1         = state_0.v + params.dt*RHS_state_0.v
    eta_1       = state_0.eta + params.dt*RHS_state_0.eta
    state_1     = State(u = u_1, v = v_1, eta = eta_1)
    #AB scheme for the second computational initial state
    RHS_state_1 = RHS(state_1, grid, params)
    u_2         = state_1.u + (params.dt/2)*(3*RHS_state_1.u-RHS_state_0.u)
    v_2         = state_1.v + (params.dt/2)*(3*RHS_state_1.v-RHS_state_0.v)
    eta_2       = state_1.eta + (params.dt/2)*(3*RHS_state_1.eta-RHS_state_0.eta)
    state_2     = State(u = u_2, v = v_2, eta = eta_2)
    return(state_1, state_2, RHS_state_0, RHS_state_1)

#@jit
def linearised_SWE(state: State, grid: Grid, params: Parameters) -> State:
    """
    Simple set of linearised shallow water equations. The equations are evaluated on a C-grid.
    Output is a state type variable collecting u_t, v_t, eta_t, forming the right-hand-side needed for any time stepping scheme.
    """

    u_t    = params.f*v_on_u_grid(state, grid) - params.g*zonal_pressure_gradient(state, grid)
    v_t    = - params.f*u_on_v_grid(state, grid) - params.g*meridional_pressure_gradient(state, grid)
    eta_t  = - params.H*horizontal_divergence(state, grid)

    RHS_state = State(u = u_t, v = v_t, eta = eta_t)
    return(RHS_state)

#@jit
def integrator(state_0: State, RHS: Callable[[State, Parameters],State], grid: Grid, params: Parameters) -> State:
    """
    Function processing time integration.
    """
    N           = round((params.t_end - params.t_0)/params.dt)
    solution    = [state_0]
    state_1, state_2, RHS_state_0, RHS_state_1 = computational_initial_states(state_0, RHS, grid, params)

    for k in range(0,N):
        state_3, RHS_state_2 = adams_bashford3(RHS_state_0,RHS_state_1,state_2, RHS, grid, params)
        RHS_state_0 = RHS_state_1
        RHS_state_1 = RHS_state_2
        state_2     = state_3

    return(state_3)

"""
Very basic setup with only zonal flow for testing the functionality.
"""

params  = Parameters()
y, x    = np.meshgrid(np.linspace(0,50000,51),np.linspace(0,50000,51))
u_0     = 0.05*np.ones(x.shape)
v_0     = np.zeros(x.shape)
eta_0   = np.zeros(x.shape)
init    = State(u = u_0, v = v_0, eta = eta_0)
grid    = Grid(x, y)

start = timeit.default_timer()
solution= integrator(init, linearised_SWE, grid, params)
stop = timeit.default_timer()

print('Runtime: ', stop - start, ' s ')
'''!!! without numba: ~5s, with numba: ~46s, numba gets confused, because it doesn't know the Dataclasses !!!'''

plt.figure()
plt.pcolor(solution.u)
plt.colorbar()
plt.show()
plt.figure()
plt.pcolor(solution.v)
plt.colorbar()
plt.show()
