import numpy as np
from numba import jit
from dataclasses import dataclass

"""
This script is the initial attempt to formulate the linearised shallow water equation as python functions.
"""

@dataclass
class parameters:
    """
    Class to organise all constant parameters.
    """
    g: float    = 9.81          #gravitational force m/s^2
    beta: float = 2./(24*3600)  #beta parameter 1/ms with f = f_0 + beta *y
    H: float    = 1000.         #reference depth in m
    dt: float   = 60.           #time stepping in s
    dim_x: int  = 0             #x dimension in numpy arrays
    dim_x: int  = 1             #y dimension in numpy arrays
    len_x: int  = 50            #size of the grid in x dimension
    len_y: int  = 50            #size of the grid in y dimension

@dataclass
class state:
    """
    Class that combines the dynamical variables u,v, eta into one state object.
    """
    x: np.array                    #zonal coordinates
    y: np.array                    #meridional coordinates
    u: np.array                    #zonal velocity
    v: np.array                    #meridional velocity
    eta: np.array                  #surface displacement

def zonal_pressure_gradient(state: state, i: int, j: int) -> float:
    """
    Computes the zonal pressure gradient with centered differencs in the longitude.
    """
    d_eta_dx = (state.eta[i+1,j] - state.eta[i,j]) / (state.x[i+1,j] - state.x[i,j])
    return(d_eta_dx)

def meridional_pressure_gradient(state: state, i: int, j: int) -> float:
    """
    Computes the meridional pressure gradient with centered differencs in the latitude.
    """
    d_eta_dy = (state.eta[i,j+1] - state.eta[i,j]) / (state.y[i,j+1] - state.y[i,j])
    return(d_eta_dx)

def horizontal_divergence(state: state, i: int, j: int) -> float:
    """
    Computes the horizontal divergence with centered differencs space.
    """
    divergence = ((state.v[i,j] - state.v[i,j-1]) / (state.y[i,j] - state.y[i,j-1]) -
                  (state.u[i,j] - state.u[i-1,j]) / (state.x[i,j] - state.x[i-1,j]))
    return(divergence)

def adams_bashford3(state_0: state, state_1: state, state_2: state, RHS: function, params: parameters) -> state:
    """
    Three-level Adams-Bashford scheme used for the time integration. Three previous states necessary
    """
    state_3 = state_2 + (params.dt/12)*(23*RHS(state_2) - 16*RHS(state_1) + 5*RHS(state_0))
    return(state_3)

def computational_initial_states(state_0: state, RHS: function, params: parameters) -> state, state:
    """
    The initial state is used to get two computational initial conditions necessary for the three-level AB.
    """
    state_1 = state_0 + params.dt*RHS(state_0)
    state_2 = state_1 + (params.dt/2)*(3*RHS(state_1)-RHS(state_0))
    return(state_1, state_2)

def linearised_SWE(state: state, params: parameters) -> state:
    """
    Simple set of linearised shallow water equations. The equations are evaluated on a C-grid.
    """
    u   = np.zeros((len_x,len_y))
    v   = np.copy(u)
    eta = np.copy(u)

    for i in range(0, params.len_x):
        for j in range(0, params.len_y):
            print("do stuff")
