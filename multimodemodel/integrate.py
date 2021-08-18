"""Time integration schemes.

To be used optional in integrate function.
"""

from collections import deque
from .datastructure import Variable, Parameters, State
from typing import Callable
from .kernel import (
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_j,
    coriolis_i,
    divergence_i,
    divergence_j,
)


def euler_forward(rhs: deque, params: Parameters) -> State:
    """Compute increment using Euler forward scheme.

    Used for the time integration. One previous state
    is necessary. The function evaluation is performed before the
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
    Two previous states are necessary. If not provided, the forward euler
    scheme is used. Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 2:
        return euler_forward(rhs, params)

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

    Used for the time integration. Three previous states necessary.
    If not provided, the adams_bashforth2 scheme is used instead.
    Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 3:
        return adams_bashforth2(rhs, params)

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
        (pressure_gradient_i(state, params) + coriolis_i(state, params))  # u_t
        + (pressure_gradient_j(state, params) + coriolis_j(state, params))  # v_t
        + (divergence_i(state, params) + divergence_j(state, params))  # eta_t
    )
    return RHS_state


def integrate(
    state_0: State,
    params: Parameters,
    scheme: Callable[..., State] = adams_bashforth3,
    RHS: Callable[..., State] = linearised_SWE,
):
    """Integrate a system of differential equations.

    Only the last time step is returned.
    """
    if scheme is euler_forward:
        level = 1
    elif scheme is adams_bashforth2:
        level = 2
    elif scheme is adams_bashforth3:
        level = 3
    else:
        raise ValueError("Unsupported scheme provided.")

    N = int((params.t_end - params.t_0) // params.dt)
    state = deque([state_0], maxlen=1)
    rhs = deque([], maxlen=level)

    for _ in range(N):
        rhs.append(RHS(state[-1], params))
        state.append(state[-1] + scheme(rhs, params))
        yield state[-1]