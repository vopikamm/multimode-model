"""Time integration schemes.

To be used optional in integrate function.
"""

from collections import deque
from .datastructure import Variable, Parameters, State
from typing import Callable, Generator, NewType
from functools import wraps
from .kernel import (
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_j,
    coriolis_i,
    divergence_i,
    divergence_j,
)

StateIncrement = NewType("StateIncrement", State)
TimeSteppingFunction = Callable[[deque, Parameters, float], StateIncrement]


def time_stepping_function(
    n_rhs: int, n_state: int
) -> Callable[[TimeSteppingFunction], TimeSteppingFunction]:
    """Decorate function by adding n_rhs and n_state attributes."""
    if n_state < 1 or n_rhs < 1:
        raise ValueError("n_rhs and n_state both needs to be larger than 0.")

    def decorator(func: TimeSteppingFunction) -> TimeSteppingFunction:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.n_rhs = n_rhs
        wrapper.n_state = n_state
        return wrapper

    return decorator


@time_stepping_function(n_rhs=1, n_state=1)
def euler_forward(rhs: deque[State], params: Parameters, step: float) -> StateIncrement:
    """Compute increment using Euler forward scheme.

    Used for the time integration. One previous state
    is necessary. The function evaluation is performed before the
    state is passed to this function. Returns the increment
    dstate = next_state - current_state.
    """
    du = step * rhs[-1].u.safe_data
    dv = step * rhs[-1].v.safe_data
    deta = step * rhs[-1].eta.safe_data

    return StateIncrement(
        State(
            u=Variable(du, rhs[-1].u.grid),
            v=Variable(dv, rhs[-1].v.grid),
            eta=Variable(deta, rhs[-1].eta.grid),
        )
    )


@time_stepping_function(n_rhs=2, n_state=1)
def adams_bashforth2(
    rhs: deque[State], params: Parameters, step: float
) -> StateIncrement:
    """Compute increment using two-level Adams-Bashforth scheme.

    Used for the time integration.
    Two previous states are necessary. If not provided, the forward euler
    scheme is used. Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 2:
        return euler_forward(rhs, params, step)

    du = (step / 2) * (3 * rhs[-1].u.safe_data - rhs[-2].u.safe_data)
    dv = (step / 2) * (3 * rhs[-1].v.safe_data - rhs[-2].v.safe_data)
    deta = (step / 2) * (3 * rhs[-1].eta.safe_data - rhs[-2].eta.safe_data)

    return StateIncrement(
        State(
            u=Variable(du, rhs[-1].u.grid),
            v=Variable(dv, rhs[-1].v.grid),
            eta=Variable(deta, rhs[-1].eta.grid),
        )
    )


@time_stepping_function(n_rhs=3, n_state=1)
def adams_bashforth3(
    rhs: deque[State], params: Parameters, step: float
) -> StateIncrement:
    """Compute increment using three-level Adams-Bashforth scheme.

    Used for the time integration. Three previous states necessary.
    If not provided, the adams_bashforth2 scheme is used instead.
    Returns the increment dstate = next_state - current_state.
    """
    if len(rhs) < 3:
        return adams_bashforth2(rhs, params, step)

    du = (step / 12) * (
        23 * rhs[-1].u.safe_data - 16 * rhs[-2].u.safe_data + 5 * rhs[-3].u.safe_data
    )
    dv = (step / 12) * (
        23 * rhs[-1].v.safe_data - 16 * rhs[-2].v.safe_data + 5 * rhs[-3].v.safe_data
    )
    deta = (step / 12) * (
        23 * rhs[-1].eta.safe_data
        - 16 * rhs[-2].eta.safe_data
        + 5 * rhs[-3].eta.safe_data
    )

    return StateIncrement(
        State(
            u=Variable(du, rhs[-1].u.grid),
            v=Variable(dv, rhs[-1].v.grid),
            eta=Variable(deta, rhs[-1].eta.grid),
        )
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
    initial_state: State,
    params: Parameters,
    RHS: Callable[..., State],
    scheme: TimeSteppingFunction = adams_bashforth3,
    step: float = 1.0,  # time stepping in s
    time: float = 3600.0,  # end time
) -> Generator[State, None, None]:
    """Integrate a system of differential equations.

    The function returns a generator which can be iterated over.

    Arguments
    ---------
    initial_state: State
      Initial conditions of the prognostic variables
    params: Parameters
      Parameters of the governing equations
    RHS: Callable[..., State]
      Function defining the set of equations to integrate
    scheme: Callable[..., State] = adams_bashforth3
      Time integration scheme to use
    step: float = 1.0
      Length of time step
    time: float = 3600.0
      Integration time. Will be reduced to the next integral multiple of `step`

    Example
    -------
    Integrate the set of linear shallow water equations:
    ```python
    integrator = integrate(init_state, params, linearised_swe, step=1., time=10e4)

    for next_state in integrator:
        pass
    ```
    """
    N = int(time // step)

    try:
        state = deque([initial_state], maxlen=scheme.n_state)
        rhs = deque([], maxlen=scheme.n_rhs)
    except AttributeError:
        raise AttributeError(
            f"Either n_state or n_rhs attribute missing for {scheme.__name__}. "
            "Consider to declare the function with time_stepping_function "
            "decorator."
        )

    for _ in range(N):
        rhs.append(RHS(state[-1], params))
        state.append(state[-1] + scheme(rhs, params, step))
        yield state[-1]
