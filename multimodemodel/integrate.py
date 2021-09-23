"""Time integration schemes.

To be used optional in integrate function.
"""

import sys
from collections import deque
import numpy as np
from dataclasses import dataclass, field
from .datastructure import Variable, Parameters, State
from typing import Callable, Generator
from itertools import starmap
from operator import mul
from .kernel import (
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_j,
    coriolis_i,
    divergence_i,
    divergence_j,
)

if sys.version_info < (3, 9):  # pragma: no cover
    # See https://docs.python.org/3/library/typing.html#typing.Deque
    from typing import Deque

    StateDeque = Deque[State]
else:
    StateDeque = deque[State]


StateIncrement = State

TimeSteppingFunction = Callable[[StateDeque, float], StateIncrement]


@dataclass
class TimeSteppingScheme:
    """Class to wrap time stepping functions.

    This class adds `n_rhs` and `n_states` attributes to a time stepping function.
    It also implements the `__call__` method, which passes the args to the time
    stepping function.

    Parameters
    ----------
    _func: TimeSteppingFunction
      Time stepping function to wrap
    n_rhs: int
      Number of previous rhs evaluations required by the scheme
    n_state: int
      Number of previous states required by the scheme
    """

    _func: TimeSteppingFunction
    n_rhs: int
    n_state: int
    __name__: str = field(init=False)

    def __post_init__(self):
        """Set derived attributes."""
        self.__name__ = self._func.__name__
        self.__doc__ = self._func.__doc__

    def __call__(self, rhs: StateDeque, step: float) -> StateIncrement:
        """Pass arguments to wrapped time stepping function and return result."""
        return self._func(rhs, step)


def seconds_to_timedelta64(dt: float) -> np.timedelta64:
    """Convert timestep in seconds to a numpy timedelta64 object.

    `dt` will be rounded to an integer at nanosecond precision.

    Parameters
    ----------
    dt : float
        Time span in seconds.

    Returns
    -------
    numpy.timedelta64
        Timedelta object with nanosecond precision.
    """
    return np.timedelta64(round(1e9 * dt), "ns")


def time_stepping_function(
    n_rhs: int, n_state: int
) -> Callable[[TimeSteppingFunction], TimeSteppingScheme]:
    """Decorate function by adding `n_rhs` and `n_state` attributes.

    Turns a :py:class:`TimeSteppingFunction` into a instance of :py:class:`TimeSteppingScheme`.
    """
    if n_state < 1 or n_rhs < 1:
        raise ValueError("n_rhs and n_state both needs to be larger than 0.")

    def decorator(func: TimeSteppingFunction) -> TimeSteppingScheme:
        return TimeSteppingScheme(func, n_rhs, n_state)

    return decorator


@time_stepping_function(n_rhs=1, n_state=1)
def euler_forward(rhs: StateDeque, step: float) -> StateIncrement:
    """Compute increment using Euler forward scheme.

    Used for the time integration. One previous state is necessary.
    The time stamp of the increment is the same as for the current state.

    Parameters
    ---------
    rhs : StateDeque
        Deque object containing the previous evaluations of the right-hand-side
        terms.
    step : float
        Time step size.

    Attributes
    ----------
    n_rhs: int
      Number of previous rhs evaluations required by the scheme
    n_state: int
      Number of previous states required by the scheme

    Returns
    -------
    StateIncrement
        The increment of the state from one time step to the next, i.e. next_state - current_state.
    """
    inc = {
        k: Variable(step * v.safe_data, v.grid, v.time)
        for k, v in rhs[-1].variables.items()
    }

    return StateIncrement(**inc)


@time_stepping_function(n_rhs=2, n_state=1)
def adams_bashforth2(rhs: StateDeque, step: float) -> StateIncrement:
    """Compute increment using two-level Adams-Bashforth scheme.

    Used for the time integration. Two previous states are required.
    If less are provided, :py:func:`euler_forward` scheme is used instead.

    Parameters
    ---------
    rhs : StateDeque
        Deque object containing the previous evaluations of the right-hand-side
        terms.
    step : float
        Time step size.

    Attributes
    ----------
    n_rhs: int
      Number of previous rhs evaluations required by the scheme
    n_state: int
      Number of previous states required by the scheme

    Returns
    -------
    StateIncrement
        The increment of the state from one time step to the next, i.e. next_state - current_state.
    """
    if len(rhs) < 2:
        return euler_forward(rhs, step)

    dt = seconds_to_timedelta64(step)

    coef = (1.5, -0.5)
    inc = {
        k: Variable(
            step
            * sum(starmap(mul, zip(coef, (r.variables[k].safe_data for r in reversed(rhs))))),  # type: ignore
            rhs[-1].variables[k].grid,
            rhs[-1].variables[k].time + dt / 2,
        )
        for k in rhs[-1].variables.keys()
    }

    return StateIncrement(**inc)


@time_stepping_function(n_rhs=3, n_state=1)
def adams_bashforth3(rhs: StateDeque, step: float) -> StateIncrement:
    """Compute increment using three-level Adams-Bashforth scheme.

    Used for the time integration. Three previous states are necessary.
    If less are provided, the scheme :py:func:`adams_bashforth2` is used instead.

    Parameters
    ---------
    rhs : StateDeque
        Deque object containing the previous evaluations of the right-hand-side
        terms.
    step : float
        Time step size.

    Attributes
    ----------
    n_rhs: int
      Number of previous rhs evaluations required by the scheme
    n_state: int
      Number of previous states required by the scheme

    Returns
    -------
    StateIncrement
        The increment, i.e. next_state - current_state.
    """
    if len(rhs) < 3:
        return adams_bashforth2(rhs, step)

    dt = seconds_to_timedelta64(step)

    coef = (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0)
    inc = {
        k: Variable(
            step
            * sum(starmap(mul, zip(coef, (r.variables[k].safe_data for r in reversed(rhs))))),  # type: ignore
            rhs[-1].variables[k].grid,
            rhs[-1].variables[k].time + dt / 2,
        )
        for k in rhs[-1].variables.keys()
    }

    return StateIncrement(**inc)


"""
Outmost functions defining the problem and what output should be computed.
"""


def linearised_SWE(state: State, params: Parameters) -> State:
    """Compute RHS of the linearised shallow water equations.

    The equations are evaluated on a C-grid. Output is a state object
    forming the right-hand-side needed for any time stepping scheme. These terms
    are evaluated:

    - :py:func:`pressure_gradient_i`
    - :py:func:`pressure_gradient_j`
    - :py:func:`coriolis_i`
    - :py:func:`coriolis_j`
    - :py:func:`divergence_i`
    - :py:func:`divergence_j`

    Parameters
    ----------
    state : State
      Present state of the system.
    params : Parameters
      Parameters of the system.

    Returns
    -------
    State
      Contains the sum of all tendency terms for all prognostic variables.
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
    RHS: Callable[[State, Parameters], State],
    scheme: TimeSteppingScheme = adams_bashforth3,
    step: float = 1.0,  # time stepping in s
    time: float = 3600.0,  # end time
) -> Generator[State, None, None]:
    """Integrate a system of differential equations.

    Generator which can be iterated over to produce new time steps.

    Parameters
    ----------
    initial_state: State
      Initial conditions of the prognostic variables.
    params: Parameters
      Parameters of the governing equations.
    RHS: Callable[[State, Parameters], State]
      Function defining the set of equations to integrate.
    scheme: TimeSteppingScheme, default=adams_bashforth3
      Time integration scheme to use
    step: float, default=1.0
      Length of time step
    time: float, default=3600.0
      Integration time. Will be reduced to the next integral multiple of `step`

    Yields
    ------
    State
      Next time step.

    Example
    -------
    Integrate the set of linear shallow water equations:

    >>> integrator = integrate(init_state, params, linearised_swe, step=1., time=10e4)
    >>> for next_state in integrator:
    ...    pass
    """
    N = int(time // step)
    dt = seconds_to_timedelta64(step)

    try:
        state = deque([initial_state], maxlen=scheme.n_state)
        rhs: StateDeque = deque([], maxlen=scheme.n_rhs)
    except AttributeError:
        raise AttributeError(
            f"Either n_state or n_rhs attribute missing for {scheme.__name__}. "
            "Consider to declare the function with time_stepping_function "
            "decorator."
        )

    for _ in range(N):
        old_state = state[-1]

        rhs.append(RHS(state[-1], params))
        state.append(state[-1] + scheme(rhs, step))

        for k, v in state[-1].variables.items():
            v.time = old_state.variables[k].time + dt

        yield state[-1]
