"""Time integration schemes.

To be used optional in integrate function.
"""
from typing import Callable, Iterator
from copy import copy

from .api import (
    TimeSteppingFunctionBase,
    RHSFunction,
    SolverBase,
    BorderDirection,
)
from .border import Border, BorderSplitter, RegularSplitMerger
from .util import add_time
from .datastructure import Domain, StateDeque, Parameter, State

from .kernel import (
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_j,
    coriolis_i,
    divergence_i,
    divergence_j,
    linear_combination,
    sum_states,
)


StateIncrement = State
CallableTSFunc = Callable[[StateDeque, float], StateIncrement]
TimeSteppingFunction = TimeSteppingFunctionBase[State, StateDeque]


def time_stepping_function(
    n_rhs: int, n_state: int
) -> Callable[[CallableTSFunc], TimeSteppingFunction]:
    """Decorate function by adding `n_rhs` and `n_state` attributes.

    Turns a :py:class:`TimeSteppingFunction` into a instance of :py:class:`TimeSteppingScheme`.
    """
    if n_state < 1 or n_rhs < 1:
        raise ValueError("n_rhs and n_state both needs to be larger than 0.")

    def decorator(func: CallableTSFunc) -> TimeSteppingFunction:
        ts_func_obj = TimeSteppingFunction(func, n_state, n_rhs)

        return ts_func_obj

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
        k: v.__class__(linear_combination((step,), (v.safe_data,)), v.grid, v.time)
        for k, v in rhs[-1].variables.items()
    }

    return rhs[-1].__class__(**inc)


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

    fac = tuple((step / 2) * n for n in (3.0, -1.0))

    inc = {
        k: v.__class__(
            linear_combination(
                fac,
                (rhs[-1].variables[k].safe_data, rhs[-2].variables[k].safe_data),
            ),
            rhs[-1].variables[k].grid,
            add_time(rhs[-1].variables[k].time, step / 2),
        )
        for k, v in rhs[-1].variables.items()
    }

    return rhs[-1].__class__(**inc)


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

    fac = tuple((step / 12) * n for n in (23.0, -16.0, 5.0))

    inc = {
        k: v.__class__(
            linear_combination(
                fac,
                (
                    rhs[-1].variables[k].safe_data,
                    rhs[-2].variables[k].safe_data,
                    rhs[-3].variables[k].safe_data,
                ),
            ),
            rhs[-1].variables[k].grid,
            add_time(rhs[-1].variables[k].time, step / 2),
        )
        for k, v in rhs[-1].variables.items()
    }

    return rhs[-1].__class__(**inc)


"""
Outmost functions defining the problem and what output should be computed.
"""


def linearised_SWE(state: State, params: Parameter) -> State:
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


def non_rotating_swe(state, params):
    """Compute RHS of the linearised shallow water equations without rotation.

    The equations are evaluated on a C-grid. Output is a state type variable
    forming the right-hand-side needed for any time stepping scheme.
    """
    rhs = (
        pressure_gradient_i(state, params)
        + pressure_gradient_j(state, params)
        + divergence_i(state, params)
        + divergence_j(state, params)
    )
    return rhs


def integrate(
    initial_state: State,
    params: Parameter,
    RHS: RHSFunction[State, Parameter],
    scheme: TimeSteppingFunction = adams_bashforth3,
    step: float = 1.0,  # time stepping in s
    time: float = 3600.0,  # end time
) -> Iterator[State]:
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

    try:
        state = StateDeque([initial_state], maxlen=scheme.n_state)
        rhs = StateDeque([], maxlen=scheme.n_rhs)
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
            v.time = add_time(old_state.variables[k].time, step)

        yield state[-1]


class Solver(SolverBase[Domain]):
    """Implement Solver class from API for use with any provided function."""

    def integrate(self, domain: Domain) -> Domain:
        """Integrate set of PDEs on domain.

        Arguments
        ---------
        domain: Domain
            Domain to integrate.

        Returns
        -------
        Domain object at the next time step.
        """
        inc = self._compute_increment(domain.state, domain.parameter)
        new = self._integrate(domain, inc)
        return new

    def get_border_width(self) -> int:
        """Retuns fixed border width."""
        return 2

    def integrate_border(  # type: ignore[override]
        self,
        domain: Domain,
        border: Border,
        neighbor_border: Border,
        direction: bool,
    ) -> Border:
        """Integrate set of PDEs on a border of a domian.

        Arguments
        ---------
        domain: Domain
            Domain of which the border is part of.
        border: Border
            The border to integrate.
        neighbor_border: Border
            Border of the neighboring domain.
        direction: bool
            True of border is the right border of the domain,
            False if it is the left border.

        Returns
        -------
        Border object at the next time step.
        """
        b_w = border.get_width()
        dim = border.dim
        assert domain.state.u.grid.x.shape[dim] >= 2 * b_w

        if direction:
            halo_slice = BorderDirection.RIGHT_HALO(b_w)  # type: ignore
        else:
            halo_slice = BorderDirection.LEFT_HALO(b_w)  # type: ignore

        halo_splitter = BorderSplitter(slice=halo_slice, axis=dim)
        # parts argument has no effect on merging logic
        list_merger = RegularSplitMerger(parts=0, dim=(dim,))
        area_of_interest_splitter = BorderSplitter(
            slice=BorderDirection.CENTER(b_w, b_w), axis=dim  # type: ignore
        )

        dom = domain.state.split(halo_splitter)[0]
        dom_param = domain.parameter.split(halo_splitter)[0]

        state_list = [dom, border.state, neighbor_border.state]
        param_list = [dom_param, border.parameter, neighbor_border.parameter]
        if not direction:
            state_list.reverse()
            param_list.reverse()
        merged_state = dom.merge(state_list, list_merger)
        merged_parameters = dom_param.merge(tuple(param_list), list_merger)
        inc = self._compute_increment(merged_state, merged_parameters)
        new = self._integrate(
            border,
            inc.split(area_of_interest_splitter)[0],
        )

        result = border.from_domain(
            new,
            width=border.get_width(),
            dim=dim,
        )
        result.id = domain.id
        return result

    def _compute_increment(self, state: State, parameter: Parameter) -> State:
        inc = self.rhs(state, parameter)
        return inc

    def _integrate(self, domain: Domain, inc: State) -> Domain:
        # shallow copy to avoid side effects
        history = copy(domain.history)
        history.append(inc)
        new = sum_states(
            (domain.state, self.ts_schema(history, self.step)), keep_time=0
        )
        # increment time
        new._increment_time(self.step)
        return Domain(
            state=new,
            history=history,
            parameter=domain.parameter,
            iteration=domain.increment_iteration(),
            id=domain.id,
        )
