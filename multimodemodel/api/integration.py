"""Provide the basic API for time integration."""
from abc import abstractmethod
from typing import Generic, Callable
from .core import StateType, ParameterType, StateDequeType, DomainType
from .border import BorderType
from .typing import DType


RHSFunction = Callable[[StateType, ParameterType], StateType]


class TimeSteppingFunctionBase(Generic[StateType, StateDequeType]):
    """Base class for time stepping function wrapper.

    This is a generic class that can be specialized w.r.t. the underlying
    State and Parameter type.
    """

    __slots__ = ["_func", "n_state", "n_rhs"]

    def __init__(
        self,
        func: Callable[[StateDequeType, DType], StateType],
        n_state: int,
        n_rhs: int,
    ):
        """Wrap a time stepping function into an object and attach meta data."""
        self._func = func
        self.n_state = n_state
        self.n_rhs = n_rhs

    def __call__(
        self,
        state_deque: StateDequeType,
        step: DType,
    ) -> StateType:
        """Evaluate wrapped time stepping function."""
        return self._func(state_deque, step)

    @property
    def __name__(self) -> str:
        """Return name of the wrapped function."""
        return self._func.__name__


class SolverBase(Generic[DomainType]):
    """Base class for Solver.

    A Solver wraps methods required to solve given problem.
    """

    ts_schema: TimeSteppingFunctionBase
    step: DType

    __slots__ = ["step", "_rhs", "ts_schema"]

    def __init__(
        self,
        rhs: RHSFunction[StateType, ParameterType],
        ts_schema: TimeSteppingFunctionBase,
        step: float = 1,
    ):
        """Initialize solver object providing functions to compute next iterations.

        Arguments
        ---------
        rhs: (State, Parameters]) -> State
            Function that takes State and Parameters and returns State.
            It is used to compute next iteration.

        ts_schema: TimeSteppingFunction
            Integration schema like euler_forward or adams_bashforth3

        step: float
            Length of time step in the integration process.
        """
        self.step = step
        self._rhs = rhs
        self.ts_schema = ts_schema

    def rhs(self, state: StateType, parameter) -> StateType:
        """Evaluate right-hand-side function."""
        return self._rhs(state, parameter)

    @abstractmethod
    def integrate(self, domain: DomainType) -> DomainType:
        """Compute next iteration."""
        ...

    @abstractmethod
    def integrate_border(
        self,
        domain: DomainType,
        border: BorderType,
        neighbor_border: BorderType,
        direction: bool,
    ) -> DomainType:
        """Integrate set of PDEs on a border of a domian."""
        ...

    @abstractmethod
    def get_border_width(self) -> int:
        """Provide minimal required border width."""
        ...
