"""Implementation of domain split API."""
import sys
from enum import Enum
from functools import wraps, lru_cache
from .config import config
from .integrate import TimeSteppingFunction
from .domain_split_API import (
    Domain,
    Border,
    Solver,
    SplitVisitor,
    MergeVisitor,
    Tailor,
    Splitable,
)
from .datastructure import State, Variable, np, Parameters
from .grid import Grid, StaggeredGrid
from dask.distributed import Client, Future
from redis import Redis
from struct import pack
from collections import deque
from typing import Callable, Optional, Sequence, Tuple, Dict
from dataclasses import dataclass, fields
from copy import copy, deepcopy

if sys.version_info < (3, 9):
    from typing import Deque
else:
    Deque = deque


def _ensure_others_is_tuple(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "others" in kwargs and type(kwargs["others"]) is not Tuple:
            kwargs["others"] = tuple(kwargs["others"])
        elif type(args[0]) is not Tuple:
            args = tuple(a if i != 0 else tuple(a) for i, a in enumerate(args))
        return func(self, *args, **kwargs)

    return wrapper


class RegularSplitMerger(SplitVisitor, MergeVisitor):
    """Implements splitting and merging into regular grid."""

    __slots__ = ["dim", "_parts"]

    def __init__(self, parts: int, dim: Tuple[int]):
        """Initialize class instance."""
        self._parts = parts
        self.dim = dim

    def split_array(self, array: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        Tuple[np.ndarray, ...]
        """
        return np.array_split(array, indices_or_sections=self.parts, axis=self.dim[0])

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        return np.concatenate(arrays, axis=self.dim[0])

    def __hash__(self):
        """Return hash based on number of parts and dimension."""
        return hash((self.parts, self.dim))

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return self._parts


class BorderDirection(Enum):
    """Enumerator of border possible directions."""

    LEFT = lambda w: slice(None, w)  # noqa: E731
    LEFT_HALO = lambda w: slice(w, 2 * w)  # noqa: E731
    RIGHT = lambda w: slice(-w, None)  # noqa: E731
    RIGHT_HALO = lambda w: slice(-2 * w, -w)  # noqa: E731
    CENTER = lambda w1, w2: slice(w1, -w2)  # noqa: E731


class BorderSplitter(SplitVisitor):
    """Implements splitting off stripes of a DomainState along a dimension."""

    __slots__ = ["_axis", "_slice"]

    def __init__(self, slice: slice, axis: int):
        """Initialize class instance."""
        self._axis = axis
        self._slice = slice

    def split_array(self, array: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split array.

        Parameter
        ---------
        array: np.ndarray
          Array to split.

        Returns
        -------
        Tuple[np.ndarray, ...]
        """
        slices = array.ndim * [slice(None)]
        slices[self._axis] = self._slice
        return (array[tuple(slices)],)

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return 1

    def __hash__(self):
        """Return hash based on axis and slice object."""
        return hash((self._axis, self._slice.start, self._slice.stop))

    def __eq__(self, other):
        """Compare based on hashes."""
        return hash(self) == hash(other)


class ParameterSplit(Parameters, Splitable):
    """Implements splitting and merging on Parameters class."""

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def split(self, splitter: SplitVisitor):
        """Split Parameter's spatially dependent data."""
        data = None
        try:
            data = self.f
        except RuntimeError:
            return splitter.parts * (self,)

        # Split array for each key, creating a new dictionary with the same keys
        # but holding lists of arrays
        new = {key: splitter.split_array(data[key]) for key in data}

        # Create list of dictionaries that hold just one part of splitted arrays
        out = [{key: new[key][i] for key in new} for i in range(splitter.parts)]

        return tuple(self.__class__.from_parameters_with_data(self, o) for o in out)

    @classmethod
    @_ensure_others_is_tuple
    @lru_cache(maxsize=config.lru_cache_maxsize)
    def merge(cls, others: Sequence[Parameters], merger: MergeVisitor):
        """Merge Parameter's spatially varying data."""
        data = {}
        try:
            data = {
                key: merger.merge_array([o.f[key] for o in others])
                for key in others[0].f
            }
        except RuntimeError:
            pass

        return cls.from_parameters_with_data(others[0], data)

    @classmethod
    def from_parameters(cls, other: Parameters) -> "ParameterSplit":
        """Create from Parameters object."""
        if isinstance(other, cls):
            return other
        return cls.from_parameters_with_data(other, other._f)

    @classmethod
    def from_parameters_with_data(
        cls, other: Parameters, data: Dict[str, np.ndarray]
    ) -> "ParameterSplit":
        """Create from Parameters object."""
        kwargs = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if f.name not in ("_f", "_id")
        }
        kwargs["f_init"] = data
        return cls(**kwargs)


class GridSplit(Grid, Splitable):
    """Implements splitting and merging on Grid class."""

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def split(self, splitter: SplitVisitor):
        """Split grid."""
        x, y, mask, dx, dy = (
            splitter.split_array(arr)
            for arr in (self.x, self.y, self.mask, self.dx, self.dy)
        )
        return tuple(
            self.__class__(
                *args[:3],
                dx_init=args[3],
                dy_init=args[4],
                dim_x=self.dim_x,
                dim_y=self.dim_y,
            )
            for args in zip(x, y, mask, dx, dy)
        )

    @classmethod
    @_ensure_others_is_tuple
    @lru_cache(maxsize=config.lru_cache_maxsize)
    def merge(cls, others: Sequence[Grid], merger: MergeVisitor):
        """Merge grids."""
        x = merger.merge_array(tuple(o.x for o in others))
        y = merger.merge_array(tuple(o.y for o in others))
        mask = merger.merge_array(tuple(o.mask for o in others))
        dx = merger.merge_array(tuple(o.dx for o in others))
        dy = merger.merge_array(tuple(o.dy for o in others))
        return cls(
            x,
            y,
            mask,
            dim_x=others[0].dim_x,
            dim_y=others[0].dim_y,
            dx_init=dx,
            dy_init=dy,
        )

    @classmethod
    def from_grid(cls, grid):
        """Create from Grid object."""
        if isinstance(grid, cls):
            return grid
        return cls(x=grid.x, y=grid.y, mask=grid.mask, dx_init=grid.dx, dy_init=grid.dy)


@dataclass
class StaggeredGridSplit(StaggeredGrid, Splitable):
    """Implements splitting and merging on StaggeredGrid class."""

    def __post_init__(self, *args, **kwargs):
        """Cast Grid objects to GridSplit."""
        # super().__post_init__(*args, **kwargs)  # Not defined on StaggeredGrid
        self.u = GridSplit.from_grid(self.u)
        self.v = GridSplit.from_grid(self.v)
        self.eta = GridSplit.from_grid(self.eta)
        self.q = GridSplit.from_grid(self.q)

    def split(self, splitter: SplitVisitor):
        """Split staggered grids."""
        splitted_grids = {
            g: getattr(self, g).split(splitter) for g in ("u", "v", "eta", "q")
        }
        return tuple(
            self.__class__(**{g: splitted_grids[g][i] for g in ("u", "v", "eta", "q")})
            for i in range(splitter.parts)
        )

    @classmethod
    def merge(cls, others: Sequence[StaggeredGrid], merger: MergeVisitor):
        """Merge staggered grids."""
        return cls(
            **{
                g: GridSplit.merge(tuple(getattr(o, g) for o in others), merger)
                for g in ("u", "v", "eta", "q")
            }
        )

    @classmethod
    def from_staggered_grid(cls, staggered_grid):
        """Create from StaggeredGrid object."""
        if isinstance(staggered_grid, cls):
            return staggered_grid
        else:
            return cls(
                **{k: getattr(staggered_grid, k) for k in ("u", "v", "eta", "q")}
            )


@dataclass
class VariableSplit(Variable, Splitable):
    """Implements splitting and merging on Variable class."""

    def __post_init__(self):
        """Post initialization logic."""
        if not isinstance(self.grid, GridSplit):
            self.grid: GridSplit = GridSplit.from_grid(self.grid)

    def split(self, splitter: SplitVisitor):
        """Split variable."""
        splitted_grid = self.grid.split(splitter)
        if self.data is None:
            splitted_data = splitter.parts * (None,)
        else:
            splitted_data = splitter.split_array(self.safe_data)
        return tuple(
            self.__class__(data=d, grid=g) for d, g in zip(splitted_data, splitted_grid)
        )

    @classmethod
    def merge(cls, others: Sequence[Variable], merger: MergeVisitor):
        """Merge variable."""
        if all(o.data is None for o in others):
            data = None
        else:
            data = merger.merge_array([o.safe_data for o in others])
        return cls(
            data=data,
            grid=GridSplit.merge(tuple(o.grid for o in others), merger),
        )

    @classmethod
    def from_variable(cls, var):
        """Create from Variable object."""
        if isinstance(var, cls):
            return var
        else:
            return cls(data=var.data, grid=GridSplit.from_grid(var.grid))

    def __eq__(self, other):
        """Return true if other is identical or the same as self."""
        return super().__eq__(other)


class StateSplit(State, Splitable):
    """Implements splitting and merging on State class."""

    def split(self, splitter: SplitVisitor):
        """Split state."""
        splitted_u = VariableSplit.from_variable(self.u).split(splitter)
        splitted_v = VariableSplit.from_variable(self.v).split(splitter)
        splitted_eta = VariableSplit.from_variable(self.eta).split(splitter)
        return tuple(
            self.__class__(u, v, eta)
            for u, v, eta in zip(splitted_u, splitted_v, splitted_eta)
        )

    @classmethod
    def merge(cls, others: Sequence[State], merger: MergeVisitor):
        """Merge variables."""
        return cls(
            u=VariableSplit.merge([o.u for o in others], merger),
            v=VariableSplit.merge([o.v for o in others], merger),
            eta=VariableSplit.merge([o.eta for o in others], merger),
        )

    @classmethod
    def from_state(cls, state: State):
        """Create from state."""
        if isinstance(state, cls):
            return state
        else:
            return cls(
                u=VariableSplit.from_variable(state.u),
                v=VariableSplit.from_variable(state.v),
                eta=VariableSplit.from_variable(state.eta),
            )


class StateDequeSplit(deque, Splitable):
    """Implements splitting and merging on deque class."""

    def split(self, splitter: SplitVisitor):
        """Split StateDeque."""
        splitted_states = []
        for s in self:
            if isinstance(s, StateSplit):
                split_state = s.split(splitter)
            else:
                split_state = StateSplit.from_state(s).split(splitter)
            splitted_states.append(split_state)
        splitted_states = tuple(splitted_states)
        if len(splitted_states) == 0:
            return splitter.parts * (self.__class__([], maxlen=self.maxlen),)
        return tuple(
            self.__class__(states, maxlen=self.maxlen)
            for states in zip(*splitted_states)
        )

    @classmethod
    def merge(cls, others: Sequence[deque], merger: MergeVisitor):
        """Merge StateDeques."""
        return cls(
            (StateSplit.merge(states, merger) for states in zip(*others)),
            maxlen=others[0].maxlen,
        )

    @classmethod
    def from_state_deque(cls, state_deque):
        """Create from StateDeque object."""
        if isinstance(state_deque, cls):
            return state_deque
        else:
            return cls(state_deque, maxlen=state_deque.maxlen)


class DomainState(State, Domain, Splitable):
    """Implements Domain and Splitable interface on State class."""

    __slots__ = ["u", "v", "eta", "id", "it", "history", "parameter"]

    def __init__(
        self,
        u: Variable,
        v: Variable,
        eta: Variable,
        history: Optional[StateDequeSplit] = None,
        parameter: Optional[Parameters] = None,
        it: int = 0,
        id: int = 0,
    ):
        """Create new DomainState instance from references on Variable objects."""
        self.u: VariableSplit = VariableSplit.from_variable(u)
        self.v: VariableSplit = VariableSplit.from_variable(v)
        self.eta: VariableSplit = VariableSplit.from_variable(eta)
        if history is None:
            self.history = StateDequeSplit([], maxlen=3)
        else:
            self.history = StateDequeSplit.from_state_deque(history)
        if parameter is None:
            self.parameter = ParameterSplit()
        else:
            self.parameter = ParameterSplit.from_parameters(parameter)

        self.id = id
        self.it = it

    @classmethod
    def make_from_State(
        cls, s: State, history: Deque, parameter: Parameters, it: int, id: int = 0
    ):
        """Make DomainState object from State objects, without copying Variables."""
        return cls(
            s.u,
            s.v,
            s.eta,
            StateDequeSplit.from_state_deque(history),
            parameter,
            it,
            id,
        )

    @property
    def as_state(self) -> StateSplit:
        """Return state variables as StateSplit object."""
        return StateSplit(u=self.u, v=self.v, eta=self.eta)

    def set_id(self, id):
        """Set id value."""
        self.id = id
        return self

    def get_id(self) -> int:
        """Get domain's ID."""
        return self.id

    def get_iteration(self) -> int:
        """Get domain's iteration."""
        return self.it

    def get_data(self):
        """Provide tuple of all Variables in this order: (u, v, eta)."""
        return self.u, self.v, self.eta

    def increment_iteration(self) -> int:
        """Return incremented iteration from domain, not modify object itself."""
        return self.it + 1

    def split(
        self, splitter: SplitVisitor
    ) -> Tuple["DomainState", ...]:  # TODO: raise error if shape[dim[0]] // parts < 2
        """Implement the split method from API."""
        splitted = (
            self.u.split(splitter),
            self.v.split(splitter),
            self.eta.split(splitter),
            self.history.split(splitter),
            self.parameter.split(splitter),
        )

        out = tuple(
            self.__class__(
                u,
                v,
                eta,
                h,
                p,
                self.it,
                self.get_id(),
            )
            for u, v, eta, h, p in zip(*splitted)
        )

        return out

    @classmethod
    def merge(cls, others: Sequence["DomainState"], merger: MergeVisitor):
        """Implement merge method from API."""
        if any(tuple(o.it != others[0].it for o in others)):
            raise ValueError(
                "Try to merge DomainStates that differ in iteration counter."
            )
        return DomainState(
            VariableSplit.merge([o.u for o in others], merger),
            VariableSplit.merge([o.v for o in others], merger),
            VariableSplit.merge([o.eta for o in others], merger),
            StateDequeSplit.merge([o.history for o in others], merger),
            ParameterSplit.merge((o.parameter for o in others), merger),
            others[0].get_iteration(),
            others[0].get_id(),
        )

    def copy(self):
        """Return a deep copy of the object."""
        return deepcopy(self)

    def __eq__(self, other) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, DomainState):
            return NotImplemented
        if self is other:
            return True
        for a in self.__slots__:
            if not getattr(self, a) == getattr(other, a):
                return False
        return True


class BorderState(DomainState, Border):
    """Implementation of Border class from API on State class."""

    def __init__(  # differs from Border.__init__ signature
        self,
        u: Variable,
        v: Variable,
        eta: Variable,
        width: int,
        dim: int,
        iteration: int,
        history: Optional[StateDequeSplit] = None,
        parameter: Optional[ParameterSplit] = None,
        id: int = 0,
    ):
        """Create BorderState in the same way as DomainState."""
        super().__init__(u, v, eta, history, parameter, iteration, id)
        self.width = width
        self.dim = dim

    @classmethod
    def create_border(cls, base: DomainState, width: int, direction: bool, dim: int):
        """Split border of a DomainState instance.

        The data of the boarder will be copied to avoid data races.
        """
        if direction:
            border_slice = BorderDirection.RIGHT(width)
        else:
            border_slice = BorderDirection.LEFT(width)
        splitter = BorderSplitter(slice=border_slice, axis=dim)
        splitted_state = base.split(splitter)[0]

        return cls.from_domain_state(splitted_state, width=width, dim=dim)

    @classmethod
    def from_domain_state(cls, domain_state: DomainState, width: int, dim: int):
        """Create an instance from a DomainState instance.

        No copies are created.
        """
        return cls(
            u=domain_state.u,
            v=domain_state.v,
            eta=domain_state.eta,
            width=width,
            dim=dim,
            iteration=domain_state.get_iteration(),
            history=domain_state.history,
            parameter=domain_state.parameter,
            id=domain_state.get_id(),
        )

    def get_width(self) -> int:
        """Get border's width."""
        return self.width

    def get_dim(self) -> int:
        """Get border's dimension."""
        return self.dim


class BorderMerger(MergeVisitor):
    """Implements merging of the borders with a DomainState along a dimension.

    This merger is suppose to be used in the merge classmethod of the DomainState class.
    The order of arguments must be (left_border, domain, right_border).
    """

    __slots__ = ["_axis", "_slice_left", "_slice_right", "_slice_center", "_width"]

    def __init__(self, width: int, axis: int):
        """Initialize class instance."""
        self._axis = axis
        self._slice_left = BorderDirection.LEFT(width)
        self._slice_right = BorderDirection.RIGHT(width)
        self._slice_center = BorderDirection.CENTER(width, width)
        self._width = width

    @classmethod
    def from_borders(
        cls, left_border: BorderState, right_border: BorderState
    ) -> "BorderMerger":
        """Create BorderManager from left and right border instance."""
        assert left_border.width == right_border.width
        assert left_border.dim == right_border.dim
        return cls(width=left_border.width, axis=left_border.dim)

    def merge_array(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Merge array.

        Parameter
        ---------
        arrays: Sequence[np.ndarray]
          Arrays to merge.

        Returns
        -------
        np.ndarray
        """
        slices_center = arrays[1].ndim * [slice(None)]
        slices_center[self._axis] = self._slice_center

        left, base, right = arrays
        out = np.concatenate((left, base[tuple(slices_center)], right), axis=self._axis)
        return out

    def __hash__(self):
        """Return hash based on axis and slice objects."""
        return hash((self._axis, self._width))

    def __eq__(self, other):
        """Compare based on hash values."""
        return hash(self) == hash(other)


# TODO: make this work for dim != 1
class Tail(Tailor):
    """Implement Tailor class from API."""

    @staticmethod
    def split_domain(
        base: DomainState, splitter: SplitVisitor
    ) -> Tuple[DomainState, ...]:
        """Split domain in subdomains.

        When splitting, the ids of the subdomains are set to `range(0, splitter.parts)`.
        """
        splitted = base.split(splitter)
        for i, s in enumerate(splitted):
            s.id = i
        return splitted

    @staticmethod
    def make_borders(
        base: DomainState, width: int, dim: int
    ) -> Tuple[BorderState, BorderState]:
        """Implement make_borders method from API."""
        return (
            BorderState.create_border(base, width, False, dim),
            BorderState.create_border(base, width, True, dim),
        )

    @staticmethod
    def stitch(
        base: DomainState, borders: Tuple[BorderState, BorderState], dims: tuple
    ) -> DomainState:
        """Implement stitch method from API.

        borders need to be ordered left_border, right_border
        """
        left_border, right_border = borders
        border_merger = BorderMerger.from_borders(left_border, right_border)

        if (
            base.get_iteration()
            == left_border.get_iteration()
            == right_border.get_iteration()
        ):
            assert base.get_id() == left_border.get_id() == right_border.get_id()
        else:
            raise ValueError(
                "Borders iteration mismatch. Left: {}, right: {}, domain: {}".format(
                    left_border.get_iteration(),
                    right_border.get_iteration(),
                    base.get_iteration(),
                )
            )
        # necessary for caching of split / merge operations since
        # hashing of ParameterSplit and GridSplit is id based.
        vars = {
            v: VariableSplit(
                data=border_merger.merge_array(
                    tuple(
                        getattr(o, v).safe_data
                        for o in (left_border, base, right_border)
                    )
                ),
                grid=base.__getattribute__(v).grid,
            )
            for v in ("u", "v", "eta")
        }
        merged_state = StateSplit(**vars)
        return DomainState.make_from_State(
            merged_state,
            base.history,
            base.parameter,
            base.get_iteration(),
            base.get_id(),
        )


def _dump_to_redis(domain: DomainState):
    r = Redis(host="localhost", port="6379", db="0")

    if r.ping():
        flag = int(r.get("_avg_eta"))

        if flag == 1:
            k = format(domain.id, "05d") + "_" + format(domain.it, "05d") + "_eta"
            h, w = domain.eta.safe_data.shape
            shape = pack(">II", h, w)
            encoded = shape + domain.eta.safe_data.tobytes()

            r.set(k, encoded)


class GeneralSolver(Solver):
    """Implement Solver class from API for use with any provided function.

    Currently it performs only Euler forward scheme.
    """

    __slots__ = ["step", "slv", "schema"]

    def __init__(
        self,
        solution: Callable[[State, Parameters], State],
        schema: TimeSteppingFunction,
        step: float = 1,
    ):
        """Initialize GeneralSolver object providing function to compute next iterations.

        Arguments
        ---------
        solution: (State, Parameters]) -> State
            Function that takes State and Parameters and returns State.
            It is used to compute next iteration.
            Functions like linearised_SWE are highly recommended.

        schema: TimeSteppingFunction
            integration schema like euler_forward or adams_bashforth3

        step: float
            Quanta of time in the integration process.
        """
        self.step = step
        self.slv = solution
        self.sch = schema

    def integration(self, domain: DomainState) -> DomainState:
        """Implement integration method from API."""
        inc = self._compute_increment(domain, domain.parameter)
        new = self._integrate(domain, inc)
        return new

    def get_border_width(self) -> int:
        """Retuns fixed border width."""
        return 2

    def partial_integration(
        self,
        border: BorderState,
        domain: DomainState,
        neighbor_border: BorderState,
        direction: bool,
        dim: int,
    ) -> Border:
        """Implement partial_integration from API."""
        b_w = border.get_width()
        dim = border.dim
        assert domain.u.grid.x.shape[dim] >= 2 * b_w

        if direction:
            halo_slice = BorderDirection.RIGHT_HALO(b_w)
        else:
            halo_slice = BorderDirection.LEFT_HALO(b_w)

        halo_splitter = BorderSplitter(slice=halo_slice, axis=dim)
        list_merger = RegularSplitMerger(2, (dim,))
        area_of_interest_splitter = BorderSplitter(
            slice=BorderDirection.CENTER(b_w, b_w), axis=dim
        )

        dom = domain.as_state.split(halo_splitter)[0]
        dom_param = domain.parameter.split(halo_splitter)[0]

        state_list = [dom, border.as_state, neighbor_border.as_state]
        param_list = [dom_param, border.parameter, neighbor_border.parameter]
        if not direction:
            state_list.reverse()
            param_list.reverse()
        merged_state = StateSplit.merge(state_list, list_merger)
        merged_parameters = ParameterSplit.merge(tuple(param_list), list_merger)
        # tmp = self.integration(tmp)
        inc = self._compute_increment(merged_state, merged_parameters)
        new = self._integrate(
            border,
            inc.split(area_of_interest_splitter)[0],
        )

        result = BorderState.from_domain_state(
            new,
            width=border.get_width(),
            dim=dim,
        )
        result.id = domain.get_id()
        return result

    def window(self, domain: Future, client: Client) -> Future:
        """Do nothing."""
        # fire_and_forget(client.submit(_dump_to_redis, domain))
        return domain

    def _compute_increment(self, state: State, parameter: Parameters) -> StateSplit:
        inc = StateSplit.from_state(self.slv(state, parameter))
        return inc

    def _integrate(self, domain: DomainState, inc: State) -> DomainState:
        # shallow copy to avoid side effects
        history = copy(domain.history)
        history.append(inc)
        new = self.sch(history, domain.parameter, self.step)
        return DomainState(
            domain.u + new.u,
            domain.v + new.v,
            domain.eta + new.eta,
            history,
            domain.parameter,
            domain.increment_iteration(),
            domain.get_id(),
        )