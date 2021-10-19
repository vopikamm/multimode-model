"""Implementation of domain split API."""
from .domain_split_API import Domain, Border, Solver, SplitMerger, Tailor, Splitable
from .datastructure import State, Variable, np, Parameters
from .grid import Grid, StaggeredGrid
from dask.distributed import Client, Future
from redis import Redis
from struct import pack
from collections import deque
from typing import Optional, Sequence, Tuple


def _new_grid(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Grid:
    return Grid(x.copy(), y.copy(), mask.copy())


def _new_variable(
    data: np.ndarray, x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> Variable:
    """Create explicit copies of all input arrays and creates new Variable object."""
    return Variable(data.copy(), _new_grid(x, y, mask))


def _copy_variable(var: Variable) -> Variable:
    return Variable(
        var.safe_data.copy(),
        Grid(var.grid.x.copy(), var.grid.y.copy(), var.grid.mask.copy()),
    )


class RegularSplitMerger(SplitMerger):
    """Implements splitting and merging into regular grid."""

    __slots__ = ["tuple", "_parts"]

    def __init__(self, parts: int, dim: Tuple[int]):
        """Initialize class instance."""
        self._parts = parts
        self.dim = dim

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

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return self._parts


class BorderSplitter(SplitMerger):
    """Implements splitting off the borders of a DomainState along a dimension.

    The required merge_array method is not implemented, hence the use in a call to merge
    of a class implementing the `Splittable` interface will raise an runtime exception.
    """

    __slots__ = ["_axis", "_slice"]

    def __init__(self, width: int, axis: int, direction: bool):
        """Initialize class instance."""
        self._axis = axis

        if direction:
            self._slice = slice(-width, None)
        else:
            self._slice = slice(None, width)

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
        raise NotImplementedError("Merging not supported by {self.__class__.__name__}")

    @property
    def parts(self) -> int:
        """Return number of parts created by split."""
        return 1


class ParameterSplit(Parameters, Splitable):
    """Implements splitting and merging on Parameters class."""

    def __init__(self, other: Parameters, data: dict):
        """Create class instance from another one and Coriolis data."""
        self.g = other.g
        self.H = other.H
        self.rho_0 = other.rho_0
        self._f = data

    def split(self, splitter: SplitMerger):
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

        return tuple(self.__class__(self, o) for o in out)

    @classmethod
    def merge(cls, others: Sequence[Parameters], merger: SplitMerger):
        """Merge Parameter's Coriolis data."""
        data = {}
        try:
            data = {
                key: merger.merge_array([o.f[key] for o in others])
                for key in others[0].f
            }
        except RuntimeError:
            pass

        return cls(others[0], data)

    @classmethod
    def from_parameters(cls, params):
        """Create from Parameters object."""
        return cls(params, params.f)


class GridSplit(Grid, Splitable):
    """Implements splitting and merging on Grid class."""

    def split(self, splitter: SplitMerger):
        """Split grid."""
        x, y, mask = (splitter.split_array(arr) for arr in (self.x, self.y, self.mask))
        return tuple(self.__class__(*args) for args in zip(x, y, mask))

    @classmethod
    def merge(cls, others, merger: SplitMerger):
        """Merge grids."""
        x = merger.merge_array(tuple(o.x for o in others))
        y = merger.merge_array(tuple(o.y for o in others))
        mask = merger.merge_array(tuple(o.mask for o in others))
        return cls(x, y, mask)

    @classmethod
    def from_grid(cls, grid):
        """Create from Grid object."""
        return cls(x=grid.x, y=grid.y, mask=grid.mask)


class StaggeredGridSplit(StaggeredGrid, Splitable):
    """Implements splitting and merging on StaggeredGrid class."""

    def __post_init__(self, *args, **kwargs):
        """Cast Grid objects to GridSplit."""
        # super().__post_init__(*args, **kwargs)  # Not defined on StaggeredGrid
        self.u = GridSplit.from_grid(self.u)
        self.v = GridSplit.from_grid(self.v)
        self.eta = GridSplit.from_grid(self.eta)
        self.q = GridSplit.from_grid(self.q)

    def split(self, splitter: SplitMerger):
        """Split staggered grids."""
        splitted_grids = {
            g: getattr(self, g).split(splitter) for g in ("u", "v", "eta", "q")
        }
        return tuple(
            self.__class__(**{g: splitted_grids[g][i] for g in ("u", "v", "eta", "q")})
            for i in range(splitter.parts)
        )

    @classmethod
    def merge(cls, others, merger: SplitMerger):
        """Merge staggered grids."""
        return cls(
            **{
                g: GridSplit.merge((getattr(o, g) for o in others), merger)
                for g in ("u", "v", "eta", "q")
            }
        )

    @classmethod
    def from_staggered_grid(cls, staggered_grid):
        """Create from StaggeredGrid object."""
        return cls(**{k: getattr(staggered_grid, k) for k in ("u", "v", "eta", "q")})


class VariableSplit(Variable, Splitable):
    """Implements splitting and merging on Variable class."""

    def __post_init__(self):
        """Post initialization logic."""
        if not isinstance(self.grid, Splitable):
            self.grid: GridSplit = GridSplit.from_grid(self.grid)

    def split(self, splitter: SplitMerger):
        """Split variable."""
        data = self.safe_data
        splitted_grid = self.grid.split(splitter)
        if self.data is None:
            splitted_data = splitter.parts * (None,)
        else:
            splitted_data = splitter.split_array(data)
        return tuple(
            self.__class__(data=d, grid=g) for d, g in zip(splitted_data, splitted_grid)
        )

    @classmethod
    def merge(cls, others, merger: SplitMerger):
        """Merge variable."""
        return cls(
            data=merger.merge_array([o.data for o in others]),
            grid=GridSplit.merge([o.grid for o in others], merger),
        )

    @classmethod
    def from_variable(cls, var):
        """Create from Variable object."""
        return cls(data=var.data, grid=GridSplit.from_grid(var.grid))


class StateSplit(State, Splitable):
    """Implements splitting and merging on State class."""

    def split(self, splitter: SplitMerger):
        """Split state."""
        splitted_u = VariableSplit.from_variable(self.u).split(splitter)
        splitted_v = VariableSplit.from_variable(self.v).split(splitter)
        splitted_eta = VariableSplit.from_variable(self.eta).split(splitter)
        return tuple(
            self.__class__(u, v, eta)
            for u, v, eta in zip(splitted_u, splitted_v, splitted_eta)
        )

    @classmethod
    def merge(cls, others, merger: SplitMerger):
        """Merge variables."""
        return cls(
            u=VariableSplit.merge([o.u for o in others], merger),
            v=VariableSplit.merge([o.v for o in others], merger),
            eta=VariableSplit.merge([o.eta for o in others], merger),
        )

    @classmethod
    def from_state(cls, state):
        """Create from state."""
        return cls(
            u=VariableSplit.from_variable(state.u),
            v=VariableSplit.from_variable(state.v),
            eta=VariableSplit.from_variable(state.eta),
        )


class StateDequeSplit(deque, Splitable):
    """Implements splitting and merging on deque class."""

    def split(self, splitter: SplitMerger):
        """Split StateDeque."""
        splitted_states = tuple(StateSplit.from_state(s).split(splitter) for s in self)
        if len(splitted_states) == 0:
            return splitter.parts * (self.__class__([], maxlen=self.maxlen),)
        return tuple(
            self.__class__(states, maxlen=self.maxlen)
            for states in zip(*splitted_states)
        )

    @classmethod
    def merge(cls, others, merger: SplitMerger):
        """Merge StateDeques."""
        return cls(
            (StateSplit.merge(states, merger) for states in zip(*others)),
            maxlen=others[0].maxlen,
        )

    @classmethod
    def from_state_deque(cls, state_deque):
        """Create from StateDeque object."""
        return cls(state_deque, maxlen=state_deque.maxlen)


class DomainState(State, Domain, Splitable):
    """Implements Domain and Splitable interface on State class."""

    __slots__ = ["u", "v", "eta", "id", "it", "history", "p"]

    def __init__(
        self,
        u: Variable,
        v: Variable,
        eta: Variable,
        history: Optional[StateDequeSplit] = None,
        parameter: Optional[ParameterSplit] = None,
        it: int = 0,
        id: int = 0,
    ):
        """Create new DomainState instance from references on Variable objects."""
        self.u: VariableSplit = VariableSplit.from_variable(u)
        self.v: VariableSplit = VariableSplit.from_variable(v)
        self.eta: VariableSplit = VariableSplit.from_variable(eta)
        if history is None:
            self.history: StateDequeSplit = StateDequeSplit([], maxlen=3)
        else:
            self.history = history
        if parameter is None:
            self.parameter = ParameterSplit(Parameters(), {})
        else:
            self.parameter = parameter

        self.id = id
        self.it = it

    @classmethod
    def make_from_State(
        cls, s: State, history, parameter: Parameters, it: int, id: int = 0
    ):
        """Make DomainState object from State objects, without copying Variables."""
        return cls(
            VariableSplit.from_variable(s.u),
            VariableSplit.from_variable(s.v),
            VariableSplit.from_variable(s.eta),
            StateDequeSplit.from_state_deque(history),
            ParameterSplit.from_parameters(parameter),
            it,
            id,
        )

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
        self, splitter: SplitMerger
    ):  # TODO: raise error if shape[dim[0]] // parts < 2
        """Implement the split method from API."""
        splitted = (
            self.u.split(splitter),
            self.v.split(splitter),
            self.eta.split(splitter),
            self.history.split(splitter),
            self.parameter.split(splitter),
        )

        # [print(len(e)) for e in splitted]

        out = tuple(
            self.__class__(
                u,
                v,
                eta,
                h,
                p,
                self.it,
                i,
            )
            for i, (u, v, eta, h, p) in enumerate(zip(*splitted))
        )

        return out

    @classmethod
    def merge(cls, others, merger: SplitMerger):
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
            ParameterSplit.merge([o.parameter for o in others], merger),
            others[0].get_iteration(),
            others[0].get_id(),
        )


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
        """Create BorderState instance from DomainState."""
        splitter = BorderSplitter(width=width, axis=dim, direction=direction)
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


class Tail(Tailor):
    """Implement Tailor class from API."""

    def make_borders(
        self, base: DomainState, width: int, dim: int
    ) -> Tuple[BorderState, BorderState]:
        """Implement make_borders method from API."""
        return (
            BorderState.create_border(base, width, False, dim),
            BorderState.create_border(base, width, True, dim),
        )

    def stitch(self, base: DomainState, borders: tuple, dims: tuple) -> DomainState:
        """Implement stitch method from API."""
        u, v, eta = (_copy_variable(v) for v in base.get_data())
        l_border, r_border = borders[0]

        if base.get_iteration() == l_border.get_iteration() == r_border.get_iteration():
            assert base.get_id() == l_border.get_id() == r_border.get_id()
        else:
            raise Exception(
                "Borders iteration mismatch. Left: {}, right: {}, domain: {}".format(
                    l_border.get_iteration(),
                    r_border.get_iteration(),
                    base.get_iteration(),
                )
            )

        u.data[:, (u.data.shape[1] - r_border.get_width()) :] = r_border.get_data()[
            0
        ].safe_data.copy()
        v.data[:, (u.data.shape[1] - r_border.get_width()) :] = r_border.get_data()[
            1
        ].safe_data.copy()
        eta.data[:, (u.data.shape[1] - r_border.get_width()) :] = r_border.get_data()[
            2
        ].safe_data.copy()

        u.data[:, : l_border.get_width()] = l_border.get_data()[0].safe_data.copy()
        v.data[:, : l_border.get_width()] = l_border.get_data()[1].safe_data.copy()
        eta.data[:, : l_border.get_width()] = l_border.get_data()[2].safe_data.copy()

        return DomainState(
            u, v, eta, base.history, base.parameter, base.get_iteration(), base.get_id()
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

            print(k)
            r.set(k, encoded)


class GeneralSolver(Solver):
    """Implement Solver class from API for use with any provided function.

    Currently it performs only Euler forward scheme.
    """

    def __init__(self, solution, schema, step=1):
        """Initialize GeneralSolver object providing function to compute next iterations.

        Arguments
        ---------
        solution
            function that takes State and Parameters and returns State.
            It is used to compute next iteration.
            Functions like linearised_SWE are highly recommended.

        schema
            integration schema like fourier_forward or adams_bashforth3

        step
            Quanta of time in the integration process.
        """
        self.step = step
        self.slv = solution
        self.sch = schema

    def _integrate(self, domain: DomainState) -> DomainState:
        inc = self.slv(domain, domain.parameter)
        domain.history.append(inc)
        new = self.sch(domain.history, domain.parameter, self.step)
        return DomainState(
            domain.u + new.u,
            domain.v + new.v,
            domain.eta + new.eta,
            domain.history,
            domain.parameter,
            domain.increment_iteration(),
            domain.get_id(),
        )

    def integration(self, domain: DomainState) -> DomainState:
        """Implement integration method from API."""
        return self._integrate(domain)

    def get_border_width(self) -> int:
        """Retuns fixed border width."""
        return 2

    def partial_integration(
        self,
        domain: DomainState,
        border: BorderState,
        past: BorderState,
        direction: bool,
        dim: int,
    ) -> Border:
        """Implement partial_integration from API."""
        b_w = border.get_width()
        dom = BorderState.create_border(domain, 2 * b_w, direction, dim)
        list = (
            [dom, border] if direction else [border, dom]
        )  # order inside list shows if it's left of right border
        tmp = DomainState.merge(
            list, RegularSplitMerger(2, (dim,))
        )  # TODO: refactor this
        tmp.history = past.history
        tmp = self._integrate(tmp)

        u = Variable(
            tmp.u.data[:, b_w : 2 * b_w],
            Grid(
                tmp.u.grid.x[:, b_w : 2 * b_w],
                tmp.u.grid.y[:, b_w : 2 * b_w],
                tmp.u.grid.mask[:, b_w : 2 * b_w],
            ),
        )

        v = Variable(
            tmp.v.safe_data[:, b_w : 2 * b_w],
            Grid(
                tmp.v.grid.x[:, b_w : 2 * b_w],
                tmp.v.grid.y[:, b_w : 2 * b_w],
                tmp.v.grid.mask[:, b_w : 2 * b_w],
            ),
        )

        eta = Variable(
            tmp.eta.safe_data[:, b_w : 2 * b_w],
            Grid(
                tmp.eta.grid.x[:, b_w : 2 * b_w],
                tmp.eta.grid.y[:, b_w : 2 * b_w],
                tmp.eta.grid.mask[:, b_w : 2 * b_w],
            ),
        )

        return BorderState(
            u,
            v,
            eta,
            border.get_width(),
            dim,
            domain.increment_iteration(),
            past.history,
            past.parameter,
            domain.get_id(),
        )

    def window(self, domain: Future, client: Client) -> Future:
        """Do nothing."""
        # fire_and_forget(client.submit(_dump_to_redis, domain))
        return domain
