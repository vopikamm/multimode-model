"""Implementation of domain split API."""
from .domain_split_API import Domain, Border, Solver, Tailor
from .datastructure import State, Variable, np, Parameters
from .grid import Grid
from dask.distributed import Client, Future, fire_and_forget
from redis import Redis
from struct import pack
from collections import deque


def _new_grid(x: np.array, y: np.array, mask: np.array) -> Grid:
    return Grid(x.copy(), y.copy(), mask.copy())


def _new_variable(data: np.array, x: np.array, y: np.array, mask: np.array) -> Variable:
    """Create explicit copies of all input arrays and creates new Variable object."""
    return Variable(data.copy(), _new_grid(x, y, mask))


def _copy_variable(var: Variable) -> Variable:
    return Variable(
        var.safe_data.copy(),
        Grid(var.grid.x.copy(), var.grid.y.copy(), var.grid.mask.copy()),
    )


class DomainState(State, Domain):
    """Implements Domain interface on State class."""

    def __init__(
        self, u: Variable, v: Variable, et: Variable, h: deque, it: int = 0, id: int = 0
    ):
        """Create new DomainState instance from references on Variable objects."""
        self.u = u
        self.v = v
        self.eta = et
        self.id = id
        self.it = it
        self.history = h

    @classmethod
    def make_from_State(cls, s: State, h, it: int, id: int = 0):
        """Make DomainState object from State objects, without copying Variables."""
        return cls(s.u, s.v, s.eta, h, it, id)

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

    def split(self, parts: int, dim: tuple):
        """Implement the split method from API."""
        v_x = np.array_split(self.v.grid.x, parts, dim[0])
        v_y = np.array_split(self.v.grid.y, parts, dim[0])
        v_mask = np.array_split(self.v.grid.mask, parts, dim[0])
        v = np.array_split(self.v.safe_data, parts, dim[0])

        u_x = np.array_split(self.u.grid.x, parts, dim[0])
        u_y = np.array_split(self.u.grid.y, parts, dim[0])
        u_mask = np.array_split(self.u.grid.mask, parts, dim[0])
        u = np.array_split(self.u.safe_data, parts, dim[0])

        eta_x = np.array_split(self.eta.grid.x, parts, dim[0])
        eta_y = np.array_split(self.eta.grid.y, parts, dim[0])
        eta_mask = np.array_split(self.eta.grid.mask, parts, dim[0])
        eta = np.array_split(self.eta.safe_data, parts, dim[0])

        out = [
            self.__class__(
                _new_variable(u[i], u_x[i], u_y[i], u_mask[i]),
                _new_variable(v[i], v_x[i], v_y[i], v_mask[i]),
                _new_variable(eta[i], eta_x[i], eta_y[i], eta_mask[i]),
                deque([], maxlen=self.history.maxlen),
                0,
                i
            )
            for i in range(parts)
        ]

        return out

    @classmethod
    def merge(cls, others, dim: int):
        """Implement merge method from API."""
        v_x = np.concatenate([o.v.grid.x for o in others], dim)
        v_y = np.concatenate([o.v.grid.y for o in others], dim)
        v_mask = np.concatenate([o.v.grid.mask for o in others], dim)
        v = np.concatenate([o.v.safe_data for o in others], dim)

        u_x = np.concatenate([o.u.grid.x for o in others], dim)
        u_y = np.concatenate([o.u.grid.y for o in others], dim)
        u_mask = np.concatenate([o.u.grid.mask for o in others], dim)
        u = np.concatenate([o.u.safe_data for o in others], dim)

        eta_x = np.concatenate([o.eta.grid.x for o in others], dim)
        eta_y = np.concatenate([o.eta.grid.y for o in others], dim)
        eta_mask = np.concatenate([o.eta.grid.mask for o in others], dim)
        eta = np.concatenate([o.eta.safe_data for o in others], dim)

        return DomainState(
            _new_variable(u, u_x, u_y, u_mask),
            _new_variable(v, v_x, v_y, v_mask),
            _new_variable(eta, eta_x, eta_y, eta_mask),
            deque([], maxlen=others[0].history.maxlen),
            others[0].get_iteration(),
            others[0].get_id()
        )


class BorderState(DomainState, Border):
    """Implementation of Border class from API on State class."""

    def __init__(
            self,
            u: Variable,
            v: Variable,
            eta: Variable,
            ancestors: deque,
            width: int,
            dim: int,
            iteration: int,
            id: int = 0
    ):
        """Create BorderState in the same way as DomainState."""
        super().__init__(u, v, eta, ancestors, iteration, id)
        self.width = width
        self.dim = dim

    @classmethod
    def create_border(cls, base: DomainState, width: int, direction: bool, dim: int):
        """Create BorderState instance from DomainState."""
        u, v, eta = base.get_data()

        u_x = u.grid.x
        u_y = u.grid.y
        u_mask = u.grid.mask
        u = u.safe_data

        v_x = v.grid.x
        v_y = v.grid.y
        v_mask = v.grid.mask
        v = v.safe_data

        eta_x = eta.grid.x
        eta_y = eta.grid.y
        eta_mask = eta.grid.mask
        eta = eta.safe_data

        place = u.shape[dim] - width
        if direction:
            return BorderState(
                _new_variable(
                    u[:, place:], u_x[:, place:], u_y[:, place:], u_mask[:, place:]
                ),
                _new_variable(
                    v[:, place:], v_x[:, place:], v_y[:, place:], v_mask[:, place:]
                ),
                _new_variable(
                    eta[:, place:],
                    eta_x[:, place:],
                    eta_y[:, place:],
                    eta_mask[:, place:],
                ),
                deque([], base.history.maxlen),
                width,
                dim,
                base.get_iteration(),
                base.get_id(),
            )
        else:
            return BorderState(
                _new_variable(
                    u[:, :width], u_x[:, :width], u_y[:, :width], u_mask[:, :width]
                ),
                _new_variable(
                    v[:, :width], v_x[:, :width], v_y[:, :width], v_mask[:, :width]
                ),
                _new_variable(
                    eta[:, :width],
                    eta_x[:, :width],
                    eta_y[:, :width],
                    eta_mask[:, :width],
                ),
                deque([], base.history.maxlen),
                width,
                dim,
                base.get_iteration(),
                base.get_id(),
            )

    def get_width(self) -> int:
        """Get border's width."""
        return self.width

    def get_dim(self) -> int:
        """Get border's dimension."""
        return self.dim


class Tail(Tailor):
    """Implement Tailor class from API."""

    def make_borders(self, base: Domain, width: int, dim: int) -> (Border, Border):
        """Implement make_borders method from API."""
        return (BorderState.create_border(base, width, False, dim),
                BorderState.create_border(base, width, True, dim))

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

        u.data[:, (u.data.shape[1] - r_border.get_width()):] = r_border.get_data()[
            0
        ].safe_data.copy()
        v.data[:, (u.data.shape[1] - r_border.get_width()):] = r_border.get_data()[
            1
        ].safe_data.copy()
        eta.data[:, (u.data.shape[1] - r_border.get_width()):] = r_border.get_data()[
            2
        ].safe_data.copy()

        u.data[:, : l_border.get_width()] = l_border.get_data()[0].safe_data.copy()
        v.data[:, : l_border.get_width()] = l_border.get_data()[1].safe_data.copy()
        eta.data[:, : l_border.get_width()] = l_border.get_data()[2].safe_data.copy()

        return DomainState(u, v, eta, base.history, base.get_iteration(), base.get_id())


def _dump_to_redis(domain: DomainState):
    r = Redis(host='localhost', port='6379', db='0')

    if r.ping():
        flag = int(r.get("_avg_eta"))

        if flag == 1:
            k = format(domain.id, '05d') + "_" + format(domain.it, '05d') + "_eta"
            h, w = domain.eta.safe_data.shape
            shape = pack('>II', h, w)
            encoded = shape + domain.eta.safe_data.tobytes()

            print(k)
            r.set(k, encoded)


class GeneralSolver(Solver):
    """Implement Solver class from API for use with any provided function.

    Currently it performs only Euler forward scheme.
    """

    def __init__(self, solution, schema, params: Parameters = Parameters(), step=1):
        """Initialize GeneralSolver object providing function to compute next iterations.

        Arguments
        ---------
        solution
            function that takes State and Parameters and returns State.
            It is used to compute next iteration.
            Functions like linearised_SWE are highly recommended.

        schema
            integration schema like fourier_forward or adams_bashforth3

        params: Parameters
            Object with parameter, passed to solution function along DomainState object.

        step
            Quanta of time in the integration process.
        """
        self.step = step
        self.params = params
        self.slv = solution
        self.sch = schema

    def _integrate(self, domain: DomainState) -> DomainState:
        inc = self.slv(domain, self.params)
        domain.history.append(inc)
        new = self.sch(domain.history, self.params, self.step)
        return DomainState(domain.u + new.u,
                           domain.v + new.v,
                           domain.eta + new.eta,
                           domain.history,
                           domain.increment_iteration(),
                           domain.get_id())

    def integration(self, domain: Domain) -> Domain:
        """Implement integration method from API."""
        return self._integrate(domain)

    def get_border_width(self) -> int:
        """Retuns fixed border width."""
        return 2

    def partial_integration(
        self, domain: Domain, border: Border, past: Border, direction: bool, dim: int
    ) -> Border:
        """Implement partial_integration from API."""
        b_w = border.get_width()
        dom = BorderState.create_border(domain, 2 * b_w, direction, dim)
        list = (
            [dom, border] if direction else [border, dom]
        )  # order inside list shows if it's left of right border
        tmp = DomainState.merge(list, dim)
        tmp.history = past.history
        tmp = self._integrate(tmp)

        u = Variable(
            tmp.u.data[:, b_w: 2 * b_w],
            Grid(
                tmp.u.grid.x[:, b_w: 2 * b_w],
                tmp.u.grid.y[:, b_w: 2 * b_w],
                tmp.u.grid.mask[:, b_w: 2 * b_w],
            ),
        )

        v = Variable(
            tmp.v.safe_data[:, b_w: 2 * b_w],
            Grid(
                tmp.v.grid.x[:, b_w: 2 * b_w],
                tmp.v.grid.y[:, b_w: 2 * b_w],
                tmp.v.grid.mask[:, b_w: 2 * b_w],
            ),
        )

        eta = Variable(
            tmp.eta.safe_data[:, b_w: 2 * b_w],
            Grid(
                tmp.eta.grid.x[:, b_w: 2 * b_w],
                tmp.eta.grid.y[:, b_w: 2 * b_w],
                tmp.eta.grid.mask[:, b_w: 2 * b_w],
            ),
        )

        return BorderState(u, v, eta,
                           past.history,
                           border.get_width(),
                           dim,
                           domain.increment_iteration(),
                           domain.get_id())

    def window(self, domain: Future, client: Client) -> Future:
        """Do nothing."""
        fire_and_forget(client.submit(_dump_to_redis, domain))
        return domain
