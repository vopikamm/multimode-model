"""Run this file using
```bash
python -m cProfile -o $(command).dat benchmark/domain_split_API.py $(command)
```

where command is one of "classic", "no_split", "split". This will create a profiling dump
which can be analyzed using, e.g., snakeviz.
"""

# flake8: noqa

from collections import deque
import sys
import numpy as np
from multimodemodel import (
    StaggeredGrid,
    State,
    Variable,
    Parameters,
    integrate,
    adams_bashforth3,
    pressure_gradient_i,
    pressure_gradient_j,
    divergence_i,
    divergence_j,
)
from multimodemodel.API_implementation import (
    DomainState,
    GeneralSolver,
    RegularSplitMerger,
    BorderState,
    BorderMerger,
    Tail,
)


def staggered_grid(shape, grid_spacing):
    nx, ny = shape
    dx, dy = grid_spacing
    x = np.arange(0.0, dx * nx, dx)
    y = np.arange(0.0, dy * ny, dy)
    return StaggeredGrid.cartesian_c_grid(x, y)


def initial_condition(staggered_grid, parameter):
    nx, ny = staggered_grid.u.len_x, staggered_grid.u.len_y
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    x, y = staggered_grid.eta.x, staggered_grid.eta.y
    eta = np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / (x.max() // 5) ** 2)

    initial_state = State(
        u=Variable(u, staggered_grid.u),
        v=Variable(v, staggered_grid.v),
        eta=Variable(eta, staggered_grid.eta),
    )

    return DomainState.make_from_State(
        initial_state, history=deque([], maxlen=3), parameter=parameter, it=0, id=0
    )


def non_rotating_swe(state, params):
    rhs = (
        pressure_gradient_i(state, params)
        + pressure_gradient_j(state, params)
        + divergence_i(state, params)
        + divergence_j(state, params)
    )
    return rhs


def get_dt(grid, parameter):
    dx = min(grid.dx.min(), grid.dy.min())
    c = np.sqrt(parameter.H * parameter.g)
    return 0.15 * dx / c


def classic_API(initial_state, dt):
    state = State(initial_state.u, initial_state.v, initial_state.eta)
    for next_state in integrate(
        state,
        initial_state.parameter,
        RHS=non_rotating_swe,
        step=dt,
        time=n_step * dt,
        scheme=adams_bashforth3,
    ):
        pass
    return next_state


def new_API_without_split(initial_state, dt):
    gs = GeneralSolver(solution=non_rotating_swe, schema=adams_bashforth3, step=dt)
    next = initial_state
    for _ in range(n_step):
        next = gs.integration(next)
    return next


def new_API_with_split_no_dask(initial_state, dt, parts=4):
    border_width = 2
    dim = (1,)
    splitter = RegularSplitMerger(parts, dim)
    border_merger = BorderMerger(border_width, dim[0])
    tailor = Tail()
    gs = GeneralSolver(solution=non_rotating_swe, schema=adams_bashforth3, step=dt)

    domain_stack = deque([initial_state.split(splitter)], maxlen=2)
    border_stack = deque(
        [[tailor.make_borders(sub, border_width, dim[0]) for sub in domain_stack[-1]]],
        maxlen=2,
    )
    for _ in range(n_step):
        new_borders = []
        new_subdomains = []
        for i, s in enumerate(domain_stack[-1]):
            new_borders.append(
                (
                    gs.partial_integration(
                        domain=s,
                        border=border_stack[-1][i - 1][1],
                        direction=False,
                        dim=dim[0],
                    ),
                    gs.partial_integration(
                        domain=s,
                        border=border_stack[-1][(i + 1) % (splitter.parts)][0],
                        direction=True,
                        dim=dim[0],
                    ),
                )
            )
        for s, borders in zip(domain_stack[-1], new_borders):
            new_subdomains.append(
                DomainState.merge(
                    (borders[0], gs.integration(s), borders[1]), border_merger
                )
            )
        domain_stack.append(new_subdomains)
        border_stack.append(new_borders)

    return DomainState.merge(domain_stack[-1], splitter)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Exactly one argument required")
        sys.exit(1)
    else:
        command = sys.argv[-1]

    nx, ny = 100, 100
    dx, dy = 1.0, 1.0
    n_step = 500
    parameter = Parameters(H=1.0)
    grid = staggered_grid((nx, ny), (dx, dy))

    if command == "classic":
        out = classic_API(initial_condition(grid, parameter), get_dt(grid.u, parameter))
    elif command == "no_split":
        out = new_API_without_split(
            initial_condition(grid, parameter),
            get_dt(grid.u, parameter),
        )
    elif command == "split":
        out = new_API_with_split_no_dask(
            initial_condition(grid, parameter),
            get_dt(grid.u, parameter),
        )
    else:
        print('Command must be one of "classic", "no_split", "split"')
        sys.exit(1)
    sys.exit(0)