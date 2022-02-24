"""Test implementation of distributed API."""
# flake8: noqa
from math import prod
import pytest
import numpy as np
from collections import deque
from multimodemodel import (
    Parameter,
    Variable,
    f_constant,
    f_on_sphere,
    beta_plane,
    State,
    Solver,
    BorderMerger,
    BorderSplitter,
    Parameter,
    Grid,
    StaggeredGrid,
    Variable,
    State,
    StateDeque,
    Domain,
    Border,
    RegularSplitMerger,
    Tail,
)
from multimodemodel.integrate import adams_bashforth3, euler_forward
from multimodemodel.util import str_to_date

some_datetime = str_to_date("2000-01-01 00:00")


@pytest.fixture(
    params=(
        # (100, 50),
        (1, 20, 15),
        (5, 20, 15),
    )
)
def staggered_grid(request):
    nz, nx, ny = request.param
    z = np.arange(nz) if nz > 1 else None
    kwargs = dict(
        x=np.arange(nx),
        y=np.arange(ny),
        z=z,
        mask=None,
    )
    return StaggeredGrid.cartesian_c_grid(**kwargs)


@pytest.fixture(
    params=[
        (f_constant, (1.0,)),
        (beta_plane, (10, 0.1, 0.0)),
        (f_on_sphere, ()),
    ]
)
def coriolis_func(request):
    return request.param[0](*request.param[1])


@pytest.fixture
def param(staggered_grid, coriolis_func):
    return Parameter(
        coriolis_func=coriolis_func,
        on_grid=staggered_grid,
    )


@pytest.fixture(params=[1, 2, 3])
def parts(request):
    return request.param


@pytest.fixture(params=[-2, -1])
def dim(request):
    return (request.param,)


@pytest.fixture(params=[RegularSplitMerger])
def split_merger(parts, dim, request):
    return request.param(parts, dim)


@pytest.fixture(params=[True, False])
def state_param(staggered_grid, coriolis_func, request):
    param = Parameter(coriolis_func=coriolis_func, on_grid=staggered_grid)
    u = Variable(
        np.arange(prod(staggered_grid.u.shape)).reshape(staggered_grid.u.shape) + 0.0,
        staggered_grid.u,
        some_datetime,
    )
    v = Variable(
        np.arange(prod(staggered_grid.v.shape)).reshape(staggered_grid.v.shape) + 0.0,
        staggered_grid.v,
        some_datetime,
    )
    eta = Variable(
        np.arange(prod(staggered_grid.eta.shape)).reshape(staggered_grid.eta.shape)
        + 0.0,
        staggered_grid.eta,
        some_datetime,
    )
    if request.param:
        eta.data = None
    return State(u=u, v=v, eta=eta), param


@pytest.fixture(params=[0, 99])
def ident(request):
    return request.param


@pytest.fixture
def domain_state(state_param):
    state, param = state_param
    return Domain(
        state=state,
        history=StateDeque(),
        parameter=param,
        iteration=0,
        id=0,
    )


@pytest.fixture(params=[-1, 0, 99])
def it(request):
    return request.param


@pytest.fixture(params=[False, True])
def border_direction(request):
    return request.param


@pytest.fixture
def border_state(domain_state):
    return Border(
        state=domain_state.state,
        history=domain_state.history,
        iteration=domain_state.iteration,
        id=domain_state.id,
        width=2,
        dim=-1,
    )


@pytest.fixture(params=(1, 2, 3))
def dt(request):
    return request.param


@pytest.fixture(params=(euler_forward, adams_bashforth3))
def scheme(request):
    return request.param


def test_Parameter_split(param, split_merger):
    out = param.split(split_merger)
    assert len(out) == split_merger.parts
    assert all(isinstance(o, param.__class__) for o in out)
    assert all(o.g == param.g for o in out)
    assert all(o.H == param.H for o in out)
    assert all(o.rho_0 == param.rho_0 for o in out)
    assert all((g in o.f for o in out) for g in ("u", "v", "eta", "q"))
    assert all(
        (split_merger.merge_array(tuple(o.f[g] for o in out)) == param.f[g]).all()
        for g in ("u", "v", "eta", "q")
    )
    tuple(
        (_not_share_memory(o.f[g], param.f[g]) for g in ("u", "v", "eta", "q"))
        for o in out
    )
    param = Parameter._new_with_data(param, {})
    out = param.split(split_merger)
    assert all(o is param for o in out)
    assert len(out) == split_merger.parts


def test_Parameter_merge(param, split_merger):
    others = param.split(split_merger)

    merged = Parameter.merge(others, split_merger)

    assert param == merged


def test_Parameter_merge_no_coriolis(param, split_merger):
    param = Parameter._new_with_data(param, {})
    others = param.split(split_merger)
    merged = Parameter.merge(others, split_merger)
    assert merged.g == param.g
    assert merged.H == param.H
    assert merged.rho_0 == param.rho_0


def test_Grid_split_merge_roundtrip(staggered_grid, split_merger):
    grid = staggered_grid.u
    sm_roundtrip = Grid.merge(grid.split(split_merger), split_merger)
    assert grid == sm_roundtrip


def test_StaggeredGrid_split_merge_roundtrip(staggered_grid, split_merger):
    sm_roundtrip = StaggeredGrid.merge(staggered_grid.split(split_merger), split_merger)
    assert staggered_grid == sm_roundtrip


def test_Variable_split_None_data_into_tuple_of_None(state_param, split_merger):
    var = state_param[0].u
    var.data = None
    splitted_var = var.split(split_merger)
    assert tuple(s.data for s in splitted_var) == split_merger.parts * (None,)


def test_Variable_merge_none_properly_treated(state_param, split_merger):
    var = state_param[0].u
    var_split = var.split(split_merger)
    replace_data = np.zeros_like(var_split[0].safe_data)
    var_split[0].data = None
    sm_var = Variable.merge(var_split, split_merger)
    oracle = split_merger.merge_array(
        [replace_data if i == 0 else vs.data for i, vs in enumerate(var_split)]
    )
    if split_merger.parts == 1:
        assert None == sm_var.data
    else:
        assert (sm_var.data == oracle).all()


def test_State_split_merge_roundtrip(state_param, split_merger):
    state = state_param[0]
    sm_roundtrip = State.merge(state.split(split_merger), split_merger)
    assert state == sm_roundtrip


def test_StateDeque_split_State_in_history(domain_state, split_merger):
    hist_state = domain_state.state
    domain_state.history.append(hist_state)
    splitted_hist_states = hist_state.split(split_merger)
    out = domain_state.split(split_merger)
    assert all(
        splitted_hist_state == o.history[-1]
        for splitted_hist_state, o in zip(splitted_hist_states, out)
    )


def test_Domain_split(domain_state, split_merger):
    out = domain_state.split(split_merger)
    assert len(out) == split_merger.parts
    assert all(isinstance(o, domain_state.__class__) for o in out)
    assert State.merge(tuple(o.state for o in out), split_merger) == domain_state.state
    tuple(
        (
            _not_share_memory(
                getattr(o.state, v).data, getattr(domain_state.state, v).data
            )
            if getattr(domain_state.state, v).data is not None
            else getattr(o.state, v).data is None
            for v in ("u", "v", "eta")
        )
        for o in out
    )


def test_Domain_split_splits_params(domain_state, split_merger):
    out = domain_state.split(split_merger)
    out_params = domain_state.parameter.split(split_merger)
    assert all(
        (o.p.f[g] == o_p.f[g] for g in domain_state.parameter.f)
        for o, o_p in zip(out, out_params)
    )


def test_Domain_split_splits_history(domain_state, split_merger):
    hist_state = domain_state.state
    domain_state.history.append(hist_state)
    splitted_hist_states = hist_state.split(split_merger)
    out = domain_state.split(split_merger)
    assert all(
        splitted_hist_state == o.history[-1]
        for splitted_hist_state, o in zip(splitted_hist_states, out)
    )


def test_Domain_split_conserves_iteration_counter(domain_state, it, split_merger):
    domain_state.iteration = it
    out = domain_state.split(split_merger)
    assert all(o.iteration == it for o in out)


def test_Domain_merge_raises_on_iteration_counter_discrepancy(
    domain_state, split_merger
):
    out = domain_state.split(split_merger)
    out[0].iteration = 6
    if split_merger.parts > 1:
        with pytest.raises(
            ValueError,
            match="Try to merge Domains that differ in iteration counter.",
        ):
            _ = Domain.merge(out, split_merger)


def test_Domain_split_preserves_id(domain_state, split_merger, ident):
    domain_state.id = ident
    out = domain_state.split(split_merger)
    assert all(ident == o.id for o in out)


def test_Domain_merge(domain_state, split_merger):
    splitted = domain_state.split(split_merger)
    merged = Domain.merge(splitted, split_merger)
    merged == domain_state


def test_Domain_merge_param(domain_state, split_merger):
    splitted = domain_state.split(split_merger)
    merged = Domain.merge(splitted, split_merger)
    assert merged.parameter == domain_state.parameter


def test_BorderState_init(state_param):
    width, dim = 1, 0
    state, param = state_param
    bs = Border(
        state,
        history=StateDeque(),
        parameter=param,
        width=width,
        dim=0,
        iteration=0,
        id=1,
    )
    if state.u.data is None:
        assert bs.state.u.data is None
    else:
        assert (bs.state.u.data == state.u.data).all()
        _may_share_memory(bs.state.u.data, state.u.data)
    if state.v.data is None:
        assert bs.state.v.data is None
    else:
        assert (bs.state.v.data == state.v.data).all()
        _may_share_memory(bs.state.v.data, state.v.data)
    if state.eta.data is None:
        assert bs.state.eta.data is None
    else:
        assert (bs.state.eta.data == state.eta.data).all()
        _may_share_memory(bs.state.eta.data, state.eta.data)
    assert bs.width == width
    assert bs.dim == dim


def test_BorderState_get_width_returns_width(border_state):
    assert border_state.get_width() == border_state.width


def test_BorderState_get_dim_returns_dim(border_state):
    assert border_state.get_dim() == border_state.dim


def test_BorderState_create_border(state_param, border_direction, dim):
    width = 2
    dim = dim[0]
    direction = border_direction
    state, param = state_param

    if direction:
        b_slice = slice(-width, None)
    else:
        b_slice = slice(width)

    b_slices = state.u.data.ndim * [slice(None)]
    b_slices[dim] = b_slice
    b_slices = tuple(b_slices)
    ds = Domain(state, history=StateDeque(), parameter=param, iteration=0)
    bs = Border.create_border(ds, width=width, direction=direction, dim=dim)
    if state.u.data is None:
        assert bs.state.u.data is None
    else:
        assert (bs.state.u.data == state.u.data[b_slices]).all()
    if state.v.data is None:
        assert bs.state.v.data is None
    else:
        assert (bs.state.v.data == state.v.data[b_slices]).all()
    if state.eta.data is None:
        assert bs.state.eta.data is None
    else:
        assert (bs.state.eta.data == state.eta.data[b_slices]).all()


def _may_share_memory(a1, a2):
    if a1 is not None and a2 is not None:
        assert np.may_share_memory(a1, a2)


def _not_share_memory(a1, a2):
    if a1 is not None and a2 is not None:
        assert not np.may_share_memory(a1, a2)


def test_BorderState_create_border_returns_reference(
    state_param, border_direction, dim
):
    width = 2
    dim = dim[0]
    direction = border_direction
    state, param = state_param

    ds = Domain(state, history=None, parameter=param, iteration=0)
    bs = Border.create_border(ds, width=width, direction=direction, dim=dim)
    _may_share_memory(bs.state.u.data, state.u.data)
    _may_share_memory(bs.state.v.data, state.v.data)
    _may_share_memory(bs.state.eta.data, state.eta.data)
    _may_share_memory(bs.state.eta.grid.x, state.eta.grid.x)


def test_Tail_split_domain_sets_ids_correctly(domain_state, split_merger):
    splitted = Tail.split_domain(domain_state, split_merger)
    assert all(s.id == i for i, s in enumerate(splitted))


def test_Tail_make_borders_returns_two_borders(domain_state):
    width = 2
    dim = -1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    assert type(borders) is tuple
    assert len(borders) == 2


def test_Tail_stitch_correctly_stitch(domain_state):
    width = 2
    dim = -1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    stitched_domain = t.stitch(base=domain_state, borders=borders)
    assert domain_state == stitched_domain


def test_Tail_stitch_returns_copy(domain_state):
    width = 2
    dim = -1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    stitched_domain = t.stitch(base=domain_state, borders=borders)
    tuple(
        _not_share_memory(
            getattr(domain_state.state, v).data, getattr(stitched_domain.state, v).data
        )
        for v in ("u", "v", "eta")
    )


def test_Tail_stitch_raise_ValueError_on_iteration_mismatch(domain_state):
    width = 2
    dim = -1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    domain_state.iteration += 1
    with pytest.raises(ValueError, match="Borders iteration mismatch"):
        _ = t.stitch(base=domain_state, borders=borders)


def rhs(state: State, _: Parameter) -> State:
    it = 1
    return State(
        u=Variable((it + 1) * state.u.safe_data, state.u.grid, time=state.u.time),
        v=Variable((it + 1) * state.v.safe_data, state.v.grid, time=state.v.time),
        eta=Variable(
            (it + 1) * state.eta.safe_data, state.eta.grid, time=state.eta.time
        ),
    )


def test_Solver_integrate_appends_inc_to_history(domain_state):
    inc = rhs(domain_state.state, domain_state.parameter)
    gs = Solver(rhs=rhs, ts_schema=euler_forward, step=1)
    next = gs.integrate(domain_state)
    assert next.history[-1] == inc


def test_Solver_integrate_has_no_side_effects_on_domain(domain_state):
    gs = Solver(rhs=rhs, ts_schema=euler_forward, step=1)
    next = gs.integrate(domain_state)
    assert next.history != domain_state.history


def test_Solver_integrate(domain_state, dt, scheme):
    inc = rhs(domain_state.state, domain_state.parameter)
    gs = Solver(rhs=rhs, ts_schema=scheme, step=dt)
    next = gs.integrate(domain_state)

    assert all(
        (
            getattr(domain_state.state, v).safe_data + dt * getattr(inc, v).safe_data
            == getattr(next.state, v).safe_data
        ).all()
        for v in ("u", "v", "eta")
    )
    assert all(
        getattr(domain_state.state, v).grid == getattr(next.state, v).grid
        for v in ("u", "v", "eta")
    )


def test_Solver_integrate_get_border_width_returns_2():
    gs = Solver(rhs=rhs, ts_schema=euler_forward, step=1)
    assert gs.get_border_width() == 2


def test_Solver_integrate_border(domain_state, dt, dim, parts, scheme):
    border_width = 2
    splitter = RegularSplitMerger(parts, dim)
    border_merger = BorderMerger(border_width, dim[0])
    tailor = Tail()
    gs1 = Solver(rhs=rhs, ts_schema=scheme, step=dt)
    gs2 = Solver(rhs=rhs, ts_schema=scheme, step=dt)

    next = gs1.integrate(domain_state)
    next_2 = gs1.integrate(next)
    next_3 = gs1.integrate(next_2)

    domain_stack = deque([domain_state.split(splitter)], maxlen=2)
    border_stack = deque(
        [[tailor.make_borders(sub, border_width, dim[0]) for sub in domain_stack[-1]]],
        maxlen=2,
    )
    for _ in range(3):
        new_borders = []
        new_subdomains = []
        for i, s in enumerate(domain_stack[-1]):
            new_borders.append(
                (
                    gs2.integrate_border(
                        border=border_stack[-1][i][0],
                        domain=s,
                        neighbor_border=border_stack[-1][i - 1][1],
                        direction=False,
                    ),
                    gs2.integrate_border(
                        border=border_stack[-1][i][1],
                        domain=s,
                        neighbor_border=border_stack[-1][(i + 1) % (splitter.parts)][0],
                        direction=True,
                    ),
                )
            )
        for s, borders in zip(domain_stack[-1], new_borders):
            new_subdomains.append(
                Domain.merge((borders[0], gs2.integrate(s), borders[1]), border_merger)
            )
            print(new_subdomains[-1].iteration)
        domain_stack.append(new_subdomains)
        border_stack.append(new_borders)

    final = Domain.merge(domain_stack[-1], splitter)
    assert final == next_3


def test_BorderSplitter__eq__():
    start, stop, axis = 1, 2, 3
    bs1 = BorderSplitter(slice(start, stop), axis)
    bs2 = BorderSplitter(slice(start, stop), axis)
    bs3 = BorderSplitter(slice(start, stop + 1), axis)
    bs4 = BorderSplitter(slice(start, stop), axis + 1)
    assert bs1 == bs2
    assert bs1 != bs3
    assert bs1 != bs4


def test_BorderMerger__eq__():
    width, axis = 1, 2
    bm1 = BorderMerger(width, axis)
    bm2 = BorderMerger(width, axis)
    bm3 = BorderMerger(width + 1, axis)
    bm4 = BorderMerger(width, axis + 1)
    assert bm1 == bm2
    assert bm1 != bm3
    assert bm1 != bm4


def test_RegularSplitMerger__eq__():
    parts, dim = 1, (2,)
    sm1 = RegularSplitMerger(parts, dim)
    sm2 = RegularSplitMerger(parts, dim)
    sm3 = RegularSplitMerger(parts + 1, dim)
    sm4 = RegularSplitMerger(parts, (1,))
    assert sm1 == sm2
    assert sm1 != sm3
    assert sm1 != sm4
