"""Test implementation of distributed API."""
# flake8: noqa
import pytest
import numpy as np
from collections import deque
from copy import copy, deepcopy
from multimodemodel.API_implementation import (
    GeneralSolver,
    ParameterSplit,
    GridSplit,
    StaggeredGridSplit,
    VariableSplit,
    StateSplit,
    StateDequeSplit,
    DomainState,
    BorderState,
    RegularSplitMerger,
    BorderSplitter,
    Tail,
)
from multimodemodel import (
    Parameters,
    StaggeredGrid,
    Variable,
    f_constant,
    f_on_sphere,
    beta_plane,
    State,
)
from multimodemodel.integrate import euler_forward


@pytest.fixture(
    params=(
        # (100, 50),
        (20, 10),
    )
)
def staggered_grid(request):
    nx, ny = request.param
    return StaggeredGrid.cartesian_c_grid(
        np.arange(nx),
        np.arange(ny),
        mask=None,
    )


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
    return Parameters(
        coriolis_func=coriolis_func,
        on_grid=staggered_grid,
    )


@pytest.fixture
def param_split(param):
    return ParameterSplit.from_parameters(param)


@pytest.fixture(params=[1, 2])
def parts(request):
    return request.param


@pytest.fixture(params=[0, 1])
def dim(request):
    return (request.param,)


@pytest.fixture(params=[RegularSplitMerger])
def split_merger(parts, dim, request):
    return request.param(parts, dim)


@pytest.fixture
def state_param(staggered_grid, coriolis_func):
    param = Parameters(coriolis_func=coriolis_func, on_grid=staggered_grid)
    u = Variable(
        np.arange(staggered_grid.u.x.size).reshape(staggered_grid.u.x.shape) + 0.0,
        staggered_grid.u,
    )
    v = Variable(
        np.arange(staggered_grid.v.x.size).reshape(staggered_grid.v.x.shape) + 1.0,
        staggered_grid.v,
    )
    eta = Variable(
        np.arange(staggered_grid.eta.x.size).reshape(staggered_grid.eta.x.shape) + 2.0,
        staggered_grid.eta,
    )
    return State(u=u, v=v, eta=eta), param


@pytest.fixture(params=[0, 99])
def ident(request):
    return request.param


@pytest.fixture
def domain_state(state_param):
    state, param = state_param
    return DomainState.make_from_State(state, history=deque(), parameter=param, it=0)


@pytest.fixture(params=[-1, 0, 99])
def it(request):
    return request.param


@pytest.fixture(params=[False, True])
def border_direction(request):
    return request.param


@pytest.fixture
def border_state(domain_state):
    return BorderState(
        u=domain_state.u,
        v=domain_state.v,
        eta=domain_state.eta,
        history=domain_state.history,
        iteration=domain_state.it,
        id=domain_state.id,
        width=2,
        dim=-1,
    )


@pytest.fixture(params=(1, 2, 3))
def dt(request):
    return request.param


def test_ParameterSplit_from_parameters(param):
    ps = ParameterSplit.from_parameters(param)
    assert ps.g == param.g
    assert ps.H == param.H
    assert ps.rho_0 == param.rho_0
    assert ps.f == param.f


def test_ParameterSplit_split(param_split, split_merger):
    out = param_split.split(split_merger)
    assert len(out) == split_merger.parts
    assert all(isinstance(o, param_split.__class__) for o in out)
    assert all(o.g == param_split.g for o in out)
    assert all(o.H == param_split.H for o in out)
    assert all(o.rho_0 == param_split.rho_0 for o in out)
    assert all((g in o.f for o in out) for g in ("u", "v", "eta", "q"))
    assert all(
        (split_merger.merge_array(tuple(o.f[g] for o in out)) == param_split.f[g]).all()
        for g in ("u", "v", "eta", "q")
    )
    assert all(
        (~np.may_share_memory(o.f[g], param_split.f[g]) for g in ("u", "v", "eta", "q"))
        for o in out
    )
    param_split._f = {}
    out = param_split.split(split_merger)
    assert all(o is param_split for o in out)
    assert len(out) == split_merger.parts


def test_ParameterSplit_merge(param_split, split_merger):
    others = param_split.split(split_merger)

    merged = ParameterSplit.merge(others, split_merger)

    assert param_split == merged


def test_ParameterSplit_merge_no_coriolis(param_split, split_merger):
    param_split._f = {}
    others = param_split.split(split_merger)
    merged = ParameterSplit.merge(others, split_merger)
    assert merged.g == param_split.g
    assert merged.H == param_split.H
    assert merged.rho_0 == param_split.rho_0


def test_GridSplit_split_merge_roundtrip(staggered_grid, split_merger):
    grid = GridSplit.from_grid(staggered_grid.u)
    sm_roundtrip = GridSplit.merge(grid.split(split_merger), split_merger)
    assert grid == sm_roundtrip


def test_StaggeredGrid_init(staggered_grid):
    splittable = StaggeredGridSplit.from_staggered_grid(staggered_grid)
    assert all(
        isinstance(getattr(splittable, g), GridSplit) for g in ("u", "v", "eta", "q")
    )


def test_StaggeredGrid_split_merge_roundtrip(staggered_grid, split_merger):
    splittable = StaggeredGridSplit.from_staggered_grid(staggered_grid)
    sm_roundtrip = StaggeredGridSplit.merge(
        splittable.split(split_merger), split_merger
    )
    assert splittable == sm_roundtrip


def test_VariableSplit_init(state_param):
    state, _ = state_param
    var = state.u
    assert not isinstance(var.grid, GridSplit)
    v_splittable = VariableSplit(var.data, var.grid)
    assert isinstance(v_splittable.grid, GridSplit)


def test_VariableSplit_split_None_data_into_tuple_of_None(state_param, split_merger):
    var = state_param[0].u
    var.data = None
    var = VariableSplit.from_variable(var)
    splitted_var = var.split(split_merger)
    assert tuple(s.data for s in splitted_var) == split_merger.parts * (None,)


def test_VariableSplit_merge_none_properly_treated(state_param, split_merger):
    var = state_param[0].u
    var = VariableSplit.from_variable(var)
    var_split = var.split(split_merger)
    replace_data = np.zeros_like(var_split[0].data)
    var_split[0].data = None
    sm_var = VariableSplit.merge(var_split, split_merger)
    oracle = split_merger.merge_array(
        [replace_data if i == 0 else vs.data for i, vs in enumerate(var_split)]
    )
    if split_merger.parts == 1:
        assert None == sm_var.data
    else:
        assert (sm_var.data == oracle).all()


def test_StateSplit_split_merge_roundtrip(state_param, split_merger):
    splittable_state = StateSplit.from_state(state_param[0])
    sm_roundtrip = StateSplit.merge(splittable_state.split(split_merger), split_merger)
    assert splittable_state == sm_roundtrip


def test_DomainState_init(state_param):
    state, param = state_param
    ds = DomainState.make_from_State(state, history=deque(), parameter=param, it=0)
    assert (ds.u.data == state.u.data).all()
    assert np.may_share_memory(ds.u.data, state.u.data)
    assert (ds.v.data == state.v.data).all()
    assert np.may_share_memory(ds.v.data, state.v.data)
    assert (ds.eta.data == state.eta.data).all()
    assert np.may_share_memory(ds.eta.data, state.eta.data)


def test_DomainState_init_None_history_to_empty_StateDequeSplit(state_param):
    state, param = state_param
    ds = DomainState(state.u, state.v, state.eta, None, param)
    assert isinstance(ds.history, deque)
    assert len(ds.history) == 0
    assert ds.history.maxlen == 3


def test_DomainState_set_id(domain_state, ident):
    domain_state.set_id(ident)
    assert domain_state.id == ident


def test_DomainState_get_id(domain_state, ident):
    domain_state.id = ident
    assert ident == domain_state.get_id()


def test_DomainState_get_iteration(domain_state, it):
    domain_state.it = it
    assert domain_state.get_iteration() == it


def test_DomainState_get_data(domain_state):
    data = domain_state.get_data()
    assert len(data) == 3  # three variables
    assert all(isinstance(d, Variable) for d in data)
    assert data[0] is domain_state.u
    assert data[1] is domain_state.v
    assert data[2] is domain_state.eta


def test_DomainState_increment_iteration(domain_state, it):
    domain_state.it = it
    assert domain_state.increment_iteration() == it + 1


def test_DomainState_split(domain_state, split_merger):
    out = domain_state.split(split_merger)
    assert len(out) == split_merger.parts
    assert all(isinstance(o, domain_state.__class__) for o in out)
    assert all(
        (
            VariableSplit.merge(tuple(getattr(o, v) for o in out), split_merger).data
            == getattr(domain_state, v).data
        ).all()
        for v in ("u", "v", "eta")
    )
    assert all(
        (
            not np.may_share_memory(getattr(o, v).data, getattr(domain_state, v).data)
            for v in ("u", "v", "eta")
        )
        for o in out
    )


def test_DomainState_split_splits_params(domain_state, split_merger):
    out = domain_state.split(split_merger)
    out_params = domain_state.parameter.split(split_merger)
    assert all(
        (o.p.f[g] == o_p.f[g] for g in domain_state.parameter.f)
        for o, o_p in zip(out, out_params)
    )


def test_DomainState_split_splits_history(domain_state, split_merger):
    hist_state = State(u=domain_state.u, v=domain_state.v, eta=domain_state.eta)
    domain_state.history.append(hist_state)
    splitted_hist_states = StateSplit.from_state(hist_state).split(split_merger)
    out = domain_state.split(split_merger)
    assert all(
        (
            (getattr(splitted_hist_state, v) == getattr(o.history[-1], v)).all()
            for v in ("u", "v", "eta")
        )
        for splitted_hist_state, o in zip(splitted_hist_states, out)
    )


def test_DomainState_split_conserves_iteration_counter(
    domain_state, it, split_merger, request
):
    domain_state.it = it
    out = domain_state.split(split_merger)
    assert all(o.get_iteration() == it for o in out)


def test_DomainState_merge_raises_on_iteration_counter_discrepancy(
    domain_state, split_merger
):
    out = domain_state.split(split_merger)
    out[0].it = 6
    if split_merger.parts > 1:
        with pytest.raises(
            ValueError,
            match="Try to merge DomainStates that differ in iteration counter.",
        ):
            _ = DomainState.merge(out, split_merger)


def test_DomainState_merge(domain_state, split_merger):
    splitted = domain_state.split(split_merger)
    merged = DomainState.merge(splitted, split_merger)
    merged == domain_state


def test_DomainState_merge_param(domain_state, split_merger):
    splitted = domain_state.split(split_merger)
    merged = DomainState.merge(splitted, split_merger)
    assert merged.parameter == domain_state.parameter


def test_DomainState_copy(domain_state):
    ds_copy = copy(domain_state)
    assert ds_copy == domain_state
    assert id(ds_copy) is not id(domain_state)
    assert all(
        id(getattr(ds_copy, v)) is not id(getattr(domain_state, v))
        for v in ("u", "v", "eta")
    )
    assert all(
        ~np.may_share_memory(getattr(ds_copy, v).data, getattr(domain_state, v).data)
        for v in ("u", "v", "eta")
    )
    assert all(
        id(getattr(ds_copy, v).grid) is not id(getattr(domain_state, v).grid)
        for v in ("u", "v", "eta")
    )
    assert all(
        (
            ~np.may_share_memory(
                getattr(getattr(ds_copy, v).grid, c),
                getattr(getattr(domain_state, v).grid, c),
            )
            for c in ("x", "y", "mask")
        )
        for v in ("u", "v", "eta")
    )


def test_DomainState_comparison_with_identical_returns_true(domain_state):
    d2 = domain_state
    assert domain_state == d2


def test_DomainState_comparison_with_same_returns_true(domain_state):
    d2 = domain_state.copy()
    assert domain_state == d2


def test_DomainState_comparison_with_different_returns_false(domain_state):
    d2 = domain_state.copy()
    d2.id = 100
    assert domain_state != d2


def test_DomainState_comparison_with_wrong_type_returns_false(domain_state):
    assert domain_state != 5


def test_BorderState_init(state_param):
    width, dim = 1, 0
    state, param = state_param
    bs = BorderState(
        state.u,
        state.v,
        state.eta,
        history=StateDequeSplit(),
        parameter=param,
        width=width,
        dim=0,
        iteration=0,
        id=1,
    )
    assert (bs.u.data == state.u.data).all()
    assert np.may_share_memory(bs.u.data, state.u.data)
    assert (bs.v.data == state.v.data).all()
    assert np.may_share_memory(bs.v.data, state.v.data)
    assert (bs.eta.data == state.eta.data).all()
    assert np.may_share_memory(bs.eta.data, state.eta.data)
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
    b_slices = state.u.data.ndim * [
        slice(None),
    ]
    b_slices[dim] = b_slice
    b_slices = tuple(b_slices)
    ds = DomainState.make_from_State(state, history=deque(), parameter=param, it=0)
    bs = BorderState.create_border(ds, width=width, direction=direction, dim=dim)
    assert np.allclose(bs.u.data, state.u.data[b_slices])
    assert (bs.v.data == state.v.data[b_slices]).all()
    assert (bs.eta.data == state.eta.data[b_slices]).all()


def test_BorderState_create_border_returns_copies(state_param, border_direction, dim):
    width = 2
    dim = dim[0]
    direction = border_direction
    state, param = state_param

    ds = DomainState.make_from_State(state, history=deque(), parameter=param, it=0)
    bs = BorderState.create_border(ds, width=width, direction=direction, dim=dim)
    assert not np.may_share_memory(bs.u.data, state.u.data)
    assert not np.may_share_memory(bs.v.data, state.v.data)
    assert not np.may_share_memory(bs.eta.data, state.eta.data)


def test_Tail_make_borders_returns_two_borders(domain_state):
    width = 2
    dim = 1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    assert type(borders) is tuple
    assert len(borders) == 2


def test_Tail_stitch_correctly_stitch(domain_state):
    width = 2
    dim = 1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    stitched_domain = t.stitch(base=domain_state, borders=borders, dims=(dim,))
    assert domain_state == stitched_domain


def test_Tail_stitch_returns_copy(domain_state):
    width = 2
    dim = 1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    stitched_domain = t.stitch(base=domain_state, borders=borders, dims=(dim,))
    assert all(
        (
            not np.may_share_memory(
                getattr(domain_state, v).data, getattr(stitched_domain, v).data
            )
            for v in ("u", "v", "eta")
        )
    )


def test_Tail_stitch_raise_ValueError_on_iteration_mismatch(domain_state):
    width = 2
    dim = 1
    t = Tail()
    borders = t.make_borders(domain_state, width, dim)
    domain_state.it += 1
    with pytest.raises(ValueError, match="Borders iteration mismatch"):
        _ = t.stitch(base=domain_state, borders=borders, dims=(dim,))


def rhs(state, _):
    new = deepcopy(state)
    new.u.data = np.ones(new.u.grid.x.shape)
    new.v.data = np.ones(new.v.grid.x.shape)
    new.eta.data = np.ones(new.eta.grid.x.shape)
    return new


def test_GeneralSolver_integration_appends_inc_to_history(domain_state):
    inc = rhs(domain_state, None)
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=1)
    next = gs.integration(domain_state)
    assert next.history[-1] == inc


def test_GeneralSolver_integration_has_no_side_effects_on_domain(domain_state):
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=1)
    next = gs.integration(domain_state)
    assert next.history != domain_state.history


def test_GeneralSolver_integration_with_euler_forward(domain_state, dt):
    inc = rhs(domain_state, None)
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=dt)
    next = gs.integration(domain_state)

    assert all(
        (
            getattr(domain_state, v).safe_data + dt * getattr(inc, v).safe_data
            == getattr(next, v).safe_data
        ).all()
        for v in ("u", "v", "eta")
    )
    assert all(
        getattr(domain_state, v).grid == getattr(next, v).grid
        for v in ("u", "v", "eta")
    )


def test_GeneralSolver_integration_get_border_width_returns_2():
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=1)
    assert gs.get_border_width() == 2


def test_GeneralSolver_partial_integration(domain_state, dt):
    dim = 1
    splitter = RegularSplitMerger(2, (dim,))
    tailor = Tail()
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=dt)

    next = gs.integration(domain_state)
    next_sub_domains = next.split(splitter)
    next_borders = [tailor.make_borders(sub, 2, dim) for sub in next_sub_domains]

    sub_domains = domain_state.split(splitter)
    borders = [tailor.make_borders(sub, 2, dim) for sub in sub_domains]
    integrate_borders = []
    for i, s in enumerate(sub_domains):
        new_left_border = gs.partial_integration(
            domain=s, border=borders[i - 1][1], direction=False, dim=dim
        )
        new_right_border = gs.partial_integration(
            domain=s,
            border=borders[(i + 1) % (splitter.parts)][0],
            direction=True,
            dim=dim,
        )
        integrate_borders.append((new_left_border, new_right_border))

    for (new_l, new_r), (next_l, next_r) in zip(integrate_borders, next_borders):
        # assert new_l == next_l
        # assert new_r == next_r
        for v in ("u", "v", "eta"):
            assert getattr(new_l, v) == getattr(next_l, v)
            assert getattr(new_r, v) == getattr(next_r, v)
        assert new_l.history == next_l.history
        assert new_r.history == next_r.history
        assert new_l.parameter == next_l.parameter
        assert new_r.parameter == next_r.parameter


def test_GeneralSolver_window_returns_argument(domain_state, dt):
    gs = GeneralSolver(solution=rhs, schema=euler_forward, step=dt)
    assert gs.window(domain_state, None) is domain_state