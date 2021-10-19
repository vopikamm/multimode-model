"""Test implementation of distributed API."""
# flake8: noqa
import pytest
import numpy as np
from collections import deque
from copy import copy
from multimodemodel.API_implementation import (
    ParameterSplit,
    DomainState,
    BorderState,
    RegularSplitMerger,
    StateDequeSplit,
    VariableSplit,
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


@pytest.fixture(
    params=(
        # (100, 50),
        (10, 5),
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
    return ParameterSplit(param, data=param.f)


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


def test_ParameterSplit_init(param):
    ps = ParameterSplit(param, data=param.f)
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

    assert all((merged.f[k] == param_split.f[k]).all() for k in param_split.f)
    assert all(
        ~np.may_share_memory(merged.f[k], param_split.f[k]) for k in param_split.f
    )
    assert merged.g == param_split.g
    assert merged.H == param_split.H
    assert merged.rho_0 == param_split.rho_0


def test_ParameterSplit_merge_no_coriolis(param_split, split_merger):
    param_split._f = {}
    others = param_split.split(split_merger)
    merged = ParameterSplit.merge(others, split_merger)
    assert merged.g == param_split.g
    assert merged.H == param_split.H
    assert merged.rho_0 == param_split.rho_0


def test_DomainState_init(state_param):
    state, param = state_param
    ds = DomainState.make_from_State(state, history=deque(), parameter=param, it=0)
    assert (ds.u.data == state.u.data).all()
    assert np.may_share_memory(ds.u.data, state.u.data)
    assert (ds.v.data == state.v.data).all()
    assert np.may_share_memory(ds.v.data, state.v.data)
    assert (ds.eta.data == state.eta.data).all()
    assert np.may_share_memory(ds.eta.data, state.eta.data)


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
            ~np.may_share_memory(getattr(o, v).data, getattr(domain_state, v).data)
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
    assert all(
        (getattr(merged, v).data == getattr(domain_state, v).data).all()
        for v in ("u", "v", "eta")
    )
    assert all(
        (getattr(merged, v).grid.mask == getattr(domain_state, v).grid.mask).all()
        for v in ("u", "v", "eta")
    )


def test_DomainState_merge_param(domain_state, split_merger):
    splitted = domain_state.split(split_merger)
    merged = DomainState.merge(splitted, split_merger)
    assert all(
        getattr(merged.parameter, a) == getattr(domain_state.parameter, a)
        for a in ("g", "H", "rho_0")
    )
    assert all(
        (merged.parameter.f[g] == domain_state.parameter.f[g]).all()
        for g in domain_state.parameter.f
    )


def test_DomainState_copy(domain_state):
    ds_copy = copy(domain_state)
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


def test_BorderState_create_border(state_param, border_direction, dim, request):
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
    # assert not np.may_share_memory(bs.u.data, state.u.data)
    # assert not np.may_share_memory(bs.v.data, state.v.data)
    # assert not np.may_share_memory(bs.eta.data, state.eta.data)
