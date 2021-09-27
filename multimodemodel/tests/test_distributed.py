"""Test implementation of distributed API."""
# flake8: noqa
from multimodemodel import domain_split_API
from multimodemodel.domain_split_API import Border, Domain
import pytest
import numpy as np
from collections import deque
from copy import copy
from multimodemodel.API_implementation import (
    ParameterSplit,
    DomainState,
    BorderState,
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


def test_ParameterSplit_init(param):
    ps = ParameterSplit(param, data=param.f)
    assert ps.g == param.g
    assert ps.H == param.H
    assert ps.rho_0 == param.rho_0
    assert ps.f == param.f


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
    return DomainState.make_from_State(state, h=deque(), p=param, it=0)


@pytest.fixture(params=[-1, 0, 99])
def it(request):
    return request.param


@pytest.fixture(params=[False, True])
def border_direction(request):
    return request.param


def test_ParameterSplit_split(param_split, parts, dim):
    out = param_split.split(parts, dim)
    assert len(out) == parts
    assert all(isinstance(o, ParameterSplit) for o in out)
    assert all(o.g == param_split.g for o in out)
    assert all(o.H == param_split.H for o in out)
    assert all(o.rho_0 == param_split.rho_0 for o in out)
    assert all((g in o.f for o in out) for g in ("u", "v", "eta", "q"))
    assert all(
        (
            (
                np.concatenate(tuple(o.f[g] for o in out), axis=dim[0])
                == param_split.f[g]
            ).all()
            for g in ("u", "v", "eta", "q")
        )
    )
    assert all(
        (~np.may_share_memory(o.f[g], param_split.f[g]) for g in ("u", "v", "eta", "q"))
        for o in out
    )
    param_split._f = {}
    out = param_split.split(parts, dim)
    assert all(o is param_split for o in out)
    assert len(out) == parts


def test_ParameterSplit_merge(param_split, parts, dim):
    others = param_split.split(parts, dim)

    merged = ParameterSplit.merge(others, dim[0])

    assert all((merged.f[k] == param_split.f[k]).all() for k in param_split.f)
    assert all(
        ~np.may_share_memory(merged.f[k], param_split.f[k]) for k in param_split.f
    )
    assert merged.g == param_split.g
    assert merged.H == param_split.H
    assert merged.rho_0 == param_split.rho_0


def test_ParameterSplit_merge_no_coriolis(param_split):
    param_split._f = {}
    dim = (0,)
    parts = 2
    others = param_split.split(parts, dim)
    merged = ParameterSplit.merge(others, dim[0])
    assert merged.g == param_split.g
    assert merged.H == param_split.H
    assert merged.rho_0 == param_split.rho_0


def test_DomainState_init(state_param):
    state, param = state_param
    ds = DomainState.make_from_State(state, h=deque(), p=param, it=0)
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


def test_DomainState_split(domain_state, parts, dim):
    out = domain_state.split(parts, dim)
    assert len(out) == parts
    assert all(isinstance(o, domain_state.__class__) for o in out)
    assert all(
        (
            (
                np.concatenate(tuple(getattr(o, v).data for o in out), axis=dim[0])
                == getattr(domain_state, v).data
            ).all()
            for v in ("u", "v", "eta")
        )
    )
    assert all(
        (
            ~np.may_share_memory(getattr(o, v).data, getattr(domain_state, v).data)
            for v in ("u", "v", "eta")
        )
        for o in out
    )


@pytest.mark.xfail(reason="Parameter object is not splitted")
def test_DomainState_split_splits_params(domain_state):
    out = domain_state.split(2, (0,))
    out_params = domain_state.p.split(2, (0,))
    assert all(
        (o.p.f[g] == o_p.f[g] for g in domain_state.p.f)
        for o, o_p in zip(out, out_params)
    )


def test_DomainState_split_conserves_iteration_counter(domain_state, it, request):
    if it != 0:
        request.node.add_marker(
            pytest.mark.xfail(reason="Iteration counter not passed through in split.")
        )
    domain_state.it = it
    out = domain_state.split(2, (0,))
    assert all(o.get_iteration() == it for o in out)


def test_DomainState_merge(domain_state, parts, dim):
    splitted = domain_state.split(parts, dim)
    merged = DomainState.merge(splitted, dim[0])
    assert all(
        (getattr(merged, v).data == getattr(domain_state, v).data).all()
        for v in ("u", "v", "eta")
    )
    assert all(
        (getattr(merged, v).grid.mask == getattr(domain_state, v).grid.mask).all()
        for v in ("u", "v", "eta")
    )


@pytest.mark.xfail(reason="parameters are not splitted")
def test_DomainState_merge_param(domain_state, parts, dim):
    splitted = domain_state.split(parts, dim)
    merged = DomainState.merge(splitted, dim[0])
    assert all(
        getattr(merged.p, a) == getattr(domain_state.p, a) for a in ("g", "H", "rho_0")
    )
    assert all(
        (getattr(merged.p.f, g) == getattr(domain_state.p.f, g)).all()
        for g in domain_state.p.f
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
    width = 1
    state, param = state_param
    bs = BorderState(
        state.u,
        state.v,
        state.eta,
        ancestors=deque(),
        p=param,
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


def test_BorderState_create_border(state_param, border_direction, dim, request):
    width = 2
    dim = dim[0]
    if dim == 0:
        request.node.add_marker(
            pytest.mark.xfail(reason="only splitting along last dimension implemented!")
        )
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
    ds = DomainState.make_from_State(state, h=deque(), p=param, it=0)
    bs = BorderState.create_border(ds, width=width, direction=direction, dim=dim)
    assert np.allclose(bs.u.data, state.u.data[b_slices])
    assert not np.may_share_memory(bs.u.data, state.u.data)
    assert (bs.v.data == state.v.data[b_slices]).all()
    assert not np.may_share_memory(bs.v.data, state.v.data)
    assert (bs.eta.data == state.eta.data[b_slices]).all()
    assert not np.may_share_memory(bs.eta.data, state.eta.data)
