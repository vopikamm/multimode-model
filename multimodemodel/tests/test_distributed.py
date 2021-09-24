"""Test implementation of distributed API."""
# flake8: noqa
import pytest
import numpy as np
from multimodemodel.API_implementation import ParameterSplit
from multimodemodel import (
    Parameters,
    StaggeredGrid,
    f_constant,
    f_on_sphere,
    beta_plane,
)


@pytest.fixture(
    params=(
        (100, 50),
        # (50, 50),
        # (50, 10),
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
        # (beta_plane, (10, 0.1, 0.0)),
        # (f_on_sphere, ()),
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


def test_ParameterSplit_init(param):
    ps = ParameterSplit(param, data=param.f)
    assert ps.g == param.g
    assert ps.H == param.H
    assert ps.rho_0 == param.rho_0
    assert ps.f == param.f


def test_ParameterSplit_split(param_split):
    parts = 2
    dim = (0,)

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


def test_ParameterSplit_merge(param_split):
    dim = (0,)
    parts = 2
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
