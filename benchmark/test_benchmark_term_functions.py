"""Benchmark single terms."""
import numpy as np
import pytest

from multimodemodel import (
    Parameter,
    Variable,
    State,
    StaggeredGrid,
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_i,
    coriolis_j,
    divergence_i,
    divergence_j,
    f_on_sphere,
)
from multimodemodel.util import str_to_date


@pytest.fixture(params=(2, 3))
def ndim(request):
    """Return number of dimensions."""
    return request.param


@pytest.fixture()
def grids(ndim):
    """Return regular lon/lat staggered grid."""
    grid_pars = dict(
        lon_start=0.0,
        lon_end=100.0,
        lat_start=0.0,
        lat_end=50,
        nx=101,
        ny=51,
    )
    if ndim > 2:
        grid_pars["z"] = np.arange(10)
    return StaggeredGrid.regular_lat_lon_c_grid(**grid_pars)


def _get_params(staggered_grid):
    if staggered_grid.u.ndim > 2:
        H = np.ones_like(staggered_grid.u.z, dtype=np.float64)
    else:
        H = np.array([1.0])
    return Parameter(H=H, coriolis_func=f_on_sphere(1.0), on_grid=staggered_grid)


@pytest.mark.benchmark(warmup="on", warmup_iterations=5, max_time=2.0, min_time=0.01)
@pytest.mark.parametrize(
    "func",
    (
        pressure_gradient_i,
        pressure_gradient_j,
        coriolis_i,
        coriolis_j,
        divergence_i,
        divergence_j,
    ),
)
def test_benchmark_term(benchmark, grids, func):
    """Benchmark pressure_gradient_i."""
    params = _get_params(grids)

    t0 = str_to_date("2000")

    state = State(
        u=Variable(1.0 * np.ones(grids.u.shape), grids.u, time=t0),
        v=Variable(2.0 * np.ones(grids.v.shape), grids.v, time=t0),
        eta=Variable(3.0 * np.ones(grids.eta.shape), grids.eta, time=t0),
    )

    _ = benchmark(func, state, params)
