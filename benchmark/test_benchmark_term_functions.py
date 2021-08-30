"""Benchmark single terms."""
import pytest

from multimodemodel import (
    Parameters,
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


def _get_grids():
    grid_pars = dict(
        lon_start=0.0,
        lon_end=100.0,
        lat_start=0.0,
        lat_end=50,
        nx=101,
        ny=51,
    )
    return StaggeredGrid.regular_lat_lon_c_grid(**grid_pars)


def _get_params(staggered_grid):
    return Parameters(coriolis_func=f_on_sphere(1.0), on_grid=staggered_grid)


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
def test_benchmark_term(benchmark, func):
    """Benchmark pressure_gradient_i."""
    c_grid = _get_grids()
    params = _get_params(c_grid)

    state = State(
        u=Variable(c_grid.u.x.copy(), c_grid.u, time=0.0),
        v=Variable(c_grid.v.x.copy(), c_grid.v, time=0.0),
        eta=Variable(c_grid.eta.x.copy(), c_grid.eta, time=0.0),
    )

    _ = benchmark(func, state, params)
