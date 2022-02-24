"""Benchmark summation of states with numba."""
import pytest
import numpy as np
from multimodemodel import State, sum_states, Grid, Variable, str_to_date


def _get_state(nx=100, ny=100, dx=1.0, dy=1.0, val_u=1.0, val_v=1.0, val_eta=1.0):
    x, y = np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx)[::-1]
    g1 = Grid(x, y)
    d_u = None if val_u is None else np.zeros_like(g1.x) + val_u
    d_v = None if val_v is None else np.zeros_like(g1.x) + val_v
    d_eta = None if val_eta is None else np.zeros_like(g1.x) + val_eta
    t0 = str_to_date("2000")
    return State(
        u=Variable(d_u, g1, t0),
        v=Variable(d_v, g1, t0),
        eta=Variable(d_eta, g1, t0),
    )


def _get_states(n, size=100):
    states = []
    for i in range(n):
        states.append(
            _get_state(
                nx=size,
                val_u=1.0,
                val_v=None if n % 2 == 0 else 2.0,
                val_eta=None,
            )
        )
    return tuple(states)


states = 10 * (_get_state(nx=100, ny=100),)


def _not_fused(states):
    return sum(
        states[1:],
        start=states[0],
    )


@pytest.mark.benchmark(warmup="on", warmup_iterations=5, max_time=2.0, min_time=0.01)
@pytest.mark.parametrize("n_states", range(2, 11))
@pytest.mark.parametrize("impl", (("fused", "not_fused")))
def test_benchmark_sum_states(benchmark, impl, n_states):
    """Benchmark linear combination operators."""
    states = _get_states(n_states, size=100)
    if impl == "not_fused":
        assert _not_fused(states) == sum_states(states)
        _ = benchmark(_not_fused, states)
    else:
        _ = benchmark(sum_states, states)
