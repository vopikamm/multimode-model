"""Benchmark summation of arrays with numba."""
import pytest
import numpy as np
from multimodemodel.jit import sum_arr


arr = tuple(tuple(n * np.ones(ndim * (100,)) for n in range(10)) for ndim in (2, 3))


def _not_fused(a):
    """Return default numpy implementation."""
    return sum(a[1:], start=a[0])


@pytest.mark.benchmark(warmup="on", warmup_iterations=5, max_time=2.0, min_time=0.01)
@pytest.mark.parametrize("ndims", (2, 3))
@pytest.mark.parametrize("n_arr", range(2, 11))
@pytest.mark.parametrize("impl", (("fused", "notfused")))
def test_benchmark_sum_arrs(benchmark, impl, n_arr, ndims):
    """Benchmark linear combination operators."""
    arrs = arr[ndims - 2]
    if impl == "not_fused":
        assert (_not_fused(arrs[:n_arr]) == sum_arr(arrs[:n_arr])).all()
        _ = benchmark(_not_fused, arrs[:n_arr])
    else:
        _ = benchmark(sum_arr, arrs[:n_arr])
