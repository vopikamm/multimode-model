"""Benchmark linear combination of arrays with numba."""
import pytest
import numpy as np
from multimodemodel.jit import _lin_comb_1, _lin_comb_2, _lin_comb_3

np_impl = {
    1: lambda fac, arr1: fac * arr1,
    2: lambda fac1, fac2, arr1, arr2: fac1 * arr1 + fac2 * arr2,
    3: lambda fac1, fac2, fac3, arr1, arr2, arr3: fac1 * arr1
    + fac2 * arr2
    + fac3 * arr3,
}
nb_impl = {
    1: _lin_comb_1,
    2: _lin_comb_2,
    3: _lin_comb_3,
}

arr = tuple(n * np.ones((100, 10000)) for n in range(len(np_impl)))
fac = (1.5, 3.4, 5.5)


@pytest.mark.benchmark(warmup="on", warmup_iterations=5, max_time=5.0, min_time=0.01)
@pytest.mark.parametrize("n_arr", ((1, 2, 3)))
@pytest.mark.parametrize("impl", (("numpy", "numba")))
def test_benchmark_lin_comb(benchmark, impl, n_arr):
    """Benchmark linear combination operators."""
    if impl == "numpy":
        impl = np_impl
    else:
        impl = nb_impl
    _ = benchmark(impl[n_arr], *fac[:n_arr], *arr[:n_arr])
