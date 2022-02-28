"""Just-in-time compiler logic."""
import numba
from inspect import signature
from typing import Any, Callable, Type, Optional
from functools import wraps, partial
import numpy as np
from .api import Array


@numba.njit(inline="always")  # type: ignore
def _expand_20_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
        args[12],
        args[13],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_19_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
        args[12],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_18_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_17_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_16_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_15_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_14_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_13_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i,
        j,
        k,
        ni,
        nj,
        nk,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_12_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3], args[4], args[5]
    )


@numba.njit(inline="always")  # type: ignore
def _expand_11_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3], args[4])


@numba.njit(inline="always")  # type: ignore
def _expand_10_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3])


@numba.njit(inline="always")  # type: ignore
def _expand_9_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2])


@numba.njit(inline="always")  # type: ignore
def _expand_8_arguments(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1])


_arg_expand_map = {
    8: _expand_8_arguments,
    9: _expand_9_arguments,
    10: _expand_10_arguments,
    11: _expand_11_arguments,
    12: _expand_12_arguments,
    13: _expand_13_arguments,
    14: _expand_14_arguments,
    15: _expand_15_arguments,
    16: _expand_16_arguments,
    17: _expand_17_arguments,
    18: _expand_18_arguments,
    19: _expand_19_arguments,
    20: _expand_20_arguments,
}


@numba.njit(inline="always")  # type: ignore
def _cyclic_shift(i: int, ni: int, shift: int = 1) -> int:  # pragma: no cover
    """Shift index and wrap it around if necessary.

    Note that negative indices are wrapped automatically.
    """
    if i + shift < ni:
        return i + shift
    else:
        return shift - 1


def _numba_3D_grid_iterator_template(
    func: Callable[..., float], return_type: Type, parallel_kji: bool = False
):
    """Evaluate func at every gridpoint for every normal mode.

    func must take the indices of the grid point and the grid size as
    first arguments, e.g. func(i, j, k, ni, nj, nk, other_args).
    A mode dependent parameter must take the last argument position.
    The evaluation can be parallelized over all dimensions for computational heavy functions with parallel_kij set to True. Otherwise the parallelization is performed over k only.
    """
    jitted_func = numba.njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit(parallel=True)  # type: ignore
    def _iterate_over_grid_3D_parallel_over_kji(
        ni: int, nj: int, nk: int, *args: tuple[Any, ...]
    ) -> Array:  # pragma: no cover
        result = np.empty((nk, nj, ni), dtype=return_type)
        for ind in numba.prange(nk * nj * ni):
            k, residual = divmod(ind, nj * ni)
            j, i = divmod(residual, ni)
            k = int(k)
            j = int(j)
            i = int(i)
            result[k, j, i] = exp_args(jitted_func, i, j, k, ni, nj, nk, args)
        return result

    @wraps(func)
    @numba.njit(parallel=True)  # type: ignore
    def _iterate_over_grid_3D_parallel_over_k(
        ni: int, nj: int, nk: int, *args: tuple[Any, ...]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((nk, nj, ni), dtype=return_type)
        for k in numba.prange(nk):
            for j in range(nj):
                for i in range(ni):
                    result[k, j, i] = exp_args(jitted_func, i, j, k, ni, nj, nk, args)
        return result

    if parallel_kji:
        return _iterate_over_grid_3D_parallel_over_kji
    else:
        return _iterate_over_grid_3D_parallel_over_k


_numba_3D_grid_iterator = partial(
    _numba_3D_grid_iterator_template, return_type=np.float64, parallel_kji=False
)

_numba_3D_grid_iterator_i8 = partial(
    _numba_3D_grid_iterator_template, return_type=np.int8, parallel_kji=False
)

_numba_3D_grid_iterator_parallel_over_kji = partial(
    _numba_3D_grid_iterator_template, return_type=np.float64, parallel_kji=True
)


@numba.njit
def _lin_comb_1(fac1: float, arr1: np.ndarray) -> np.ndarray:  # pragma: no cover
    result = fac1 * arr1
    return result


@numba.njit
def _lin_comb_2(
    fac1: float, fac2: float, arr1: np.ndarray, arr2: np.ndarray  # pragma: no cover
) -> np.ndarray:
    result = fac1 * arr1 + fac2 * arr2
    return result


@numba.njit
def _lin_comb_3(
    fac1: float,
    fac2: float,
    fac3: float,
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr3: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    return fac1 * arr1 + fac2 * arr2 + fac3 * arr3


_lin_comb = {1: _lin_comb_1, 2: _lin_comb_2, 3: _lin_comb_3}


def _sum_arrs_default(*arrs: np.ndarray) -> np.ndarray:
    if len(arrs) < 2:
        return arrs[0]
    return sum(arrs[1:], start=arrs[0])


_vectorize_types = (numba.int32, numba.int64, numba.float32, numba.float64)


def _get_vectorize_signature(n_args: int) -> list:
    return [t(*(n_args * (t,))) for t in _vectorize_types]


@numba.vectorize(_get_vectorize_signature(2))
def _sum_arrs_2(x1, x2):  # pragma: no cover
    return x1 + x2


@numba.vectorize(_get_vectorize_signature(3))
def _sum_arrs_3(x1, x2, x3):  # pragma: no cover
    return x1 + x2 + x3


@numba.vectorize(_get_vectorize_signature(4))
def _sum_arrs_4(x1, x2, x3, x4):  # pragma: no cover
    return x1 + x2 + x3 + x4


@numba.vectorize(_get_vectorize_signature(5))
def _sum_arrs_5(x1, x2, x3, x4, x5):  # pragma: no cover
    return x1 + x2 + x3 + x4 + x5


@numba.vectorize(_get_vectorize_signature(6))
def _sum_arrs_6(x1, x2, x3, x4, x5, x6):  # pragma: no cover
    return x1 + x2 + x3 + x4 + x5 + x6


@numba.vectorize(_get_vectorize_signature(7))
def _sum_arrs_7(x1, x2, x3, x4, x5, x6, x7):  # pragma: no cover
    return x1 + x2 + x3 + x4 + x5 + x6 + x7


@numba.vectorize(_get_vectorize_signature(8))
def _sum_arrs_8(x1, x2, x3, x4, x5, x6, x7, x8):  # pragma: no cover
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8


@numba.vectorize(_get_vectorize_signature(9))
def _sum_arrs_9(x1, x2, x3, x4, x5, x6, x7, x8, x9):  # pragma: no cover
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9


_sum_func: dict[int, Any] = {
    2: _sum_arrs_2,
    3: _sum_arrs_3,
    4: _sum_arrs_4,
    5: _sum_arrs_5,
    6: _sum_arrs_6,
    7: _sum_arrs_7,
    8: _sum_arrs_8,
    9: _sum_arrs_9,
}


def sum_arr(arrs: tuple[Optional[np.ndarray], ...]) -> Optional[np.ndarray]:
    """Sum over a sequence of arrays.

    Optimized implementations will be used for up to nine arrays.
    For more arguments, a default implementation will be used falling
    back to pure numpy.

    The implementations are optimized by fusing the addition operations.
    The benefit of using this function is greatest for summing many
    arrays.

    Arguments
    ---------
    arrs: tuple of numpy arrays

    Returns
    -------
    Elementwise sum of all arrays
    """
    filtered_arrs = tuple(x for x in arrs if x is not None)
    n_arrs = len(filtered_arrs)
    if n_arrs == 0:
        return None
    func = _sum_func.get(n_arrs, _sum_arrs_default)
    return func(*filtered_arrs)
