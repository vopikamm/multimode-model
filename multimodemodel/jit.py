"""Just-in-time compiler logic."""
import numba
from inspect import signature
from typing import Tuple, Any, Callable, Type
from functools import wraps, partial
import numpy as np


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
        ni: int, nj: int, nk: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
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
        ni: int, nj: int, nk: int, *args: Tuple[Any]
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

_numba_3D_grid_iterator_parallel_over_kji = partial(
    _numba_3D_grid_iterator_template, return_type=np.float64, parallel_kji=True
)
