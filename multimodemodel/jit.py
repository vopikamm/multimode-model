"""Just-in-time compiler logic."""
import numba
from inspect import signature
from typing import Tuple, Any, Callable, Type
from functools import wraps, partial
import numpy as np


@numba.njit(inline="always")  # type: ignore
def _expand_15_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i,
        j,
        ni,
        nj,
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
def _expand_14_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i,
        j,
        ni,
        nj,
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
def _expand_13_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i,
        j,
        ni,
        nj,
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
def _expand_12_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i,
        j,
        ni,
        nj,
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
def _expand_11_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i,
        j,
        ni,
        nj,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
    )


@numba.njit(inline="always")  # type: ignore
def _expand_10_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5])


@numba.njit(inline="always")  # type: ignore
def _expand_9_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4])


@numba.njit(inline="always")  # type: ignore
def _expand_8_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3])


@numba.njit(inline="always")  # type: ignore
def _expand_7_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2])


@numba.njit(inline="always")  # type: ignore
def _expand_6_arguments_ij(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1])


_arg_expand_map_ij = {
    6: _expand_6_arguments_ij,
    7: _expand_7_arguments_ij,
    8: _expand_8_arguments_ij,
    9: _expand_9_arguments_ij,
    10: _expand_10_arguments_ij,
    11: _expand_11_arguments_ij,
    12: _expand_12_arguments_ij,
    13: _expand_13_arguments_ij,
    14: _expand_14_arguments_ij,
    15: _expand_15_arguments_ij,
}


@numba.njit(inline="always")  # type: ignore
def _expand_17_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
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
def _expand_16_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
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
def _expand_15_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
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
def _expand_14_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
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
def _expand_13_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
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
def _expand_12_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3], args[4], args[5]
    )


@numba.njit(inline="always")  # type: ignore
def _expand_11_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3], args[4])


@numba.njit(inline="always")  # type: ignore
def _expand_10_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3])


@numba.njit(inline="always")  # type: ignore
def _expand_9_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1], args[2])


@numba.njit(inline="always")  # type: ignore
def _expand_8_arguments_ijk(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(i, j, k, ni, nj, nk, args[0], args[1])


_arg_expand_map_ijk = {
    8: _expand_8_arguments_ijk,
    9: _expand_9_arguments_ijk,
    10: _expand_10_arguments_ijk,
    11: _expand_11_arguments_ijk,
    12: _expand_12_arguments_ijk,
    13: _expand_13_arguments_ijk,
    14: _expand_14_arguments_ijk,
    15: _expand_15_arguments_ijk,
    16: _expand_16_arguments_ijk,
    17: _expand_17_arguments_ijk,
}


@numba.njit(inline="always")  # type: ignore
def _expand_20_arguments_ijkmn(
    func, i, j, k, m, n, ni, nj, nk, args
):  # pragma: no cover
    return func(
        i,
        j,
        k,
        m,
        n,
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
def _expand_21_arguments_ijkmn(
    func, i, j, k, m, n, ni, nj, nk, args
):  # pragma: no cover
    return func(
        i,
        j,
        k,
        m,
        n,
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


_arg_expand_map_ijkmn = {20: _expand_20_arguments_ijkmn, 21: _expand_21_arguments_ijkmn}


@numba.njit(inline="always")  # type: ignore
def _cyclic_shift(i: int, ni: int, shift: int = 1) -> int:  # pragma: no cover
    """Shift index and wrap it around if necessary.

    Note that negative indices are wrapped automatically.
    """
    if i + shift < ni:
        return i + shift
    else:
        return shift - 1


def _numba_2D_grid_iterator_template(func: Callable[..., float], return_type: Type):
    """Evaluate func at every gridpoint of a horizontal domain slice.

    func must take the indices of the grid point and the grid size as
    first arguments, e.g. func(i, j, ni, nj, other_args).
    """
    jitted_func = numba.njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map_ij[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit
    def _iterate_over_grid_2D(
        ni: int, nj: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((1, nj, ni), dtype=return_type)
        for j in range(nj):
            for i in range(ni):
                result[0, j, i] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _iterate_over_grid_2D


def _numba_3D_grid_iterator_template(func: Callable[..., float], return_type: Type):
    """Evaluate func at every gridpoint for every normal mode.

    func must take the indices of the grid point and the grid size as
    first arguments, e.g. func(i, j, k, ni, nj, nk, other_args).
    A mode dependent parameter must take the last argument position.
    """
    jitted_func = numba.njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map_ijk[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit(parallel=True)  # type: ignore
    def _iterate_over_grid_3D(
        ni: int, nj: int, nk: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((nk, nj, ni), dtype=return_type)
        for k in numba.prange(nk):
            for j in range(nj):
                for i in range(ni):
                    result[k, j, i] = exp_args(jitted_func, i, j, k, ni, nj, nk, args)
        return result

    return _iterate_over_grid_3D


def _numba_double_sum_template(func: Callable[..., float], return_type: Type):
    """Compute the double sum over all mode numbers.

    func is evaluated at every gridpoint for every normal mode.
    """
    jitted_func = numba.njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map_ijkmn[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit(parallel=True)  # type: ignore
    def _double_sum_over_modes(
        ni: int, nj: int, nk: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((nk * nk, nk, nj, ni), dtype=return_type)
        for ind in numba.prange(nk * nk):
            n, m = divmod(ind, nk)
            n = int(n)
            m = int(m)
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        result[ind, k, j, i] = exp_args(
                            jitted_func, i, j, k, m, n, ni, nj, nk, args
                        )
        return result.sum(axis=0)

    return _double_sum_over_modes


_numba_3D_grid_iterator_f8 = partial(
    _numba_3D_grid_iterator_template, return_type=np.float64
)
_numba_3D_grid_iterator_i8 = partial(
    _numba_3D_grid_iterator_template, return_type=np.int64
)
_numba_3D_grid_iterator_i1 = partial(
    _numba_3D_grid_iterator_template, return_type=np.int8
)
_numba_3D_grid_iterator_b1 = partial(
    _numba_3D_grid_iterator_template, return_type=np.bool_
)

_numba_3D_grid_iterator = _numba_3D_grid_iterator_f8


_numba_2D_grid_iterator_f8 = partial(
    _numba_2D_grid_iterator_template, return_type=np.float64
)
_numba_2D_grid_iterator_i8 = partial(
    _numba_2D_grid_iterator_template, return_type=np.int64
)
_numba_2D_grid_iterator_i1 = partial(
    _numba_2D_grid_iterator_template, return_type=np.int8
)
_numba_2D_grid_iterator_b1 = partial(
    _numba_2D_grid_iterator_template, return_type=np.bool_
)

_numba_2D_grid_iterator = _numba_2D_grid_iterator_f8


_numba_double_sum_f8 = partial(_numba_double_sum_template, return_type=np.float64)
_numba_double_sum_i8 = partial(_numba_double_sum_template, return_type=np.int64)
_numba_double_sum_i1 = partial(_numba_double_sum_template, return_type=np.int8)
_numba_double_sum_b1 = partial(_numba_double_sum_template, return_type=np.bool_)

_numba_double_sum = _numba_double_sum_f8
