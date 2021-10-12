"""Just-in-time compiler logic."""
import numba
from inspect import signature
from typing import Tuple, Any, Callable, Type
from functools import wraps, partial
import numpy as np


@numba.njit(inline="always")  # type: ignore
def _expand_12_arguments_3D(func, i, j, k, ni, nj, nk, args):  # pragma: no cover
    return func(
        i, j, k, ni, nj, nk, args[0], args[1], args[2], args[3], args[4], args[5]
    )


@numba.njit(inline="always")  # type: ignore
def _expand_11_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    )


@numba.njit(inline="always")  # type: ignore
def _expand_10_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5])


@numba.njit(inline="always")  # type: ignore
def _expand_9_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4])


@numba.njit(inline="always")  # type: ignore
def _expand_8_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3])


@numba.njit(inline="always")  # type: ignore
def _expand_7_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2])


@numba.njit(inline="always")  # type: ignore
def _expand_6_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1])


_arg_expand_map = {
    6: _expand_6_arguments,
    7: _expand_7_arguments,
    8: _expand_8_arguments,
    9: _expand_9_arguments,
    10: _expand_10_arguments,
    11: _expand_11_arguments,
}

_arg_expand_map_3D = {12: _expand_12_arguments_3D}


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
    exp_args = _arg_expand_map[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit
    def _iterate_over_grid_2D(
        ni: int, nj: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((nj, ni), dtype=return_type)
        for j in range(nj):
            for i in range(ni):
                result[j, i] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _iterate_over_grid_2D


def _numba_3D_grid_iterator_template(func: Callable[..., float], return_type: Type):
    """Evaluate func at every gridpoint of a horizontal domain slice.

    func must take the indices of the grid point and the grid size as
    first arguments, e.g. func(i, j, k, ni, nj, nk, other_args).
    A mode dependent parameter must take the last argument position.
    """
    jitted_func = numba.njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map_3D[len(signature(func).parameters)]

    @wraps(func)
    @numba.njit
    def _iterate_over_grid_3D(
        ni: int, nj: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        nk = len(args[-1])
        result = np.empty((nk, nj, ni), dtype=return_type)
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    result[k, j, i] = exp_args(jitted_func, i, j, k, ni, nj, nk, args)
        return result

    return _iterate_over_grid_3D


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

_numba_2D_grid_iterator = _numba_2D_grid_iterator_f8
_numba_3D_grid_iterator = _numba_3D_grid_iterator_f8
