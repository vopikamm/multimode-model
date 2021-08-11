"""Just-in-time compiler logic."""
from numba import njit
from inspect import signature
from typing import Tuple, Any, Callable
from functools import wraps
import numpy as np


@njit(inline="always")  # type: ignore
def _expand_11_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(
        i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    )


@njit(inline="always")  # type: ignore
def _expand_10_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4], args[5])


@njit(inline="always")  # type: ignore
def _expand_9_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3], args[4])


@njit(inline="always")  # type: ignore
def _expand_8_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2], args[3])


@njit(inline="always")  # type: ignore
def _expand_7_arguments(func, i, j, ni, nj, args):  # pragma: no cover
    return func(i, j, ni, nj, args[0], args[1], args[2])


@njit(inline="always")  # type: ignore
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


@njit(inline="always")  # type: ignore
def _cyclic_shift(i: int, ni: int, shift: int = 1) -> int:  # pragma: no cover
    """Shift index and wrap it around if necessary.

    Note that negative indices are wrapped automatically.
    """
    if i + shift < ni:
        return i + shift
    else:
        return shift - 1


def _numba_2D_grid_iterator(func: Callable[..., float]):
    """Evaluate func at every gridpoint of a horizontal domain slice.

    func must take the indices of the grid point and the grid sice as
    first arguments, e.g. func(i, j, ni, nj, other_args).
    """
    jitted_func = njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map[len(signature(func).parameters)]

    @wraps(func)
    @njit
    def _interate_over_grid_2D(
        ni: int, nj: int, *args: Tuple[Any]
    ) -> np.ndarray:  # pragma: no cover
        result = np.empty((ni, nj))
        for i in range(ni):
            for j in range(nj):
                result[i, j] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _interate_over_grid_2D
