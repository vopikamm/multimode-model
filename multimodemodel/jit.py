"""Just-in-time compiler logic."""
from numba import njit
from inspect import signature
from typing import Tuple, Any
import numpy as np


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
}


def _numba_2D_grid_iterator(func):
    jitted_func = njit(inline="always")(func)  # type: ignore
    exp_args = _arg_expand_map[len(signature(func).parameters)]

    @njit
    def _interate_over_grid_2D(ni: int, nj: int, *args: Tuple[Any]):  # pragma: no cover
        result = np.empty((ni, nj))
        for i in range(ni):
            for j in range(nj):
                result[i, j] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _interate_over_grid_2D
