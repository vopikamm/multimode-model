"""Just-in-time compiler logic."""
import numba
from inspect import signature
from typing import Any, Callable, Type, Optional
from functools import wraps, partial
import numpy as np
from .api import Array


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
    def _interate_over_grid_2D(
        ni: int, nj: int, *args: tuple[Any]
    ) -> Array:  # pragma: no cover
        result = np.empty((nj, ni), dtype=return_type)
        for j in range(nj):
            for i in range(ni):
                result[j, i] = exp_args(jitted_func, i, j, ni, nj, args)
        return result

    return _interate_over_grid_2D


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


@numba.njit
def _lin_comb_1(fac1: float, arr1: np.ndarray) -> np.ndarray:
    result = fac1 * arr1
    return result


@numba.njit
def _lin_comb_2(
    fac1: float, fac2: float, arr1: np.ndarray, arr2: np.ndarray
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
) -> np.ndarray:
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
def _sum_arrs_2(x1, x2):
    return x1 + x2


@numba.vectorize(_get_vectorize_signature(3))
def _sum_arrs_3(x1, x2, x3):
    return x1 + x2 + x3


@numba.vectorize(_get_vectorize_signature(4))
def _sum_arrs_4(x1, x2, x3, x4):
    return x1 + x2 + x3 + x4


@numba.vectorize(_get_vectorize_signature(5))
def _sum_arrs_5(x1, x2, x3, x4, x5):
    return x1 + x2 + x3 + x4 + x5


@numba.vectorize(_get_vectorize_signature(6))
def _sum_arrs_6(x1, x2, x3, x4, x5, x6):
    return x1 + x2 + x3 + x4 + x5 + x6


@numba.vectorize(_get_vectorize_signature(7))
def _sum_arrs_7(x1, x2, x3, x4, x5, x6, x7):
    return x1 + x2 + x3 + x4 + x5 + x6 + x7


@numba.vectorize(_get_vectorize_signature(8))
def _sum_arrs_8(x1, x2, x3, x4, x5, x6, x7, x8):
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8


@numba.vectorize(_get_vectorize_signature(9))
def _sum_arrs_9(x1, x2, x3, x4, x5, x6, x7, x8, x9):
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
