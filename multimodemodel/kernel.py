"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

from typing import Callable, Any, Optional, Sequence
import numpy as np
from functools import partial
from .api import Array, StateType
from .util import average_npdatetime64
from .jit import _numba_2D_grid_iterator, _cyclic_shift, _lin_comb, sum_arr
from .datastructure import Variable, Parameter
from .grid import Grid


def _extract_horizontal_slice(var, k):
    try:
        if var.ndim == 3:
            return var[k]
        else:
            return var
    except AttributeError:
        return var


def _map_2D_iterator_on_3D(func):
    """Turn a 2D grid iterator into a 3D grid iterator."""

    def wrapper(nz, *args):
        res = []
        for k in range(nz):
            arg_list = map(
                partial(_extract_horizontal_slice, k=k),
                args,
            )
            res.append(func(*arg_list))
        return np.array(res)

    return wrapper


@_numba_2D_grid_iterator
def _pressure_gradient_i(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: Array,
    g: float,
    dx_u: Array,
    mask_u: Array,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the first dimension."""
    return -g * mask_u[j, i] * (eta[j, i] - eta[j, i - 1]) / dx_u[j, i]


@_numba_2D_grid_iterator
def _pressure_gradient_j(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: Array,
    g: float,
    dy_v: Array,
    mask_v: Array,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the second dimension."""
    return -g * mask_v[j, i] * (eta[j, i] - eta[j - 1, i]) / dy_v[j, i]


@_numba_2D_grid_iterator
def _divergence_i(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: Array,
    mask_u: Array,
    H: float,
    dx_eta: Array,
    dy_eta: Array,
    dy_u: Array,
) -> float:  # pragma: no cover
    """Compute the divergence of the flow along the first dimension."""
    ip1 = _cyclic_shift(i, ni, 1)
    return (
        -H
        * (
            mask_u[j, ip1] * dy_u[j, ip1] * u[j, ip1]
            - mask_u[j, i] * dy_u[j, i] * u[j, i]
        )
        / dx_eta[j, i]
        / dy_eta[j, i]
    )


@_numba_2D_grid_iterator
def _divergence_j(
    i: int,
    j: int,
    ni: int,
    nj: int,
    v: Array,
    mask_v: Array,
    H: float,
    dx_eta: Array,
    dy_eta: Array,
    dx_v: Array,
) -> float:  # pragma: no cover
    """Compute the divergence of the flow along the second dimension."""
    jp1 = _cyclic_shift(j, nj, 1)
    return (
        -H
        * (
            mask_v[jp1, i] * dx_v[jp1, i] * v[jp1, i]
            - mask_v[j, i] * dx_v[j, i] * v[j, i]
        )
        / dx_eta[j, i]
        / dy_eta[j, i]
    )


@_numba_2D_grid_iterator
def _coriolis_j(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: Array,
    mask_u: Array,
    mask_v: Array,
    f: Array,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the second dimension."""
    ip1 = _cyclic_shift(i, ni, 1)
    return mask_v[j, i] * (
        -f[j, i]
        * (
            mask_u[j - 1, i] * u[j - 1, i]
            + mask_u[j, i] * u[j, i]
            + mask_u[j, ip1] * u[j, ip1]
            + mask_u[j - 1, ip1] * u[j - 1, ip1]
        )
        / 4.0
    )


@_numba_2D_grid_iterator
def _coriolis_i(
    i: int,
    j: int,
    ni: int,
    nj: int,
    v: Array,
    mask_v: Array,
    mask_u: Array,
    f: Array,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the first dimension."""
    jp1 = _cyclic_shift(j, nj, 1)
    return mask_u[j, i] * (
        f[j, i]
        * (
            mask_v[j, i - 1] * v[j, i - 1]
            + mask_v[j, i] * v[j, i]
            + mask_v[jp1, i] * v[jp1, i]
            + mask_v[jp1, i - 1] * v[jp1, i - 1]
        )
        / 4.0
    )


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses.
"""


def _apply_2D_iterator(
    func: Callable[..., Array],
    args: tuple[Any, ...],
    grid: Grid,
) -> Array:
    if grid.ndim == 3:
        func = _map_2D_iterator_on_3D(func)
        args = (grid.shape[grid.dim_z],) + args
    return func(*args)


def pressure_gradient_i(state: StateType, params: Parameter) -> StateType:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["u"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["eta"].safe_data,
        params.g,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.mask,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            _apply_2D_iterator(_pressure_gradient_i, args, grid),
            grid,
            state.variables["u"].time,
        )
    )


def pressure_gradient_j(state: StateType, params: Parameter) -> StateType:
    """Compute the second component of the pressure gradient.

    Using centered differences in space.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["v"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["eta"].safe_data,
        params.g,
        state.variables["v"].grid.dy,
        state.variables["v"].grid.mask,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            _apply_2D_iterator(_pressure_gradient_j, args, grid),
            grid,
            state.variables["v"].time,
        )
    )


def divergence_i(state: StateType, params: Parameter) -> StateType:
    """Compute divergence of flow along first dimension with centered differences.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["eta"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        params.H,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        state.variables["u"].grid.dy,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            _apply_2D_iterator(_divergence_i, args, grid),
            grid,
            state.variables["eta"].time,
        )
    )


def divergence_j(state: StateType, params: Parameter) -> StateType:
    """Compute divergence of flow along second dimension with centered differences.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["eta"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        params.H,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        state.variables["v"].grid.dx,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            _apply_2D_iterator(_divergence_j, args, grid),
            grid,
            state.variables["eta"].time,
        )
    )


def coriolis_j(state: StateType, params: Parameter) -> StateType:
    """Compute acceleration due to Coriolis force along second dimension.

    An arithmetic four point average of u onto the v-grid is performed.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["v"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
        params.f["v"],
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            _apply_2D_iterator(_coriolis_j, args, grid), grid, state.variables["v"].time
        ),
    )


def coriolis_i(state: StateType, params: Parameter) -> StateType:
    """Compute the acceleration due to the Coriolis force along the first dimension.

    An arithmetic four point average of v onto the u-grid is performed.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameters
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["u"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["u"].grid.mask,
        params.f["u"],
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            _apply_2D_iterator(_coriolis_i, args, grid), grid, state.variables["u"].time
        ),
    )


def linear_combination(
    factors: tuple[float, ...], arrays: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Return linear combination of arrays.

    Each array in arrays is multiplied by the corresponding
    factor in factors and the total sum is returned.
    """
    result = _lin_comb[len(factors)](*factors, *arrays)
    return result


def sum_states(
    states: Sequence[StateType], keep_time: Optional[int] = None
) -> StateType:
    """Sum states using optimized implementations.

    See documentation of jit.sum_arr for more information.

    Arguments
    ---------
    states: Sequence[StateType]
        Sequence of states to sum over.
    keep_time: Optional[int]
        If `None`, the resulting time will be the average of
        the timestamps of the input. If it is an integer, this will
        be the index of the variable within `variables` from which
        the timestep will be copied.
    """
    state_vars = set(sum((tuple(s.variables.keys()) for s in states), tuple()))
    vars = {
        var: tuple(s.variables[var] for s in states if var in s.variables)
        for var in state_vars
    }
    new_data = {
        var: sum_vars(vars_tuple, keep_time=keep_time)
        for var, vars_tuple in vars.items()
    }
    return states[0].__class__(**new_data)


def sum_vars(
    variables: Sequence[Variable], keep_time: Optional[int] = None
) -> Variable:
    """Sum variables using optimized implementations.

    The grid of the returned Variable will be a reference to the
    grid attributed of the first object in `variables`.

    See documentation of jit.sum_arr for more information.

    Arguments
    ---------
    variables: Sequence[Variable]
        Sequence of variables to sum over.
    keep_time: Optional[int]
        If `None`, the resulting time will be the average of
        the timestamps of the input. If it is an integer, this will
        be the index of the variable within `variables` from which
        the timestep will be copied.
    """
    if keep_time is None:
        mean_dt64 = average_npdatetime64(tuple(v.time for v in variables))
    else:
        mean_dt64 = variables[keep_time].time
    return variables[0].__class__(
        data=sum_arr(tuple(v.data for v in variables)),
        grid=variables[0].grid,
        time=mean_dt64,
    )
