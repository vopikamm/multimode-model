"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

from typing import Callable, Tuple, Any
import numpy as np
from .jit import _numba_2D_grid_iterator, _cyclic_shift
from .datastructure import State, Variable, Parameters
from .grid import Grid
from functools import partial


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
    eta: np.ndarray,
    g: float,
    dx_u: np.ndarray,
    mask_u: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the first dimension."""
    return -g * mask_u[j, i] * (eta[j, i] - eta[j, i - 1]) / dx_u[j, i]


@_numba_2D_grid_iterator
def _pressure_gradient_j(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.ndarray,
    g: float,
    dy_v: np.ndarray,
    mask_v: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the second dimension."""
    return -g * mask_v[j, i] * (eta[j, i] - eta[j - 1, i]) / dy_v[j, i]


@_numba_2D_grid_iterator
def _divergence_i(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: np.ndarray,
    mask_u: np.ndarray,
    H: float,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dy_u: np.ndarray,
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
    v: np.ndarray,
    mask_v: np.ndarray,
    H: float,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dx_v: np.ndarray,
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
    u: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    f: np.ndarray,
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
    v: np.ndarray,
    mask_v: np.ndarray,
    mask_u: np.ndarray,
    f: np.ndarray,
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
    func: Callable[..., np.ndarray],
    args: Tuple[Any, ...],
    grid: Grid,
) -> np.ndarray:
    if grid.ndim == 3:
        func = _map_2D_iterator_on_3D(func)
        args = (grid.shape[grid.dim_z],) + args
    return func(*args)


def pressure_gradient_i(state: State, params: Parameters) -> State:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.
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
    return State(
        u=Variable(
            _apply_2D_iterator(_pressure_gradient_i, args, grid),
            grid,
            state.variables["u"].time,
        )
    )


def pressure_gradient_j(state: State, params: Parameters) -> State:
    """Compute the second component of the pressure gradient.

    Using centered differences in space.
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
    return State(
        v=Variable(
            _apply_2D_iterator(_pressure_gradient_j, args, grid),
            grid,
            state.variables["v"].time,
        )
    )


def divergence_i(state: State, params: Parameters) -> State:
    """Compute divergence of flow along first dimension with centered differences."""
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
    return State(
        eta=Variable(
            _apply_2D_iterator(_divergence_i, args, grid),
            grid,
            state.variables["eta"].time,
        )
    )


def divergence_j(state: State, params: Parameters) -> State:
    """Compute divergence of flow along second dimension with centered differences."""
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
    return State(
        eta=Variable(
            _apply_2D_iterator(_divergence_j, args, grid),
            grid,
            state.variables["eta"].time,
        )
    )


def coriolis_j(state: State, params: Parameters) -> State:
    """Compute acceleration due to Coriolis force along second dimension.

    An arithmetic four point average of u onto the v-grid is performed.
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
    return State(
        v=Variable(
            _apply_2D_iterator(_coriolis_j, args, grid), grid, state.variables["v"].time
        ),
    )


def coriolis_i(state: State, params: Parameters) -> State:
    """Compute the acceleration due to the Coriolis force along the first dimension.

    An arithmetic four point average of v onto the u-grid is performed.
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
    return State(
        u=Variable(
            _apply_2D_iterator(_coriolis_i, args, grid), grid, state.variables["u"].time
        ),
    )
