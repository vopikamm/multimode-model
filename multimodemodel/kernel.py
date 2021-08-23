"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

import numpy as np
from .jit import _numba_2D_grid_iterator, _cyclic_shift
from .datastructure import State, Variable, Parameters


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


def pressure_gradient_i(state: State, params: Parameters) -> State:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.
    """
    result = _pressure_gradient_i(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.safe_data,  # type: ignore
        params.g,  # type: ignore
        state.u.grid.dx,  # type: ignore
        state.u.grid.mask,  # type: ignore
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(None, state.v.grid),
        eta=Variable(None, state.eta.grid),
    )


def pressure_gradient_j(state: State, params: Parameters) -> State:
    """Compute the second component of the pressure gradient.

    Using centered differences in space.
    """
    result = _pressure_gradient_j(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.safe_data,  # type: ignore
        params.g,  # type: ignore
        state.v.grid.dy,  # type: ignore
        state.v.grid.mask,  # type: ignore
    )
    return State(
        u=Variable(None, state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(None, state.eta.grid),
    )


def divergence_i(state: State, params: Parameters) -> State:
    """Compute divergence of flow along first dimension with centered differences."""
    result = _divergence_i(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.safe_data,  # type: ignore
        state.u.grid.mask,  # type: ignore
        params.H,  # type: ignore
        state.eta.grid.dx,  # type: ignore
        state.eta.grid.dy,  # type: ignore
        state.u.grid.dy,  # type: ignore
    )
    return State(
        u=Variable(None, state.u.grid),
        v=Variable(None, state.v.grid),
        eta=Variable(result, state.eta.grid),
    )


def divergence_j(state: State, params: Parameters) -> State:
    """Compute divergence of flow along second dimension with centered differences."""
    result = _divergence_j(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.safe_data,  # type: ignore
        state.v.grid.mask,  # type: ignore
        params.H,  # type: ignore
        state.eta.grid.dx,  # type: ignore
        state.eta.grid.dy,  # type: ignore
        state.v.grid.dx,  # type: ignore
    )
    return State(
        u=Variable(None, state.u.grid),
        v=Variable(None, state.v.grid),
        eta=Variable(result, state.eta.grid),
    )


def coriolis_j(state: State, params: Parameters) -> State:
    """Compute acceleration due to Coriolis force along second dimension.

    An arithmetic four point average of u onto the v-grid is performed.
    """
    result = _coriolis_j(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.safe_data,  # type: ignore
        state.u.grid.mask,  # type: ignore
        state.v.grid.mask,  # type: ignore
        params.f["v"],  # type: ignore
    )
    return State(
        u=Variable(None, state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(None, state.eta.grid),
    )


def coriolis_i(state: State, params: Parameters) -> State:
    """Compute the acceleration due to the Coriolis force along the first dimension.

    An arithmetic four point average of v onto the u-grid is performed.
    """
    result = _coriolis_i(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.safe_data,  # type: ignore
        state.v.grid.mask,  # type: ignore
        state.u.grid.mask,  # type: ignore
        params.f["u"],  # type: ignore
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(None, state.v.grid),
        eta=Variable(None, state.eta.grid),
    )
