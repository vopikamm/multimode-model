"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

import numpy as np
from .jit import _numba_2D_grid_iterator
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
    return -g * mask_u[i, j] * (eta[i, j] - eta[i - 1, j]) / dx_u[i, j]


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
    return -g * mask_v[i, j] * (eta[i, j] - eta[i, j - 1]) / dy_v[i, j]


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
    ip1 = (i + 1) % ni
    return (
        -H
        * (
            mask_u[ip1, j] * dy_u[ip1, j] * u[ip1, j]
            - mask_u[i, j] * dy_u[i, j] * u[i, j]
        )
        / dx_eta[i, j]
        / dy_eta[i, j]
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
    jp1 = (j + 1) % nj
    return (
        -H
        * (
            mask_v[i, jp1] * dx_v[i, jp1] * v[i, jp1]
            - mask_v[i, j] * dx_v[i, j] * v[i, j]
        )
        / dx_eta[i, j]
        / dy_eta[i, j]
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
    f: float,
) -> float:  # pragma: no cover
    ip1 = (i + 1) % ni
    return mask_v[i, j] * (
        -f
        * (
            mask_u[i, j - 1] * u[i, j - 1]
            + mask_u[i, j] * u[i, j]
            + mask_u[ip1, j] * u[ip1, j]
            + mask_u[ip1, j - 1] * u[ip1, j - 1]
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
    f: float,
) -> float:  # pragma: no cover
    jp1 = (j + 1) % nj
    return mask_u[i, j] * (
        f
        * (
            mask_v[i - 1, j] * v[i - 1, j]
            + mask_v[i, j] * v[i, j]
            + mask_v[i, jp1] * v[i, jp1]
            + mask_v[i - 1, jp1] * v[i - 1, jp1]
        )
        / 4.0
    )


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses. Periodic boundary conditions are applied.
"""


def pressure_gradient_i(state: State, params: Parameters) -> State:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.
    """
    result = _pressure_gradient_i(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,  # type: ignore
        params.g,  # type: ignore
        state.u.grid.dx,  # type: ignore
        state.u.grid.mask,  # type: ignore
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def pressure_gradient_j(state: State, params: Parameters) -> State:
    """Compute the second component of the pressure gradient.

    Using centered differences in space.
    """
    result = _pressure_gradient_j(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,  # type: ignore
        params.g,  # type: ignore
        state.v.grid.dy,  # type: ignore
        state.v.grid.mask,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def divergence_i(state: State, params: Parameters) -> State:
    """Compute first component of the divergence with centered differences in space."""
    result = _divergence_i(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.data,  # type: ignore
        state.u.grid.mask,  # type: ignore
        params.H,  # type: ignore
        state.eta.grid.dx,  # type: ignore
        state.eta.grid.dy,  # type: ignore
        state.u.grid.dy,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(result, state.eta.grid),
    )


def divergence_j(state: State, params: Parameters) -> State:
    """Compute second component of divergence with centered differences in space."""
    result = _divergence_j(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.data,  # type: ignore
        state.v.grid.mask,  # type: ignore
        params.H,  # type: ignore
        state.eta.grid.dx,  # type: ignore
        state.eta.grid.dy,  # type: ignore
        state.v.grid.dx,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(result, state.eta.grid),
    )


def coriolis_j(state: State, params: Parameters) -> State:
    """Compute acceleration due to Coriolis force along second dimension.

    An arithmetic four point average of u onto the v-grid is performed.
    """
    result = _coriolis_j(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.data,  # type: ignore
        state.u.grid.mask,  # type: ignore
        state.v.grid.mask,  # type: ignore
        params.f,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(result, state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )


def coriolis_i(state: State, params: Parameters) -> State:
    """Compute the acceleration due to the Coriolis force along the first dimension.

    An arithmetic four point average of v onto the u-grid is performed.
    """
    result = _coriolis_i(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.data,  # type: ignore
        state.v.grid.mask,  # type: ignore
        state.u.grid.mask,  # type: ignore
        params.f,  # type: ignore
    )
    return State(
        u=Variable(result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )
