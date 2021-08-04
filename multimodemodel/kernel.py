"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

import numpy as np
from .jit import _numba_2D_grid_iterator
from .datastructure import State, Variable, Parameters


@_numba_2D_grid_iterator
def _zonal_pressure_gradient(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.ndarray,
    mask: np.ndarray,
    g: float,
    dx_u: np.ndarray,
    dy_u: np.ndarray,
    dy_eta: np.ndarray,
) -> float:  # pragma: no cover
    return (
        -g
        * (
            mask[i, j] * dy_eta[i, j] * eta[i, j]
            - mask[i - 1, j] * dy_eta[i - 1, j] * eta[i - 1, j]
        )
        / dx_u[i, j]
        / dy_u[i, j]
    )


@_numba_2D_grid_iterator
def _meridional_pressure_gradient(
    i: int,
    j: int,
    ni: int,
    nj: int,
    eta: np.ndarray,
    mask: np.ndarray,
    g: float,
    dx_v: np.ndarray,
    dy_v: np.ndarray,
    dx_eta: np.ndarray,
) -> float:  # pragma: no cover
    return (
        -g
        * (
            mask[i, j] * dx_eta[i, j] * eta[i, j]
            - mask[i, j - 1] * dx_eta[i, j - 1] * eta[i, j - 1]
        )
        / dx_v[i, j]
        / dy_v[i, j]
    )


@_numba_2D_grid_iterator
def _zonal_divergence(
    i: int,
    j: int,
    ni: int,
    nj: int,
    u: np.ndarray,
    mask: np.ndarray,
    H: float,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dy_u: np.ndarray,
) -> float:  # pragma: no cover
    ip1 = (i + 1) % ni
    return (
        -H
        * (mask[ip1, j] * dy_u[ip1, j] * u[ip1, j] - mask[i, j] * dy_u[i, j] * u[i, j])
        / dx_eta[i, j]
        / dy_eta[i, j]
    )


@_numba_2D_grid_iterator
def _meridional_divergence(
    i: int,
    j: int,
    ni: int,
    nj: int,
    v: np.ndarray,
    mask: np.ndarray,
    H: float,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dx_v: np.ndarray,
) -> float:  # pragma: no cover
    jp1 = (j + 1) % nj
    return (
        -H
        * (mask[i, jp1] * dx_v[i, jp1] * v[i, jp1] - mask[i, j] * dx_v[i, j] * v[i, j])
        / dx_eta[i, j]
        / dy_eta[i, j]
    )


@_numba_2D_grid_iterator
def _coriolis_u(
    i: int, j: int, ni: int, nj: int, u: np.ndarray, mask: np.ndarray, f: float
) -> float:  # pragma: no cover
    ip1 = (i + 1) % ni
    return (
        -f
        * (
            mask[i, j - 1] * u[i, j - 1]
            + mask[i, j] * u[i, j]
            + mask[ip1, j] * u[ip1, j]
            + mask[ip1, j - 1] * u[ip1, j - 1]
        )
        / 4.0
    )


@_numba_2D_grid_iterator
def _coriolis_v(
    i: int, j: int, ni: int, nj: int, v: np.ndarray, mask: np.ndarray, f: float
) -> float:  # pragma: no cover
    jp1 = (j + 1) % nj
    return (
        f
        * (
            mask[i - 1, j] * v[i - 1, j]
            + mask[i, j] * v[i, j]
            + mask[i, jp1] * v[i, jp1]
            + mask[i - 1, jp1] * v[i - 1, jp1]
        )
        / 4.0
    )


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses. Periodic boundary conditions are applied.
"""


def zonal_pressure_gradient(state: State, params: Parameters) -> State:
    """Compute the zonal pressure gradient.

    Using centered differences in space.
    """
    result = _zonal_pressure_gradient(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,  # type: ignore
        state.eta.grid.mask,  # type: ignore
        params.g,  # type: ignore
        state.u.grid.dx,  # type: ignore
        state.u.grid.dy,  # type: ignore
        state.eta.grid.dy,  # type: ignore
    )
    return State(
        u=Variable(state.u.grid.mask * result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def meridional_pressure_gradient(state: State, params: Parameters) -> State:
    """Compute the meridional pressure gradient.

    Using centered differences in space.
    """
    result = _meridional_pressure_gradient(
        state.eta.grid.len_x,
        state.eta.grid.len_y,
        state.eta.data,  # type: ignore
        state.eta.grid.mask,  # type: ignore
        params.g,  # type: ignore
        state.v.grid.dx,  # type: ignore
        state.v.grid.dy,  # type: ignore
        state.eta.grid.dx,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(state.u.grid.mask * result, state.v.grid),
        eta=Variable(np.zeros_like(state.eta.data), state.eta.grid),
    )


def zonal_divergence(state: State, params: Parameters) -> State:
    """Compute the zonal divergence with centered differences in space."""
    result = _zonal_divergence(
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
        eta=Variable(state.eta.grid.mask * result, state.eta.grid),
    )


def meridional_divergence(state: State, params: Parameters) -> State:
    """Compute the meridional divergence with centered differences in space."""
    result = _meridional_divergence(
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
        eta=Variable(state.eta.grid.mask * result, state.eta.grid),
    )


def coriolis_u(state: State, params: Parameters) -> State:
    """Compute meridional acceleration due to Coriolis force.

    An arithmetic four point average of u onto the v-grid is performed.
    """
    result = _coriolis_u(
        state.u.grid.len_x,
        state.u.grid.len_y,
        state.u.data,  # type: ignore
        state.u.grid.mask,  # type: ignore
        params.f,  # type: ignore
    )
    return State(
        u=Variable(np.zeros_like(state.u.data), state.u.grid),
        v=Variable(state.v.grid.mask * result, state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )


def coriolis_v(state: State, params: Parameters) -> State:
    """Compute the zonal acceleration due to the Coriolis force.

    An arithmetic four point average of v onto the u-grid is performed.
    """
    result = _coriolis_v(
        state.v.grid.len_x,
        state.v.grid.len_y,
        state.v.data,  # type: ignore
        state.v.grid.mask,  # type: ignore
        params.f,  # type: ignore
    )
    return State(
        u=Variable(state.u.grid.mask * result, state.u.grid),
        v=Variable(np.zeros_like(state.v.data), state.v.grid),
        eta=Variable(np.zeros_like(state.v.data), state.eta.grid),
    )
