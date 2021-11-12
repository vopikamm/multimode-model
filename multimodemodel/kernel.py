"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

import numpy as np

# from numpy.core.fromnumeric import shape
from .jit import _numba_3D_grid_iterator, _cyclic_shift
from .datastructure import State, Variable, Parameters


@_numba_3D_grid_iterator
def _pressure_gradient_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    eta: np.ndarray,
    mask_u: np.ndarray,
    dx_u: np.ndarray,
    g: float,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the first dimension."""
    return -g * mask_u[k, j, i] * (eta[k, j, i] - eta[k, j, i - 1]) / dx_u[j, i]


@_numba_3D_grid_iterator
def _pressure_gradient_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    eta: np.ndarray,
    mask_v: np.ndarray,
    dy_v: np.ndarray,
    g: float,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the second dimension."""
    return -g * mask_v[k, j, i] * (eta[k, j, i] - eta[k, j - 1, i]) / dy_v[j, i]


@_numba_3D_grid_iterator
def _divergence_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    mask_u: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dy_u: np.ndarray,
    H: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the divergence of the flow along the first dimension.

    This term depends on the mode number k.
    """
    ip1 = _cyclic_shift(i, ni, 1)
    return (
        -H[k]
        * (
            mask_u[k, j, ip1] * dy_u[j, ip1] * u[k, j, ip1]
            - mask_u[k, j, i] * dy_u[j, i] * u[k, j, i]
        )
        / dx_eta[j, i]
        / dy_eta[j, i]
    )


@_numba_3D_grid_iterator
def _divergence_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    v: np.ndarray,
    mask_v: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dx_v: np.ndarray,
    H: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the divergence of the flow along the second dimension.

    This term depends on the mode number k.
    """
    jp1 = _cyclic_shift(j, nj, 1)
    return (
        -H[k]
        * (
            mask_v[k, jp1, i] * dx_v[jp1, i] * v[k, jp1, i]
            - mask_v[k, j, i] * dx_v[j, i] * v[k, j, i]
        )
        / dx_eta[j, i]
        / dy_eta[j, i]
    )


@_numba_3D_grid_iterator
def _coriolis_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    dy_u: np.ndarray,
    dy_v: np.ndarray,
    f: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the second dimension.

    The scheme is chosen to conserve energy.
    """
    ip1 = _cyclic_shift(i, ni, 1)
    return -mask_v[k, j, i] * (
        (
            f[j, i]
            * (
                mask_u[k, j - 1, i] * dy_u[j - 1, i] * u[k, j - 1, i]
                + mask_u[k, j, i] * dy_u[j, i] * u[k, j, i]
            )
            / 2
            + f[j, ip1]
            * (
                mask_u[k, j - 1, ip1] * dy_u[j - 1, ip1] * u[k, j - 1, ip1]
                + mask_u[k, j, ip1] * dy_u[j, ip1] * u[k, j, ip1]
            )
            / 2
        )
        / 2
        / dy_v[j, i]
    )


@_numba_3D_grid_iterator
def _coriolis_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    v: np.ndarray,
    mask_v: np.ndarray,
    mask_u: np.ndarray,
    dx_u: np.ndarray,
    dx_v: np.ndarray,
    f: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the first dimension.

    The scheme is chosen to conserve energy.
    """
    jp1 = _cyclic_shift(j, nj, 1)
    return mask_u[k, j, i] * (
        (
            f[jp1, i]
            * (
                mask_v[k, jp1, i - 1] * dx_v[jp1, i - 1] * v[k, jp1, i - 1]
                + mask_v[k, jp1, i] * dx_v[jp1, i] * v[k, jp1, i]
            )
            / 2
            + f[j, i]
            * (
                mask_v[k, j, i] * dx_v[j, i] * v[k, j, i]
                + mask_v[k, j, i - 1] * dx_v[j, i - 1] * v[k, j, i - 1]
            )
            / 2
        )
        / 2
        / dx_u[j, i]
    )


@_numba_3D_grid_iterator
def _laplacian_mixing_u(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    mask_u: np.ndarray,
    mask_q: np.ndarray,
    dx_u: np.ndarray,
    dy_u: np.ndarray,
    dx_q: np.ndarray,
    dy_q: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    lbc: int,
    a_h: float,
) -> float:  # pragma: no cover
    """Compute laplacian diffusion of u."""
    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    if mask_q[k, j, i] == 0:
        lbc_j = lbc
    else:
        lbc_j = 1

    if mask_q[k, jp1, i] == 0:
        lbc_jp1 = lbc
    else:
        lbc_jp1 = 1

    return (
        a_h
        * mask_u[k, j, i]
        * (
            (dy_eta[j, i] / dx_eta[j, i])
            * (mask_u[k, j, ip1] * u[k, j, ip1] - mask_u[k, j, i] * u[k, j, i])
            - (dy_eta[j, im1] / dx_eta[j, im1])
            * (mask_u[k, j, i] * u[k, j, i] - mask_u[k, j, im1] * u[k, j, im1])
            + (dx_q[jp1, i] / dy_q[jp1, i])
            * lbc_jp1
            * (mask_u[k, jp1, i] * u[k, jp1, i] - mask_u[k, j, i] * u[k, j, i])
            - (dx_q[j, i] / dx_q[j, i])
            * lbc_j
            * (mask_u[k, j, i] * u[k, j, i] - mask_u[k, jm1, i] * u[k, jm1, i])
        )
        / dx_u[j, i]
        / dy_u[j, i]
    )


@_numba_3D_grid_iterator
def _laplacian_mixing_v(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    v: np.ndarray,
    mask_v: np.ndarray,
    mask_q: np.ndarray,
    dx_v: np.ndarray,
    dy_v: np.ndarray,
    dx_q: np.ndarray,
    dy_q: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    lbc: int,
    a_h: float,
) -> float:  # pragma: no cover
    """Compute laplacian diffusion of v."""
    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    if mask_q[k, j, i] == 0:
        lbc_i = lbc
    else:
        lbc_i = 1

    if mask_q[k, j, ip1] == 0:
        lbc_ip1 = lbc
    else:
        lbc_ip1 = 1

    return (
        a_h
        * mask_v[k, j, i]
        * (
            (dy_q[j, ip1] / dx_q[j, ip1])
            * lbc_ip1
            * (mask_v[k, j, ip1] * v[k, j, ip1] - mask_v[k, j, i] * v[k, j, i])
            - (dy_q[j, i] / dx_q[j, i])
            * lbc_i
            * (mask_v[k, j, i] * v[k, j, i] - mask_v[k, j, im1] * v[k, j, im1])
            + (dx_eta[j, i] / dy_eta[j, i])
            * (mask_v[k, jp1, i] * v[k, jp1, i] - mask_v[k, j, i] * v[k, j, i])
            - (dx_eta[jm1, i] / dx_eta[jm1, i])
            * (mask_v[k, j, i] * v[k, j, i] - mask_v[k, jm1, i] * v[k, jm1, i])
        )
        / dx_v[j, i]
        / dy_v[j, i]
    )


"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses.
"""


def pressure_gradient_i(state: State, params: Parameters) -> State:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.
    """
    grid = state.variables["u"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["eta"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["u"].grid.dx,
        params.g,
    )
    return State(
        u=Variable(
            _pressure_gradient_i(*args),
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
        grid.shape[grid.dim_z],
        state.variables["eta"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["v"].grid.dy,
        params.g,
    )
    return State(
        v=Variable(
            _pressure_gradient_j(*args),
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
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        grid.dx,
        grid.dy,
        state.variables["u"].grid.dy,
        params.H,
    )
    return State(
        eta=Variable(
            _divergence_i(*args),
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
        grid.shape[grid.dim_z],
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        grid.dx,
        grid.dy,
        state.variables["v"].grid.dx,
        params.H,
    )
    return State(
        eta=Variable(
            _divergence_j(*args),
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
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dy,
        params.f["v"],
    )
    return State(
        v=Variable(_coriolis_j(*args), grid, state.variables["v"].time),
    )


def coriolis_i(state: State, params: Parameters) -> State:
    """Compute the acceleration due to the Coriolis force along the first dimension.

    An arithmetic four point average of v onto the u-grid is performed.
    """
    grid = state.variables["u"].grid
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["u"].grid.mask,
        state.variables["u"].grid.dx,
        state.variables["v"].grid.dx,
        params.f["u"],
    )
    return State(
        u=Variable(_coriolis_i(*args), grid, state.variables["u"].time),
    )


def laplacian_mixing_u(state: State, params: Parameters) -> State:
    """Compute laplacian diffusion of zonal velocities."""
    grid = state.variables["u"].grid
    lbc = 2 * params.no_slip
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["q"].grid.mask,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        lbc,
        params.a_h,
    )
    return State(
        u=Variable(
            _laplacian_mixing_u(*args),
            grid,
            state.variables["u"].time,
        )
    )


def laplacian_mixing_v(state: State, params: Parameters) -> State:
    """Compute laplacian diffusion of meridional velocities."""
    grid = state.variables["v"].grid
    lbc = 2 * params.no_slip
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["q"].grid.mask,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        lbc,
        params.a_h,
    )
    return State(
        v=Variable(
            _laplacian_mixing_v(*args),
            grid,
            state.variables["u"].time,
        )
    )
