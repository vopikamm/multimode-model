"""Kernel functions.

Second level functions taking data and parameters,
not dataclass instances, as input. This enables numba to precompile
computationally costly operations.
"""

from typing import Any, Optional, Sequence, Callable
import numpy as np

from .api import Array, StateType, Shape
from .util import average_npdatetime64
from .jit import (
    ParallelizeIterateOver,
    _make_grid_iteration_dispatch_table,
    _cyclic_shift,
    _lin_comb,
    sum_arr,
)
from .datastructure import Variable, Parameter, MultimodeParameter


def _get_from_dispatch_table(
    grid, dispatch_table: dict[ParallelizeIterateOver, Callable]
) -> Callable:
    if grid.ndim < 3:
        par_over = ParallelizeIterateOver.KJ
    else:
        par_over = ParallelizeIterateOver.K
    return dispatch_table[par_over]


def _pressure_gradient_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    eta: Array,
    mask_u: Array,
    dx_u: Array,
    g: float,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the first dimension."""
    return -g * mask_u[k, j, i] * (eta[k, j, i] - eta[k, j, i - 1]) / dx_u[j, i]


_pressure_gradient_i_dispatch_table = _make_grid_iteration_dispatch_table(
    _pressure_gradient_i
)


def _pressure_gradient_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    eta: Array,
    mask_v: Array,
    dy_v: Array,
    g: float,
) -> float:  # pragma: no cover
    """Compute the pressure gradient along the second dimension."""
    return -g * mask_v[k, j, i] * (eta[k, j, i] - eta[k, j - 1, i]) / dy_v[j, i]


_pressure_gradient_j_dispatch_table = _make_grid_iteration_dispatch_table(
    _pressure_gradient_j
)


def _divergence_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: Array,
    mask_u: Array,
    dx_eta: Array,
    dy_eta: Array,
    dy_u: Array,
    H: Array,
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


_divergence_i_dispatch_table = _make_grid_iteration_dispatch_table(_divergence_i)


def _divergence_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    v: Array,
    mask_v: Array,
    dx_eta: Array,
    dy_eta: Array,
    dx_v: Array,
    H: Array,
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


_divergence_j_dispatch_table = _make_grid_iteration_dispatch_table(_divergence_j)


def _coriolis_nonlinear_j(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: Array,
    mask_u: Array,
    mask_v: Array,
    dy_u: Array,
    dy_v: Array,
    f: Array,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the second dimension.

    Nonlinear metric terms are included.
    """
    ip1 = _cyclic_shift(i, ni, 1)
    return -mask_v[k, j, i] * (
        (f[j, i] + f[j, ip1])
        * (
            mask_u[k, j - 1, i] * dy_u[j - 1, i] * u[k, j - 1, i]
            + mask_u[k, j, i] * dy_u[j, i] * u[k, j, i]
            + mask_u[k, j - 1, ip1] * dy_u[j - 1, ip1] * u[k, j - 1, ip1]
            + mask_u[k, j, ip1] * dy_u[j, ip1] * u[k, j, ip1]
        )
        / 8
        / dy_v[j, i]
    )


_coriolis_nonlinear_j_dispatch_table = _make_grid_iteration_dispatch_table(
    _coriolis_nonlinear_j
)


def _coriolis_nonlinear_i(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    v: np.ndarray,
    u: np.ndarray,
    mask_v: np.ndarray,
    mask_u: np.ndarray,
    dx_u: np.ndarray,
    dx_v: np.ndarray,
    dy_v: np.ndarray,
    dx_q: np.ndarray,
    dy_q: np.ndarray,
    f: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the coriolis term along the first dimension.

    Nonlinear metric terms are included.
    """
    jp1 = _cyclic_shift(j, nj, 1)
    return mask_u[k, j, i] * (
        (
            f[jp1, i]
            + f[j, i]
            + (
                mask_v[k, jp1, i] * v[k, jp1, i]
                + mask_v[k, jp1, i - 1] * v[k, jp1, i - 1]
            )
            * (dy_v[jp1, i] - dy_v[jp1, i - 1])
            / 2
            / dx_q[jp1, i]
            / dy_q[jp1, i]
            - (mask_u[k, jp1, i] * u[k, jp1, i] + mask_u[k, j, i] * u[k, j, i])
            * (dx_u[jp1, i] - dx_u[j, i])
            / 2
            / dx_q[jp1, i]
            / dy_q[jp1, i]
            + (mask_v[k, j, i] * v[k, j, i] + mask_v[k, j, i - 1] * v[k, j, i - 1])
            * (dy_v[k, j, i] - dy_v[k, j, i - 1])
            / 2
            / dx_q[j, i]
            / dy_q[j, i]
            - (mask_u[k, j, i] * u[k, j, i] + mask_u[k, j - 1, i] * u[k, j - 1, i])
            * (dx_u[j, i] - dx_u[j - 1, i])
            / 2
            / dx_q[j, i]
            / dy_q[j, i]
        )
        * (
            mask_v[k, jp1, i - 1] * dx_v[jp1, i - 1] * v[k, jp1, i - 1]
            + mask_v[k, jp1, i] * dx_v[jp1, i] * v[k, jp1, i]
            + mask_v[k, j, i] * dx_v[j, i] * v[k, j, i]
            + mask_v[k, j, i - 1] * dx_v[j, i - 1] * v[k, j, i - 1]
        )
        / 8
        / dx_u[j, i]
    )


_coriolis_nonlinear_i_dispatch_table = _make_grid_iteration_dispatch_table(
    _coriolis_nonlinear_i
)


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

    The scheme is chosen to conserve enstrophy.
    """
    ip1 = _cyclic_shift(i, ni, 1)
    return -mask_v[k, j, i] * (
        (f[j, i] + f[j, ip1])
        * (
            mask_u[k, j - 1, i] * dy_u[j - 1, i] * u[k, j - 1, i]
            + mask_u[k, j, i] * dy_u[j, i] * u[k, j, i]
            + mask_u[k, j - 1, ip1] * dy_u[j - 1, ip1] * u[k, j - 1, ip1]
            + mask_u[k, j, ip1] * dy_u[j, ip1] * u[k, j, ip1]
        )
        / 8
        / dy_v[j, i]
    )


_coriolis_j_dispatch_table = _make_grid_iteration_dispatch_table(_coriolis_j)


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

    The scheme is chosen to conserve enstrophy.
    """
    jp1 = _cyclic_shift(j, nj, 1)
    return mask_u[k, j, i] * (
        (f[jp1, i] + f[j, i])
        * (
            mask_v[k, jp1, i - 1] * dx_v[jp1, i - 1] * v[k, jp1, i - 1]
            + mask_v[k, jp1, i] * dx_v[jp1, i] * v[k, jp1, i]
            + mask_v[k, j, i] * dx_v[j, i] * v[k, j, i]
            + mask_v[k, j, i - 1] * dx_v[j, i - 1] * v[k, j, i - 1]
        )
        / 8
        / dx_u[j, i]
    )


_coriolis_i_dispatch_table = _make_grid_iteration_dispatch_table(_coriolis_i)


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

    if mask_q[k, j, i] == 0.0:
        lbc_j = lbc
    else:
        lbc_j = 1

    if mask_q[k, jp1, i] == 0.0:
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
            - (dx_q[j, i] / dy_q[j, i])
            * lbc_j
            * (mask_u[k, j, i] * u[k, j, i] - mask_u[k, jm1, i] * u[k, jm1, i])
        )
        / dx_u[j, i]
        / dy_u[j, i]
    )


_laplacian_mixing_u_dispatch_table = _make_grid_iteration_dispatch_table(
    _laplacian_mixing_u
)


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

    if mask_q[k, j, i] == 0.0:
        lbc_i = lbc
    else:
        lbc_i = 1

    if mask_q[k, j, ip1] == 0.0:
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
            - (dx_eta[jm1, i] / dy_eta[jm1, i])
            * (mask_v[k, j, i] * v[k, j, i] - mask_v[k, jm1, i] * v[k, jm1, i])
        )
        / dx_v[j, i]
        / dy_v[j, i]
    )


_laplacian_mixing_v_dispatch_table = _make_grid_iteration_dispatch_table(
    _laplacian_mixing_v
)


def _laplacian_mixing_eta(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    eta: np.ndarray,
    mask_eta: np.ndarray,
    mask_q: np.ndarray,
    dx_u: np.ndarray,
    dy_u: np.ndarray,
    dx_v: np.ndarray,
    dy_v: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    k_h: float,
) -> float:  # pragma: no cover
    """
    Compute laplacian diffusion of eta.

    Free-slip boundary conditions are applied.
    """
    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    if mask_q[k, j, i] == 0.0:
        lbc_ij = 0.0
    else:
        lbc_ij = 1.0

    if mask_q[k, jp1, i] == 0.0:
        lbc_ijp1 = 0.0
    else:
        lbc_ijp1 = 1.0

    if mask_q[k, j, ip1] == 0.0:
        lbc_ip1j = 0.0
    else:
        lbc_ip1j = 1.0

    return (
        k_h
        * mask_eta[k, j, i]
        * (
            (dy_u[j, ip1] / dx_u[j, ip1])
            * lbc_ip1j
            * (mask_eta[k, j, ip1] * eta[k, j, ip1] - mask_eta[k, j, i] * eta[k, j, i])
            - (dy_u[j, i] / dx_u[j, i])
            * lbc_ij
            * (mask_eta[k, j, i] * eta[k, j, i] - mask_eta[k, j, im1] * eta[k, j, im1])
            + (dx_v[jp1, i] / dy_v[jp1, i])
            * lbc_ijp1
            * (mask_eta[k, jp1, i] * eta[k, jp1, i] - mask_eta[k, j, i] * eta[k, j, i])
            - (dx_v[j, i] / dy_v[j, i])
            * lbc_ij
            * (mask_eta[k, j, i] * eta[k, j, i] - mask_eta[k, jm1, i] * eta[k, jm1, i])
        )
        / dx_eta[j, i]
        / dy_eta[j, i]
    )


_laplacian_mixing_eta_dispatch_table = _make_grid_iteration_dispatch_table(
    _laplacian_mixing_eta
)


def _vertical_mixing(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    var: np.ndarray,
    mask: np.ndarray,
    P: np.ndarray,
) -> float:  # pragma: no cover
    """
    Compute the vertical mixing of dynamic variables.

    The vertical structure of the mixing coefficient must be considered when
    computing the double-mode-tensor P.
    """
    result = 0.0
    for m in range(nk):
        result += mask[m, j, i] * P[m, k] * var[m, j, i]

    return -result


_vertical_mixing_dispatch_table = _make_grid_iteration_dispatch_table(_vertical_mixing)


def _vertical_mixing_density(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    v: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    k_0: float,
    U: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the vertical mixing of density."""
    ip1 = _cyclic_shift(i, ni, 1)
    jp1 = _cyclic_shift(j, nj, 1)

    result = 0.0

    for n in range(nk):
        u_eta_n_ij = mask_u[n, j, i] * u[n, j, i] + mask_u[n, j, ip1] * u[n, j, ip1]
        v_eta_n_ij = mask_v[n, j, i] * v[n, j, i] + mask_v[n, jp1, i] * v[n, jp1, i]
        for m in range(nk):
            result += (
                U[n, m, k]
                * (
                    u_eta_n_ij
                    * (mask_u[m, j, i] * u[m, j, i] + mask_u[m, j, ip1] * u[m, j, ip1])
                    + v_eta_n_ij
                    * (mask_v[m, j, i] * v[m, j, i] + mask_v[m, jp1, i] * v[m, jp1, i])
                )
                / 4.0
            )
    return k_0 * result


_vertical_mixing_density_dispatch_table = _make_grid_iteration_dispatch_table(
    _vertical_mixing_density
)


def _linear_damping(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    var: np.ndarray,
    mask: np.ndarray,
    gamma: np.ndarray,
) -> float:  # pragma: no cover
    """Compute linear damping of dynamic variables."""
    return -gamma[k] * mask[k, j, i] * var[k, j, i]


_linear_damping_dispatch_table = _make_grid_iteration_dispatch_table(_linear_damping)


def _advection_momentum_u(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    mask_q: np.ndarray,
    dx_u: np.ndarray,
    dy_u: np.ndarray,
    dx_v: np.ndarray,
    lbc: int,
    Q: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the advection of zonal momentum."""
    if mask_u[k, j, i] == 0.0:
        return 0.0

    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    if mask_q[k, j, i] == 0.0:
        lbc_j = lbc
    else:
        lbc_j = 1

    if mask_q[k, jp1, i] == 0.0:
        lbc_jp1 = lbc
    else:
        lbc_jp1 = 1

    mask_fac_Q = mask_u[k, j, i] / dx_u[j, i] / dy_u[j, i] / 4
    mask_fac_R = 0.5 * mask_u[k, j, i]

    result = 0.0

    for n in range(nk):
        u_eta_n_ij = (
            dy_u[j, ip1] * mask_u[n, j, ip1] * u[n, j, ip1]
            + dy_u[j, i] * mask_u[n, j, i] * u[n, j, i]
        )
        u_eta_n_im1j = (
            dy_u[j, i] * mask_u[n, j, i] * u[n, j, i]
            + dy_u[j, im1] * mask_u[n, j, im1] * u[n, j, im1]
        )
        v_q_n_ijp1 = (
            dx_v[jp1, i] * mask_v[n, jp1, i] * v[n, jp1, i]
            + dx_v[jp1, im1] * mask_v[n, jp1, im1] * v[n, jp1, im1]
        )
        v_q_n_ij = (
            dx_v[j, i] * mask_v[n, j, i] * v[n, j, i]
            + dx_v[j, im1] * mask_v[n, j, im1] * v[n, j, im1]
        )
        w_u_n_ij = w[n, j, i] + w[n, j, im1]

        for m in range(nk):
            result += (
                Q[n, m, k]
                * (
                    mask_fac_Q
                    * (
                        u_eta_n_ij
                        * (
                            mask_u[m, j, ip1] * u[m, j, ip1]
                            + mask_u[m, j, i] * u[m, j, i]
                        )
                        - u_eta_n_im1j
                        * (
                            mask_u[m, j, i] * u[m, j, i]
                            + mask_u[m, j, im1] * u[m, j, im1]
                        )
                        + v_q_n_ijp1
                        * (
                            mask_u[m, jp1, i] * u[m, jp1, i]
                            + lbc_jp1 * mask_u[m, j, i] * u[m, j, i]
                        )
                        - v_q_n_ij
                        * (
                            lbc_j * mask_u[m, j, i] * u[m, j, i]
                            + mask_u[m, jm1, i] * u[m, jm1, i]
                        )
                    )
                    + mask_fac_R * u[m, j, i] * w_u_n_ij / H[n]
                )
                - R[n, m, k] * mask_fac_R * u[m, j, i] * w_u_n_ij
            )
    return -result


_advection_momentum_u_dispatch_table = _make_grid_iteration_dispatch_table(
    _advection_momentum_u
)


def _advection_momentum_v(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    mask_q: np.ndarray,
    dx_v: np.ndarray,
    dy_v: np.ndarray,
    dy_u: np.ndarray,
    lbc: int,
    Q: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the advection of zonal momentum."""
    if mask_v[k, j, i] == 0.0:
        return 0.0

    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    if mask_q[k, j, i] == 0.0:
        lbc_i = lbc
    else:
        lbc_i = 1

    if mask_q[k, j, ip1] == 0.0:
        lbc_ip1 = lbc
    else:
        lbc_ip1 = 1

    mask_fac_Q = mask_v[k, j, i] / dx_v[j, i] / dy_v[j, i] / 4
    mask_fac_R = 0.5 * mask_v[k, j, i]

    result = 0.0

    for n in range(nk):
        u_q_n_ij = (
            dy_u[j, i] * mask_u[n, j, i] * u[n, j, i]
            + dy_u[jm1, i] * mask_u[n, jm1, i] * u[n, jm1, i]
        )
        u_q_n_ip1j = (
            dy_u[j, ip1] * mask_u[n, j, ip1] * u[n, j, ip1]
            + dy_u[jm1, ip1] * mask_u[n, jm1, ip1] * u[n, jm1, ip1]
        )
        v_eta_n_ij = (
            dx_v[jp1, i] * mask_v[n, jp1, i] * v[n, jp1, i]
            + dx_v[j, i] * mask_v[n, j, i] * v[n, j, i]
        )
        v_eta_n_ijm1 = (
            dx_v[j, i] * mask_v[n, j, i] * v[n, j, i]
            + dx_v[jm1, i] * mask_v[n, jm1, i] * v[n, jm1, i]
        )
        w_v_n_ij = w[n, jm1, i] + w[n, j, i]

        for m in range(nk):
            result += (
                Q[n, m, k]
                * (
                    mask_fac_Q
                    * (
                        u_q_n_ip1j
                        * (
                            lbc_ip1 * mask_v[m, j, i] * v[m, j, i]
                            + mask_v[m, j, ip1] * v[m, j, ip1]
                        )
                        - u_q_n_ij
                        * (
                            mask_v[m, j, im1] * v[m, j, im1]
                            + lbc_i * mask_v[m, j, i] * v[m, j, i]
                        )
                        + v_eta_n_ij
                        * (
                            mask_v[m, j, i] * v[m, j, i]
                            + mask_v[m, jp1, i] * v[m, jp1, i]
                        )
                        - v_eta_n_ijm1
                        * (
                            mask_v[m, j, i] * v[m, j, i]
                            + mask_v[m, jm1, i] * v[m, jm1, i]
                        )
                    )
                    + mask_fac_R * v[m, j, i] * w_v_n_ij / H[n]
                )
                - R[n, m, k] * mask_fac_R * v[m, j, i] * w_v_n_ij
            )
    return -result


_advection_momentum_v_dispatch_table = _make_grid_iteration_dispatch_table(
    _advection_momentum_v
)


def _advection_density(
    i: int,
    j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    u: np.ndarray,
    v: np.ndarray,
    eta: np.ndarray,
    w: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    mask_eta: np.ndarray,
    dx_eta: np.ndarray,
    dy_eta: np.ndarray,
    dy_u: np.ndarray,
    dx_v: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    H: np.ndarray,
) -> float:  # pragma: no cover
    """Compute the advection of density."""
    ip1 = _cyclic_shift(i, ni, 1)
    im1 = _cyclic_shift(i, ni, -1)
    jp1 = _cyclic_shift(j, nj, 1)
    jm1 = _cyclic_shift(j, nj, -1)

    mask_fac_S = mask_v[k, j, i] / dx_eta[j, i] / dy_eta[j, i] / 2

    result = 0.0

    for n in range(nk):
        u_u_n_ij = dy_u[j, i] * mask_u[n, j, i] * u[n, j, i]
        u_u_n_ip1j = dy_u[j, ip1] * mask_u[n, j, ip1] * u[n, j, ip1]
        v_v_n_ij = dx_v[j, i] * mask_v[n, j, i] * v[n, j, i]
        v_v_n_ijp1 = dx_v[jp1, i] * mask_v[n, jp1, i] * v[n, jp1, i]
        for m in range(nk):
            result += (
                S[n, m, k]
                * (
                    mask_fac_S
                    * (
                        u_u_n_ip1j
                        * (
                            mask_eta[m, j, ip1] * eta[m, j, ip1]
                            + mask_eta[m, j, i] * eta[m, j, i]
                        )
                        - u_u_n_ij
                        * (
                            mask_eta[m, j, i] * eta[m, j, i]
                            + mask_eta[m, j, im1] * eta[m, j, im1]
                        )
                        + v_v_n_ijp1
                        * (
                            mask_eta[m, jp1, i] * eta[m, jp1, i]
                            + mask_eta[m, j, i] * eta[m, j, i]
                        )
                        - v_v_n_ij
                        * (
                            mask_eta[m, j, i] * eta[m, j, i]
                            + mask_eta[m, jm1, i] * eta[m, jm1, i]
                        )
                    )
                    + mask_eta[m, j, i] * eta[m, j, i] * w[n, j, i] / H[n]
                )
                - T[n, m, k] * mask_eta[m, j, i] * eta[m, j, i] * w[n, j, i]
            )
    return -result


_advection_density_dispatch_table = _make_grid_iteration_dispatch_table(
    _advection_density
)

"""
Non jit-able functions. First level funcions connecting the jit-able
function output to dataclasses.
"""


def _at_least_3D(*arrs: Array):
    """Prepend singleton dimensions to at least 3D."""
    return tuple(a.reshape(_shape_at_least_3D(a.shape)) for a in arrs)


def _shape_at_least_3D(shape: Shape):
    """Expand shape to be at least 3D."""
    if len(shape) >= 3:
        return shape
    return (3 - len(shape)) * (1,) + shape


def pressure_gradient_i(state: StateType, params: Parameter) -> StateType:
    """Compute the pressure gradient along the first dimension.

    Using centered differences in space.

    Parameters
    ----------
    state : State
      State of the system
    params : Parameter
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["u"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, u_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["u"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _pressure_gradient_i_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        u_mask,
        state.variables["u"].grid.dx,
        params.g,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape),
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
    params : Parameter
      Parameters of the system

    Returns
    -------
    State
    """
    grid = state.variables["v"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, v_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _pressure_gradient_j_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        v_mask,
        state.variables["v"].grid.dy,
        params.g,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape),
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
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _divergence_i_dispatch_table)

    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        grid.dx,
        grid.dy,
        state.variables["u"].grid.dy,
        params.H,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
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
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _divergence_j_dispatch_table)

    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        grid.dx,
        grid.dy,
        state.variables["v"].grid.dx,
        params.H,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
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
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask, v_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _coriolis_j_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        v_mask,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dy,
        params.f["q"],
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape), grid, state.variables["v"].time
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
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask, u_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["u"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _coriolis_i_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        u_mask,
        state.variables["u"].grid.dx,
        state.variables["v"].grid.dx,
        params.f["q"],
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape), grid, state.variables["u"].time
        ),
    )


def laplacian_mixing_u(state: StateType, params: Parameter) -> StateType:
    """Compute laplacian diffusion of zonal velocities."""
    grid = state.variables["u"].grid
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask, q_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["q"].grid.mask,
    )
    lbc = 2 * params.no_slip
    func = _get_from_dispatch_table(grid, _laplacian_mixing_u_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        lbc,
        params.a_h,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["u"].time,
        )
    )


def laplacian_mixing_v(state: StateType, params: Parameter) -> StateType:
    """Compute laplacian diffusion of meridional velocities."""
    grid = state.variables["v"].grid
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask, q_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["q"].grid.mask,
    )
    lbc = 2 * params.no_slip
    func = _get_from_dispatch_table(grid, _laplacian_mixing_v_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        q_mask,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        lbc,
        params.a_h,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["v"].time,
        )
    )


def laplacian_mixing_eta(state: StateType, params: Parameter) -> StateType:
    """Compute laplacian diffusion of isopycnal displacement."""
    grid = state.variables["eta"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, eta_mask, q_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["eta"].grid.mask,
        state.variables["q"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _laplacian_mixing_eta_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        eta_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        params.k_h,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["eta"].time,
        )
    )


def biharmonic_mixing_u(state: StateType, params: Parameter) -> StateType:
    """Compute biharmonic diffusion of zonal velocities."""
    grid = state.variables["u"].grid
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask, q_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["q"].grid.mask,
    )
    lbc = 2 * params.no_slip
    func = _get_from_dispatch_table(grid, _laplacian_mixing_u_dispatch_table)
    args_1: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        lbc,
        1.0,
    )

    laplacian_u = func(*args_1).reshape(grid.shape)

    args_2: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        laplacian_u,
        u_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        0.0,
        params.b_h,
    )

    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args_2).reshape(grid.shape),
            grid,
            state.variables["u"].time,
        )
    )


def biharmonic_mixing_v(state: StateType, params: Parameter) -> StateType:
    """Compute biharmonic diffusion of meridional velocities."""
    grid = state.variables["v"].grid
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask, q_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
        state.variables["q"].grid.mask,
    )
    lbc = 2 * params.no_slip
    func = _get_from_dispatch_table(grid, _laplacian_mixing_v_dispatch_table)
    args_1: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        q_mask,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        lbc,
        1.0,
    )

    laplacian_v = func(*args_1).reshape(grid.shape)

    args_2: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        laplacian_v,
        v_mask,
        q_mask,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["q"].grid.dx,
        state.variables["q"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        0.0,
        params.b_h,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args_2).reshape(grid.shape),
            grid,
            state.variables["v"].time,
        )
    )


def biharmonic_mixing_eta(state: StateType, params: Parameter) -> StateType:
    """Compute biharmonic diffusion of isopycnal displacement."""
    grid = state.variables["eta"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, eta_mask, q_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["eta"].grid.mask,
        state.variables["q"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _laplacian_mixing_eta_dispatch_table)
    args_1: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        eta_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        1.0,
    )

    laplacian_eta = func(*args_1).reshape(grid.shape)

    args_2: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        laplacian_eta,
        eta_mask,
        q_mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        params.b_h,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args_2).reshape(grid.shape),
            grid,
            state.variables["eta"].time,
        )
    )


def constant_vertical_mixing_u(
    state: StateType, params: MultimodeParameter
) -> StateType:
    """Compute constant vertical mixing of zonal velocities."""
    grid = state.variables["u"].grid
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _vertical_mixing_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        params.a_v * params.P,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["u"].time,
        )
    )


def constant_vertical_mixing_v(
    state: StateType, params: MultimodeParameter
) -> StateType:
    """Compute constant vertical mixing of meridional velocities."""
    grid = state.variables["v"].grid
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _vertical_mixing_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        params.a_v * params.P,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["v"].time,
        )
    )


def constant_vertical_mixing_eta(
    state: StateType, params: MultimodeParameter
) -> StateType:
    """Compute constant vertical mixing of density."""
    grid = state.variables["eta"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, eta_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["eta"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _vertical_mixing_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        eta_mask,
        params.k_0 * params.P,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["eta"].time,
        )
    )


def ri_vertical_mixing_eta(state: StateType, params: MultimodeParameter) -> StateType:
    """Compute vertical mixing of density."""
    grid = state.variables["eta"].grid
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
    )
    v, v_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _vertical_mixing_density_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        v,
        u_mask,
        v_mask,
        params.k_v,
        params.U,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["eta"].time,
        )
    )


def linear_damping_u(state: StateType, params: Parameter) -> StateType:
    """Compute linear damping of zonal velocities."""
    grid = state.variables["u"].grid
    shape = _shape_at_least_3D(grid.shape)
    u, u_mask = _at_least_3D(
        state.variables["u"].safe_data,
        state.variables["u"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _linear_damping_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        u,
        u_mask,
        params.a_v,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["u"].time,
        )
    )


def linear_damping_v(state: StateType, params: Parameter) -> StateType:
    """Compute linear damping of meridional velocities."""
    grid = state.variables["v"].grid
    shape = _shape_at_least_3D(grid.shape)
    v, v_mask = _at_least_3D(
        state.variables["v"].safe_data,
        state.variables["v"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _linear_damping_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        v,
        v_mask,
        params.a_v,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["v"].time,
        )
    )


def linear_damping_eta(state: StateType, params: Parameter) -> StateType:
    """Compute linear damping of isopycnal displacement."""
    grid = state.variables["eta"].grid
    shape = _shape_at_least_3D(grid.shape)
    eta, eta_mask = _at_least_3D(
        state.variables["eta"].safe_data,
        state.variables["eta"].grid.mask,
    )
    func = _get_from_dispatch_table(grid, _linear_damping_dispatch_table)
    args: tuple[Any, ...] = (
        shape[grid.dim_x],
        shape[grid.dim_y],
        shape[grid.dim_z],
        eta,
        eta_mask,
        params.k_v,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),
            grid,
            state.variables["eta"].time,
        )
    )


def advection_momentum_u(state: StateType, params: MultimodeParameter) -> StateType:
    """Compute advection of zonal momentum."""
    grid = state.variables["u"].grid
    lbc = 2 * params.free_slip
    func = _advection_momentum_u_dispatch_table[ParallelizeIterateOver.KJI]
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["v"].safe_data,
        state.diagnostic_variables["w"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
        state.variables["q"].grid.mask,
        state.variables["u"].grid.dx,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        lbc,
        params.Q,
        params.R,
        params.H,
    )
    return state.__class__(
        u=state.variables["u"].__class__(
            func(*args).reshape(grid.shape),  # type: ignore
            grid,
            state.variables["u"].time,
        )
    )


def advection_momentum_v(state: StateType, params: MultimodeParameter) -> StateType:
    """Compute advection of meridional momentum."""
    grid = state.variables["v"].grid
    lbc = 2 * params.free_slip
    func = _advection_momentum_v_dispatch_table[ParallelizeIterateOver.KJI]
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["v"].safe_data,
        state.diagnostic_variables["w"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
        state.variables["q"].grid.mask,
        state.variables["v"].grid.dx,
        state.variables["v"].grid.dy,
        state.variables["u"].grid.dy,
        lbc,
        params.Q,
        params.R,
        params.H,
    )
    return state.__class__(
        v=state.variables["v"].__class__(
            func(*args).reshape(grid.shape),  # type: ignore
            grid,
            state.variables["v"].time,
        )
    )


def advection_density(state: StateType, params: MultimodeParameter) -> StateType:
    """Compute advection of perturbation density."""
    grid = state.variables["eta"].grid
    func = _advection_density_dispatch_table[ParallelizeIterateOver.KJI]
    args = (
        grid.shape[grid.dim_x],
        grid.shape[grid.dim_y],
        grid.shape[grid.dim_z],
        state.variables["u"].safe_data,
        state.variables["v"].safe_data,
        state.variables["eta"].safe_data,
        state.diagnostic_variables["w"].safe_data,
        state.variables["u"].grid.mask,
        state.variables["v"].grid.mask,
        state.variables["eta"].grid.mask,
        state.variables["eta"].grid.dx,
        state.variables["eta"].grid.dy,
        state.variables["u"].grid.dy,
        state.variables["v"].grid.dx,
        params.S,
        params.T,
        params.H,
    )
    return state.__class__(
        eta=state.variables["eta"].__class__(
            func(*args).reshape(grid.shape),  # type: ignore
            grid,
            state.variables["eta"].time,
        )
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
