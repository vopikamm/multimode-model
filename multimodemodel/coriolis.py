"""Coriolis parameter computation.

Provides factory functions for callables that are used to
compute the coriolis parameter on a grid together with an
explicit type of these. The callables will be used to
initialize the Parameter class.
"""

from .typing import Array
from typing import Callable
import numpy as np

CoriolisFunc = Callable[[Array], Array]


def f_constant(f: float = 0) -> CoriolisFunc:
    r"""Return a closure that returns a constant Coriolis parameter.

    Parameters
    ----------
    f : float, default=0.0
      Constant value of f

    Returns
    -------
    CoriolisFunc
    """

    def closure(y: Array) -> Array:
        return np.ones(y.shape) * f

    return closure


def beta_plane(f0: float, beta: float, y0: float) -> CoriolisFunc:
    r"""Return a closure that computes the Coriolis parameter on a beta plane.

    The coriolis parameter is computes as

    .. math::
        f = f_0 + \beta * (y - y_0)

    Parameters
    ----------
    f0 : float
      Coriolis parameter at `y=y0`.
    beta : float
      Meridional derivative of `f`.
    y0 : float
      `y` coordinate about which the Coriolis parameter is linearly approximated.

    Returns
    -------
    CoriolisFunc
    """

    def closure(y: Array) -> Array:
        return f0 + beta * (y - y0)

    return closure


def f_on_sphere(omega: float = 7.272205e-05) -> CoriolisFunc:
    r"""Return a closure that computes the Coriolis parameter on a sphere.

    The Coriolis parameter is computes as

    .. math::
        f = 2 * \Omega * \sin(\frac{\pi}{180}y)

    where `y` is the y-coordinate defined on the grids
    which is assumed to be in units of degrees north.

    Parameters
    ----------
    omega : float, default=7.272205e-05
      Angular frequency of the rotating sphere. Deafult matches that of Earth.

    Returns
    -------
    CoriolisFunc

    """

    def closure(y: Array) -> Array:
        return 2 * omega * np.sin(np.pi / 180.0 * y)

    return closure
