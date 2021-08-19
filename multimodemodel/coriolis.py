"""Coriolis parameter computation.

Provides factory functions for callables that are used to
compute the coriolis parameter on a grid together with an
explicit type of these. The callables will be used to
initialize the Parameter class.
"""

from typing import Callable
import numpy as np

CoriolisFunc = Callable[[np.ndarray], np.ndarray]


def f_constant(f: float = 0) -> CoriolisFunc:
    """Set Coriolis parameter to a constant."""

    def closure(y: np.ndarray) -> np.ndarray:
        return np.ones(y.shape) * f

    return closure


def beta_plane(f0: float, beta: float, y0: float) -> CoriolisFunc:
    """Compute Coriolis parameter on a beta plane.

    The coriolis parameter is computes as

    `f = f0 + beta * (y - y0)`

    where y is the y-coordinate defined on the grids.
    """

    def closure(y: np.ndarray) -> np.ndarray:
        return f0 + beta * (y - y0)

    return closure


def f_on_sphere(omega: float = 7.272205e-05) -> CoriolisFunc:
    """Compute Coriolis parameter on a sphere.

    The coriolis parameter is computes as

    `f = 2 * omega * sin(pi / 180. * y)`

    where y is the y-coordinate defined on the grids
    which is assumed to be in units of degrees.
    """

    def closure(y: np.ndarray) -> np.ndarray:
        return 2 * omega * np.sin(np.pi / 180.0 * y)

    return closure
