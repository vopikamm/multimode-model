"""Linear shallow water model.

This script is the initial attempt to formulate the linearised shallow
water equation as python functions.
"""

from .datastructure import (
    Parameters,
    Variable,
    State,
)

from .grid import (
    Grid,
    StaggeredGrid,
    GridShift,
)

from .integrator import (
    integrator,
    linearised_SWE,
    euler_forward,
    adams_bashforth2,
    adams_bashforth3,
)

from .kernel import (
    _pressure_gradient_i,
    coriolis_i,
    coriolis_j,
    divergence_j,
    divergence_i,
    pressure_gradient_j,
    pressure_gradient_i,
)
