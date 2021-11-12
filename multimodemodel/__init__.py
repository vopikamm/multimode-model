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

from .integrate import (
    integrate,
    linearised_SWE,
    TimeSteppingFunction,
    StateIncrement,
    time_stepping_function,
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
    laplacian_mixing_u,
    laplacian_mixing_v,
)

from .coriolis import (
    CoriolisFunc,
    f_constant,
    beta_plane,
    f_on_sphere,
)
