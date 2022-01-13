"""Linear shallow water model.

This script is the initial attempt to formulate the linearised shallow
water equation as python functions.
"""

from .datastructure import (
    Parameters,
    Variable,
    State,
    MultimodeParameters,
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
    linear_damping_u,
    linear_damping_v,
    linear_damping_eta,
    advection_momentum_u,
    advection_momentum_v,
    advection_density,
)

from .coriolis import (
    CoriolisFunc,
    f_constant,
    beta_plane,
    f_on_sphere,
)

from .jit import (
    _numba_2D_grid_iterator,
    _numba_3D_grid_iterator,
    _numba_double_sum,
    _cyclic_shift,
)
