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
    regular_lat_lon_c_grid,
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
    _zonal_pressure_gradient,
    coriolis_v,
    coriolis_u,
    meridional_divergence,
    zonal_divergence,
    meridional_pressure_gradient,
    zonal_pressure_gradient,
)
