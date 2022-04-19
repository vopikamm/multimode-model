"""Linear shallow water model.

This script is the initial attempt to formulate the linearised shallow
water equation as python functions.
"""

from .config import config

from .datastructure import (
    Parameter,
    Variable,
    State,
    StateDeque,
    Domain,
    MultimodeParameter,
)

from .grid import (
    Grid,
    StaggeredGrid,
    GridShift,
)

from .integrate import (
    integrate,
    non_rotating_swe,
    linearised_SWE,
    TimeSteppingFunction,
    StateIncrement,
    time_stepping_function,
    euler_forward,
    adams_bashforth2,
    adams_bashforth3,
    Solver,
)

from .kernel import (
    coriolis_i,
    coriolis_j,
    divergence_j,
    divergence_i,
    pressure_gradient_j,
    pressure_gradient_i,
    sum_states,
    sum_vars,
    linear_combination,
    laplacian_mixing_u,
    laplacian_mixing_v,
    biharmonic_mixing_u,
    biharmonic_mixing_v,
    constant_vertical_mixing_u,
    constant_vertical_mixing_v,
    constant_vertical_mixing_eta,
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

from .border import (
    RegularSplitMerger,
    BorderSplitter,
    Border,
    BorderMerger,
    Tail,
)

from .workflow import Workflow, DaskWorkflow

from .util import str_to_date
