"""Linear shallow water model.

This script is the initial attempt to formulate the linearised shallow
water equation as python functions.
"""
import numpy as np
import timeit
from matplotlib import pyplot as plt
from multimodemodel import (
    Parameters,
    StaggeredGrid,
    State,
    Variable,
    integrator,
    adams_bashforth3,
)


"""
Very basic setup with only zonal flow for testing the functionality.
"""

if __name__ == "__main__":
    params = Parameters(t_end=7200.0)
    grid_args = dict(
        lon_start=0.0,
        lon_end=50.0,
        lat_start=0.0,
        lat_end=50.0,
        nx=51,
        ny=51,
    )

    c_grid = StaggeredGrid.regular_lat_lon_c_grid(**grid_args)

    u_0 = np.zeros_like(c_grid.u.x)
    v_0 = np.zeros_like(c_grid.v.x)
    eta_0 = c_grid.eta.mask * (c_grid.eta.x / 50) - 0.5

    init = State(
        u=Variable(u_0, c_grid.u),
        v=Variable(v_0, c_grid.v),
        eta=Variable(eta_0, c_grid.eta),
    )

    start = timeit.default_timer()
    solution = integrator(init, params, scheme=adams_bashforth3)
    stop = timeit.default_timer()

    print("Runtime: ", stop - start, " s ")

    plt.figure()
    plt.pcolor(solution.eta.data)
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.pcolor(solution.u.data)
    plt.colorbar()
    plt.show()
