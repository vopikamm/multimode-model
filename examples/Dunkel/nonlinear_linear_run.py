"""Running a comparison run of one year linear vs nonlinear."""

import xarray as xr
import numpy as np
import functools as ft
import operator as op

from multimodemodel import StaggeredGrid
from multimodemodel import MultimodeParameters, f_on_sphere
from multimodemodel import State, Variable
from multimodemodel import integrate, adams_bashforth3
from multimodemodel import (
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_i,
    coriolis_j,
    divergence_i,
    divergence_j,
    laplacian_mixing_u,
    laplacian_mixing_v,
    linear_damping_eta,
    linear_damping_u,
    linear_damping_v,
    advection_density,
    advection_momentum_u,
    advection_momentum_v,
)

nmodes = 25

c_grid = StaggeredGrid.regular_lat_lon_c_grid(
    lon_start=-50.0,
    lon_end=0.0,
    lat_start=-10.0,
    lat_end=10.0,
    nx=50 * 4 + 1,
    ny=20 * 4 + 1,
    z=np.arange(nmodes),
)

Nsq = np.load("Nsq.npy", allow_pickle=True)
depth = Nsq[1, :]
Nsq = Nsq[0, :]

multimode_params = MultimodeParameters(
    z=depth,
    Nsq=Nsq,
    nmodes=nmodes,
    coriolis_func=f_on_sphere(omega=7.272205e-05),
    on_grid=c_grid,
    no_slip=True,
)
ds = multimode_params.as_dataset

A = 1.33e-7
H = abs(depth[0] - depth[-1])
multimode_params.gamma_h = (A / ds.c ** 2 / H).values
multimode_params.gamma_v = (A / ds.c ** 2).values

tau_x = np.empty(c_grid.u.shape)
for k in range(nmodes):
    tau_x[k, :, :] = -0.05 * ds.psi.values[0, k]

tau_x *= c_grid.u.mask


def zonal_wind(state, params):
    """Uniform westward wind stress."""
    return State(u=Variable(tau_x / params.rho_0 / H, c_grid.u, np.datetime64("NaT")))


linear_terms = [
    pressure_gradient_i,
    pressure_gradient_j,
    coriolis_i,
    coriolis_j,
    divergence_i,
    divergence_j,
    laplacian_mixing_u,
    laplacian_mixing_v,
    linear_damping_u,
    linear_damping_v,
    linear_damping_eta,
    zonal_wind,
]

nonlinear_terms = [
    advection_density,
    advection_momentum_u,
    advection_momentum_v,
]


def linear_model(state, params):
    """RHS of the linear model run."""
    w = (divergence_j(state, params) + divergence_i(state, params)).eta.safe_data
    state.set_diagnostic_variable(w=Variable(w, c_grid.eta, state.eta.time))
    return ft.reduce(op.add, (term(state, params) for term in linear_terms))


def nonlinear_model(state, params):
    """RHS of the nonlinear model run."""
    w = (divergence_j(state, params) + divergence_i(state, params)).eta.safe_data
    state.set_diagnostic_variable(w=Variable(w, c_grid.eta, state.eta.time))
    return ft.reduce(
        op.add, (term(state, params) for term in (linear_terms + nonlinear_terms))
    )


def save_as_Dataset(state: State, params: MultimodeParameters):
    """Save a state as xarray.DataSet."""
    ds = state.variables["u"].as_dataarray.to_dataset(name="u_tilde")
    ds["v_tilde"] = state.variables["v"].as_dataarray
    ds["h_tilde"] = state.variables["eta"].as_dataarray
    x = (["j", "i"], state.q.grid.x)
    y = (["j", "i"], state.q.grid.y)
    ds.assign_coords({"x": x, "y": y})
    return ds


time = 365 * 24 * 3600.0  # 1 year
step = c_grid.u.dx.min() / ds.c.values.max() / 10.0  # satisfying the CFL criterion
t0 = np.datetime64("2000-01-01")  # starting date
output_steps = 5


def run(rhs, params, step, time):
    """Define the run as function."""
    model_run = integrate(
        State(
            u=Variable(None, c_grid.u, t0),
            v=Variable(None, c_grid.v, t0),
            eta=Variable(None, c_grid.eta, t0),
            q=Variable(None, c_grid.q, t0),
        ),
        params,
        RHS=rhs,
        scheme=adams_bashforth3,
        step=step,
        time=time,
    )

    Nt = time // step

    output = []

    tolerance = 10.0
    for i, next_state in enumerate(model_run):
        if i % (Nt // output_steps) == 0:
            output.append(save_as_Dataset(next_state, params))
        if np.nanmax(abs(next_state.variables["u"].safe_data)) > tolerance:
            output.append(save_as_Dataset(next_state, params))
            tolerance += 1.0
        if tolerance > 20.0:
            return xr.combine_by_coords(output)

    return xr.combine_by_coords(output)


out_linear = run(linear_model, multimode_params, step, time)
out_nonlinear = run(nonlinear_model, multimode_params, step, time)

out_linear = out_linear.rename({"z": "nmode"})
out_nonlinear = out_nonlinear.rename({"z": "nmode"})

out_linear["u"] = xr.dot(ds.psi, out_linear.u_tilde)
out_linear["v"] = xr.dot(ds.psi, out_linear.v_tilde)
out_linear["h"] = xr.dot(ds.psi, out_linear.h_tilde)

out_nonlinear["u"] = xr.dot(ds.psi, out_nonlinear.u_tilde)
out_nonlinear["v"] = xr.dot(ds.psi, out_nonlinear.v_tilde)
out_nonlinear["h"] = xr.dot(ds.psi, out_nonlinear.h_tilde)

np.save("linear.npy", out_linear.to_dict())
np.save("nonlinear.npy", out_nonlinear.to_dict())
