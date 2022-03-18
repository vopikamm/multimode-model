"""Scripted run of Markus Dunkels model."""

import xarray as xr
import numpy as np

from multimodemodel import (
    StaggeredGrid,
    MultimodeParameter,
    f_on_sphere,
    State,
    Variable,
    integrate,
    adams_bashforth3,
)

import functools as ft
import operator as op
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


Nsq = np.load("Nsq.npy", allow_pickle=True)
depth = Nsq[1, :]
Nsq = Nsq[0, :]

nmodes = 25

c_grid = StaggeredGrid.regular_lat_lon_c_grid(
    lon_start=-50.0,
    lon_end=50.0,
    lat_start=-5.0,
    lat_end=5.0,
    nx=100 * 4 + 1,
    ny=10 * 4 + 1,
    z=np.arange(nmodes),
)

multimode_params = MultimodeParameter(
    z=depth,
    Nsq=Nsq,
    nmodes=nmodes,
    coriolis_func=f_on_sphere(omega=7.272205e-05),
    on_grid=c_grid,
)

ds = multimode_params.as_dataset

H = abs(depth[0] - depth[-1])
A = 1.33e-7 / H
gamma = A / ds.c.values**2
multimode_params.__setattr__("gamma_h", (A / ds.c**2).values)
multimode_params.__setattr__("gamma_v", (A / ds.c**2).values)


tau_x = np.empty(c_grid.u.shape)
for k in range(nmodes):
    tau_x[k, :, :] = -0.05 * ds.psi.values[:, k]
tau_x *= c_grid.u.mask


def zonal_wind(state, params):
    """Zonal wind."""
    return State(u=Variable(tau_x / params.rho_0 / H, c_grid.u, np.datetime64("NaT")))


terms = [
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
    advection_density,
    advection_momentum_u,
    advection_momentum_v,
    zonal_wind,
]


def rhs(state, params):
    """RHS."""
    w = (divergence_j(state, params) + divergence_i(state, params)).eta.safe_data
    state.set_diagnostic_variable(w=Variable(w, c_grid.eta, state.eta.time))
    return ft.reduce(op.add, (term(state, params) for term in terms))


def save_as_Dataset(state: State, params: MultimodeParameter):
    """Save output as xarray.Dataset."""
    ds = state.variables["u"].as_dataarray.to_dataset(name="u_tilde")
    ds["v_tilde"] = state.variables["v"].as_dataarray
    ds["h_tilde"] = state.variables["eta"].as_dataarray
    x = (["j", "i"], (state.variables["u"].grid.x + state.variables["v"].grid.x) / 2)
    y = (["j", "i"], (state.variables["u"].grid.y + state.variables["v"].grid.y) / 2)
    ds.assign_coords({"x": x, "y": y})
    return ds


time = 3 * 365 * 24 * 3600.0  # 1 year
step = c_grid.u.dx.min() / ds.c.values.max() / 10.0
t0 = np.datetime64("2000-01-01")


def run(params, step, time):
    """Run the model setup."""
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
    tol = 0
    for i, next_state in enumerate(model_run):
        if i % (Nt // 5) == 0:
            output.append(save_as_Dataset(next_state, params))
        elif i >= (Nt - 4):
            output.append(save_as_Dataset(next_state, params))
        if np.nanmax(abs(next_state.variables["u"].safe_data)) > 100:
            tol += 1
            output.append(save_as_Dataset(next_state, params))
        if tol > 5:
            return xr.combine_by_coords(output)

    return xr.combine_by_coords(output)


out = run(multimode_params, step=step, time=time)

out = out.rename({"z": "nmode"})

out["u"] = xr.dot(ds.psi, out.u_tilde)
out["v"] = xr.dot(ds.psi, out.v_tilde)
out["h"] = xr.dot(ds.psi, out.h_tilde)

np.save("u_3year_run.npy", out.u.values)
np.save("v_3year_run.npy", out.v.values)
np.save("h_3year_run.npy", out.h.values)
np.save("t_3year_run.npy", out.time.values)
