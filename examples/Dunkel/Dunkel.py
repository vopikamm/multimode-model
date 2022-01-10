"""Scripted run of Markus Dunkels model."""

import xarray as xr
import numpy as np

from multimodemodel import (
    StaggeredGrid,
    MultimodeParameters,
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

multimode_params = MultimodeParameters(
    z=depth,
    Nsq=Nsq,
    nmodes=25,
    coriolis_func=f_on_sphere(omega=7.272205e-05),
    on_grid=c_grid,
)

ds = multimode_params.as_dataset

A = 1.33e-7
gamma = A / ds.c.values ** 2
multimode_params.gamma = gamma


def tau(x, y):
    """Wind field according to Mccreary (1980)."""
    delta_x = abs(x[0, 0] - x[0, -1]) / 2
    delta_y = abs(y[0, 0] - y[-1, 0]) / 2

    wind_x = np.cos(np.pi * x / delta_x)
    wind_x[abs(x) > delta_x / 2] = 0

    wind_y = (1 + y ** 2 / delta_y ** 2) * np.exp(-(y ** 2) / delta_y ** 2)

    return -5e-6 * wind_x * wind_y


tau_x = np.empty(c_grid.u.shape)
for k in range(nmodes):
    tau_x[k, :, :] = tau(c_grid.u.x, c_grid.u.y) * ds.p_hat.values[k, 0]


def zonal_wind(state, params):
    """Zonal wind."""
    return State(u=Variable(tau_x / params.rho_0, c_grid.u, np.datetime64("NaT")))


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


def save_as_Dataset(state: State, params: MultimodeParameters):
    """Save output as xarray.Dataset."""
    ds = state.variables["u"].as_dataarray.to_dataset(name="u_tilde")
    ds["v_tilde"] = state.variables["v"].as_dataarray
    ds["h_tilde"] = state.variables["eta"].as_dataarray
    x = (["j", "i"], (state.u.grid.x + state.v.grid.x) / 2)
    y = (["j", "i"], (state.u.grid.y + state.v.grid.y) / 2)
    ds.assign_coords({"x": x, "y": y})
    return ds


time = 365 * 24 * 3600.0  # 1 year
step = c_grid.u.dx.min() / ds.c.values[1:].max() / 10.0
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

    for i, next_state in enumerate(model_run):
        if i % (Nt // 5) == 0:
            output.append(save_as_Dataset(next_state, params))

    return xr.combine_by_coords(output)


warmup = run(multimode_params, step=1500.0, time=6000.0)


out = run(multimode_params, step=step, time=time)

out = out.rename({"z": "nmode"})

out["u"] = xr.dot(ds.p_hat, out.u_tilde)
out["v"] = xr.dot(ds.p_hat, out.v_tilde)
out["h"] = xr.dot(ds.w_hat, out.h_tilde)

np.save("u.npy", out.u.values)
np.save("v.npy", out.v.values)
np.save("h.npy", out.h.values)
