"""Running a comparison run of one year linear vs nonlinear."""

import xarray as xr
import numpy as np
import functools as ft
import operator as op

from multimodemodel import StaggeredGrid
from multimodemodel import MultimodeParameter, f_on_sphere
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
    lat_start=-10.125,
    lat_end=10.125,
    nx=50 * 4 + 1,
    ny=20 * 4 + 2,
    z=np.arange(nmodes),
)

Nsq = np.load("Nsq.npy", allow_pickle=True)
depth = Nsq[1, :]
Nsq = Nsq[0, :]

multimode_params = MultimodeParameter(
    z=depth,
    Nsq=Nsq,
    nmodes=nmodes,
    coriolis_func=f_on_sphere(omega=7.272205e-05),
    on_grid=c_grid,
    no_slip=True,
)
ds = multimode_params.as_dataset

A = 5.5e-5 * np.max(Nsq)
H = abs(depth[0] - depth[-1])
multimode_params.__setattr__("gamma_h", (A / ds.c**2).values)
multimode_params.__setattr__("gamma_v", (A / ds.c**2).values)

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


def save_as_Dataset(state: State):
    """Save State object as xarray.Dataset."""
    return xr.Dataset(
        data_vars=dict(
            u_tilde=(
                ["k", "j", "i", "time"],
                np.expand_dims(state.variables["u"].safe_data, axis=-1),
            ),
            v_tilde=(
                ["k", "j", "i", "time"],
                np.expand_dims(state.variables["v"].safe_data, axis=-1),
            ),
            eta_tilde=(
                ["k", "j", "i", "time"],
                np.expand_dims(state.variables["eta"].safe_data, axis=-1),
            ),
        ),
        coords=dict(
            x_u=(["j", "i"], state.variables["u"].grid.x),
            y_u=(["j", "i"], state.variables["u"].grid.y),
            x_v=(["j", "i"], state.variables["v"].grid.x),
            y_v=(["j", "i"], state.variables["v"].grid.y),
            nmode=(("k",), state.variables["eta"].grid.z),
            time=(("time",), [state.variables["u"].time]),
        ),
        attrs=dict(description="Saves dynamic variables with grid information."),
    )


time = 2 * 365 * 24 * 3600.0  # 2 years
step = c_grid.u.dx.min() / ds.c.values.max() / 10.0  # satisfying the CFL criterion
t0 = np.datetime64("2000-01-01")  # starting date
output_freq = 5 * 24 * 3600.0  # 5 day mean


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

    u = np.zeros((1,) + c_grid.u.shape)
    v = np.zeros((1,) + c_grid.v.shape)
    eta = np.zeros((1,) + c_grid.eta.shape)
    t = t0
    n = 1
    output = []
    tolerance = 100
    for i, next_state in enumerate(model_run):
        u += next_state.variables["u"].safe_data
        v += next_state.variables["v"].safe_data
        eta += next_state.variables["eta"].safe_data
        n += 1
        if i % (Nt // (time // output_freq)) == 0:
            t_mean = t + (next_state.u.time - t) / 2
            output.append(
                save_as_Dataset(
                    State(
                        u=Variable(np.squeeze(u / n), c_grid.u, t_mean),
                        v=Variable(np.squeeze(v / n), c_grid.v, t_mean),
                        eta=Variable(np.squeeze(eta / n), c_grid.eta, t_mean),
                        q=Variable(None, c_grid.q, t_mean),
                    )
                )
            )
            t = t_mean
            n = 0
        if np.max(abs(next_state.variables["u"].safe_data)) >= tolerance:
            return xr.combine_by_coords(output)

    return xr.combine_by_coords(output)


out_linear = run(linear_model, multimode_params, step, time)
out_linear.to_netcdf("linear_5d.nc")

out_nonlinear = run(nonlinear_model, multimode_params, step, time)
out_nonlinear.to_netcdf("nonlinear_5d.nc")
