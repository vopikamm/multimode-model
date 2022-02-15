"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

# from os import initgroups
# from numba.core.targetconfig import Option
import numpy as np
from dataclasses import dataclass, field, asdict, InitVar

from .grid import Grid, StaggeredGrid
from .coriolis import CoriolisFunc
from typing import Dict, Optional, Tuple

try:
    import xarray

    has_xarray = True
except ModuleNotFoundError:  # pragma: no cover
    has_xarray = False

    # for type hinting
    class xarray:
        """Necessary for type hinting to work."""

        DataArray = None
        Dataset = None


@dataclass
class Parameters:
    """Class to organise all parameters.

    The parameters may be constant in space and/or time.
    Note that the objects `compute_f` method needs to be called after creation
    to provide the coriolis parameter on all subgrids of a staggered grid instance.

    Arguments
    ---------
    (of the __init__ method)

    g: float = 9.81
      Gravitational acceleration in [m/s^2]
    H: float = 1000.0
      Depth of the fluid or thickness of the undisturbed layer in [m]
    rho_0: float = 1024.0
      Reference density of sea water in [kg / m^3]
    coriolis_func: Optional[CoriolisFunc] = None
      Function used to compute the coriolis parameter on each subgrid
      of a staggered grid. The signature of this function must match
      `coriolis_func(y: numpy.ndarray) -> numpy.ndarray`
      and they are called with the y coordinate of the respective grid.
    on_grid: Optional[StaggeredGrid] = None
      StaggeredGrid object providing the necessary grid information if a
      parameter depends on space, such as the Coriolis parameter. Only
      required if such a parameter is part of the system to solve.


    Attributes
    ----------
    f: Dict[str, numpy.ndarray]
      Mapping of the subgrid names ('u', 'v', 'eta') to the coriolis
      parameter on those grids
    """

    g: float = 9.81  # gravitational force m / s^2
    H: np.ndarray = np.array([1000.0])  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    a_h: float = 2000.0  # horizontal mixing coefficient in m^2 / s
    gamma_h: np.ndarray = np.array([0.0])  # horizontal damping coefficient in 1 / s
    gamma_v: np.ndarray = np.array([0.0])  # vertical damping coefficient in 1 / s
    free_slip: bool = True  # lateral boundary conditions
    no_slip: bool = False  # lateral boundary conditions
    coriolis_func: InitVar[Optional[CoriolisFunc]] = None
    on_grid: InitVar[Optional[StaggeredGrid]] = None
    _f: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(
        self,
        coriolis_func: Optional[CoriolisFunc],
        on_grid: Optional[StaggeredGrid],
    ):
        """Set lateral boundary conditions."""
        if self.no_slip == self.free_slip:
            self.no_slip = True
            self.free_slip = False
        """Initialize derived fields."""
        self._f = self._compute_f(coriolis_func, on_grid)

    @property
    def f(self) -> Dict[str, np.ndarray]:
        """Getter of the dictionary holding the Coriolis parameter."""
        if not self._f:
            raise RuntimeError(
                "Coriolis parameter not available. "
                "Parameters object must be created with both `coriolis_func` "
                "and `on_grid` argument."
            )
        return self._f

    def _compute_f(
        self, coriolis_func: Optional[CoriolisFunc], grids: Optional[StaggeredGrid]
    ) -> Dict[str, np.ndarray]:
        """Compute the coriolis parameter for all subgrids.

        This method needs to be called before a rotating system
        can be set up. Returns None but set the attribute `f` of the object.

        Arguments
        ---------
        grids: StaggeredGrid
          Grids on which the coriolis parameter shall be provided.
        """
        if coriolis_func is None or grids is None:
            return {}
        _f = {name: coriolis_func(grid["y"]) for name, grid in asdict(grids).items()}
        return _f


@dataclass
class MultimodeParameters(Parameters):
    """Class to organise mode dependant parameters.

    Arguments
    ---------
    (of the __init__ method)

    rho: Optional[np.ndarray] = None
      Background density profile in [kg/m^3]
    Nsq: Optional[np.ndarray] = None
      Brunt-Vaisala frequencies squared in [1/s^2]
    z: Optional[np.ndarray] = None
      Depth coordinate in [m]
    nmodes: Optional[int] = None
      Number of vertical normal modes.
      Cannot be larger than len(z) - 2


    Attributes
    ----------
    psi: Optional[np.ndarray] = field(init=False)
      Vertical structure function of the pressure.
      Matrix of dimension (number of modes x number of z levels)
    dpsi_dz: Optional[np.ndarray] = field(init=False)
      Vertical structure functions of the vertical velocity.
      Matrix of dimension (number of modes x number of z levels)
    c: Optional[np.ndarray] = field(init=False)
      Gravity wave speed in [m/s].
    """

    rho: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    nmodes: int = 1
    Nsq: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = field(init=False)
    dpsi_dz: Optional[np.ndarray] = field(init=False)
    c: Optional[np.ndarray] = field(init=False)
    Q: Optional[np.ndarray] = field(init=False)
    R: Optional[np.ndarray] = field(init=False)
    S: Optional[np.ndarray] = field(init=False)
    T: Optional[np.ndarray] = field(init=False)

    def __post_init__(self, coriolis_func, on_grid):
        """Derive mode-dependent parameters from stratification."""
        super().__post_init__(coriolis_func, on_grid)
        if self.z is None:
            print("No vertical axis given. Fields cannot be initialized.")
            self.psi, self.dpsi_dz, self.c = (
                None,
                None,
                None,
            )
        else:
            if self.rho is None:
                if self.Nsq is None:
                    print(
                        "No valid stratification given. Fields cannot be initialized."
                    )
                    self.psi, self.dpsi_dz, self.c = (
                        None,
                        None,
                        None,
                    )
                else:
                    (self.psi, self.dpsi_dz, self.c) = self.modal_decomposition(
                        self.z, self.Nsq
                    )
                    self.H = self.c ** 2 / self.g
            if self.Nsq is None:
                if self.rho is None:
                    print(
                        "No valid stratification given. Fields cannot be initialized."
                    )
                    self.psi, self.dpsi_dz, self.c = (
                        None,
                        None,
                        None,
                    )
                else:
                    Nsq_depth = np.zeros_like(self.z)
                    Nsq_depth[1:] = (self.z[: self.z.size - 1] + self.z[1:]) / 2
                    self.Nsq = self.N_squared(self.z, self.rho)
                    (self.psi, self.dpsi_dz, self.c) = self.modal_decomposition(
                        Nsq_depth, self.Nsq
                    )
                    self.H = self.c ** 2 / self.g
        self.Q = self.compute_Q()
        self.R = self.compute_R()
        self.S = self.compute_S()
        self.T = self.compute_T()

    @property
    def as_dataset(self) -> xarray.Dataset:  # type: ignore
        """Return the mode-dependent parameters as xarray.Dataset."""
        if not has_xarray:
            raise ModuleNotFoundError(  # pragma: no cover
                "Cannot convert parameters to xarray.DataArray. Xarray is not available."
            )
        if self.z is None:
            raise ValueError(
                "Cannot convert parameters to xarray.DataSet. No vertical coordinate given."
            )
        attributes_dict = dict(
            psi=(["depth", "nmode"], self.psi),
            dpsi_dz=(["depth", "nmode"], self.dpsi_dz),
            c=("nmode", self.c),
            H=("nmode", self.H),
            Nsq=("depth", self.Nsq),
            rho=("depth", self.rho),
        )
        ds = xarray.Dataset(  # type: ignore
            coords=dict(nmode=np.arange(self.nmodes), depth=self.z)
        )
        for key in attributes_dict:
            param = attributes_dict[key][1]
            if param is not None:
                param_copy = param.copy()
                ds[key] = (attributes_dict[key][0], param_copy)
        return ds

    def N_squared(self, z: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Compute the Brunt-Vaisala frequencies squared."""
        if z.shape != rho.shape:
            raise ValueError("Shape of depth and density profile missmatch.")
        Nsq = np.zeros_like(rho)
        Nsq[1:] = np.diff(rho) * self.g / (np.diff(z) * self.rho_0)
        Nsq[0] = Nsq[1]
        Nsq[Nsq < 0] = 0
        return Nsq

    def d2dz2_matrix_p(self, z, Nsq) -> np.ndarray:
        """2nd vertical derivative operator.

        Build the matrix that discretizes the 2nd derivative
        over the vertical coordinate, and applies the boundary conditions for
        p-mode.
        """
        dz = np.diff(z)
        z_mid = z[1:] - dz / 2.0
        dz_mid = np.diff(z_mid)
        dz_mid = np.r_[dz[0] / 2.0, dz_mid, dz[-1] / 2.0]

        Ndz = Nsq * dz_mid

        d0 = np.r_[
            1.0 / Ndz[1] / dz[0],
            (1.0 / Ndz[2:-1] + 1.0 / Ndz[1:-2]) / dz[1:-1],
            1.0 / Ndz[-2] / dz[-1],
        ]
        d1 = -1.0 / Ndz[1:-1] / dz[:-1]
        dm1 = -1.0 / Ndz[1:-1] / dz[1:]

        d2dz2 = np.diag(d0) + np.diag(d1, k=1) + np.diag(dm1, k=-1)
        return d2dz2

    def modal_decomposition(self, z, Nsq) -> Tuple:
        """Compute vertical structure functions and gravity wave speeds."""
        d2dz2_p = self.d2dz2_matrix_p(z, Nsq)
        eigenvalues_p, pmodes = np.linalg.eig(d2dz2_p)

        # Filter out complex-values and small/negative eigenvalues
        mask = np.logical_and(eigenvalues_p >= 1e-10, eigenvalues_p.imag == 0)
        eigenvalues_p = eigenvalues_p[mask]
        pmodes = pmodes[:, mask]

        # Sort eigenvalues and modes and truncate to number of modes requests
        index = np.argsort(eigenvalues_p)
        eigenvalues_p = eigenvalues_p[index[: self.nmodes]]
        pmodes = pmodes[:, index[: self.nmodes]]

        # Modal speeds
        c = 1 / np.sqrt(eigenvalues_p)

        # Normalze mode structures to satisfy \int_{-H}^0 \hat{p}^2 dz = H
        dz = np.diff(z)[:, np.newaxis]
        factor = np.sqrt(np.abs(dz.sum() / ((pmodes ** 2.0) * dz).sum(axis=0)))
        pmodes *= factor[np.newaxis, :]

        # unify sign, that pressure modes are always positive at the surface
        sig_p = np.sign(pmodes[0, :])
        sig_p[sig_p == 0.0] = 1.0
        pmodes = sig_p[np.newaxis, :] * pmodes

        # Compute w_hat so that w_hat = - g / N^2 * dp_hat / dz
        wmodes = np.diff(pmodes, axis=0) / dz[:-1]
        # Boundary conditions w_hat = 0 at z = {0, -H}
        wmodes = np.concatenate(
            (np.zeros((1, self.nmodes)), wmodes, np.zeros((1, self.nmodes))), axis=0
        )

        # map p_hat on vertical grid of w_hat
        pmodes = np.concatenate((pmodes[[0], :], pmodes, pmodes[[-1], :]), axis=0)
        pmodes = pmodes[1:, :] - np.diff(pmodes, axis=0) / 2

        return (pmodes, wmodes, c)

    def compute_Q(self) -> Optional[np.ndarray]:
        """Compute the triple-mode-tensor PPP for the nonlinear terms.

        The array elements correspond to:
            (1 / H) * int_{-H}^{0} psi[k, :] * psi[m, :] * psi[n, :] dz.
        """
        if self.psi is None or self.z is None:
            return None
        tensor = np.empty((self.nmodes, self.nmodes, self.nmodes))
        for n in range(self.nmodes):
            for m in range(self.nmodes):
                for k in range(self.nmodes):
                    tensor[n, m, k] = np.trapz(
                        self.psi[:, n] * self.psi[:, m] * self.psi[:, k],
                        self.z,
                    )
        return tensor / abs(self.z[-1] - self.z[0])

    def compute_R(self) -> Optional[np.ndarray]:
        """Compute the triple-mode-tensor dPdPP for the nonlinear terms.

        The array elements correspond to:
            (g / H) * int_{-H}^{0} psi[k, :] * dpsi_dz[m, :] * dpsi_dz[n, :] / Nsq dz.
        """
        if (
            self.Nsq is None
            or self.dpsi_dz is None
            or self.z is None
            or self.psi is None
        ):
            return None
        tensor = np.empty((self.nmodes, self.nmodes, self.nmodes))
        for n in range(self.nmodes):
            for m in range(self.nmodes):
                for k in range(self.nmodes):
                    tensor[n, m, k] = np.trapz(
                        self.psi[:, k]
                        * self.dpsi_dz[:, m]
                        * self.dpsi_dz[:, n]
                        / self.Nsq,
                        self.z,
                    )
        return tensor * self.g / abs(self.z[-1] - self.z[0])

    def compute_S(self) -> Optional[np.ndarray]:
        """Compute the triple-mode-tensor dPdPP for the nonlinear terms.

        The array elements correspond to:
            (1 / H) * int_{-H}^{0} dpsi_dz[k, :] * dpsi_dz[m, :] * psi[n, :] dz.
        """
        if (
            self.Nsq is None
            or self.dpsi_dz is None
            or self.z is None
            or self.psi is None
        ):
            return None
        tensor = np.empty((self.nmodes, self.nmodes, self.nmodes))
        for n in range(self.nmodes):
            for m in range(self.nmodes):
                for k in range(self.nmodes):
                    tensor[n, m, k] = np.trapz(
                        self.dpsi_dz[:, k] * self.dpsi_dz[:, m] * self.psi[:, n],
                        self.z,
                    )
        return tensor / abs(self.z[-1] - self.z[0])

    def compute_T(self) -> Optional[np.ndarray]:
        """Compute the triple-mode-tensor WWW for the nonlinear terms.

        The array elements correspond to:
            (g / H) * int_{-H}^{0} dpsi_dz[k, :] * d2psi_dz[m, :] * dpsi_dz[n, :] / Nsq dz.
        """
        if (
            self.Nsq is None
            or self.dpsi_dz is None
            or self.z is None
            or self.psi is None
        ):
            return None

        d2psi_dz = np.gradient(self.dpsi_dz, self.z, axis=0)
        tensor = np.empty((self.nmodes, self.nmodes, self.nmodes))
        for n in range(self.nmodes):
            for m in range(self.nmodes):
                for k in range(self.nmodes):
                    tensor[n, m, k] = np.trapz(
                        self.dpsi_dz[:, k]
                        * d2psi_dz[:, m]
                        * self.dpsi_dz[:, n]
                        / self.Nsq,
                        self.z,
                    )
        return tensor * self.g / abs(self.z[-1] - self.z[0])


@dataclass
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: Optional[np.ndarray]
    grid: Grid
    time: np.datetime64

    def __post_init__(self):
        """Validate."""
        if self.data is not None:
            if self.data.shape != self.grid.shape:
                raise ValueError(
                    "Shape of data and grid missmatch. "
                    f"Got {self.data.shape} and {self.grid.shape}"
                )

    @property
    def as_dataarray(self) -> xarray.DataArray:  # type: ignore
        """Return variable as xarray.DataArray.

        The DataArray object contains a copy of (not a reference to) the data of
        the variable. The coordinates are multidimensional arrays to support
        curvilinear grids and copied from the grids `x` and `y` attribute. Grid
        points for which the mask of the grid equals to 0 are converted to NaNs.
        """
        if not has_xarray:
            raise ModuleNotFoundError(  # pragma: no cover
                "Cannot convert variable to xarray.DataArray. Xarray is not available."
            )

        # copy to prevent side effects on self.data
        data = self.safe_data.copy()
        data[self.grid.mask == 0] = np.nan
        data = np.expand_dims(data, axis=0)

        coords = dict(
            x=(("j", "i"), self.grid.x),
            y=(("j", "i"), self.grid.y),
            z=(("z",), self.grid.z),
        )
        dims = ("z", "j", "i")

        dims = ("time",) + dims
        coords["time"] = (("time",), [self.time])  # type: ignore

        return xarray.DataArray(  # type: ignore
            data=data,
            coords=coords,
            dims=dims,
        )

    @property
    def safe_data(self) -> np.ndarray:
        """Return self.data or, if it is None, a zero array of appropriate shape."""
        if self.data is None:
            return np.zeros(self.grid.shape)
        else:
            return self.data

    def copy(self):
        """Return a copy.

        The data attribute is a deep copy while the grid is a reference.
        """
        if self.data is None:
            data = None
        else:
            data = self.data.copy()
        return self.__class__(data, self.grid, self.time.copy())

    def __add__(self, other):
        """Add data of two variables.

        The timestamp of the sum of two variables is set to their mean.
        """
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")

        try:
            if self.data is None and other.data is None:
                new_data = None
            elif self.data is None:
                new_data = other.data.copy()  # type: ignore
            elif other.data is None:
                new_data = self.data.copy()
            else:
                new_data = self.data + other.data
        except (AttributeError, TypeError):
            return NotImplemented

        new_time = self.time + (other.time - self.time) / 2

        return self.__class__(data=new_data, grid=self.grid, time=new_time)


class State:
    """State class.

    Combines the prognostic variables into one state object.

    The variables are passed as keyword arguments to the __init__ method
    of the state object and stored in the attribute `variables` which is a
    dict.

    State objects can be added such that the individual variable objects are added.
    If an variable is missing in one state object, it is treated as zeros.

    For convenience, it is also possible to access the variables directly as atttributes
    of the state object.
    """

    def __init__(self, **kwargs):
        """Create State object.

        Variables are given by keyword arguments.
        """
        self.variables = dict()

        for k, v in kwargs.items():
            if type(v) is not Variable:
                raise ValueError("Keyword arguments must be of type Variable.")
            else:
                self.variables[k] = v
                self.__setattr__(k, self.variables[k])

    def __add__(self, other):
        """Add all variables of two states.

        If one of the state object is missing a variable, this variable is copied
        from the other state object. Note that this means, that the time stamp of
        this particular variable will remain unchanged.
        """
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented  # pragma: no cover
        try:
            sum = dict()
            for k in self.variables:
                if k in other.variables:
                    sum[k] = self.variables[k] + other.variables[k]
                else:
                    sum[k] = self.variables[k].copy()
            for k in other.variables:
                if k not in self.variables:
                    sum[k] = other.variables[k].copy()
            return self.__class__(**sum)
        except (AttributeError, TypeError):  # pragma: no cover
            return NotImplemented

    def set_diagnostic_variable(self, **kwargs):
        """Set variables for diagnostic purposes.

        Diagnostic variables are given by keyword arguments.
        Attributes are not considered by the add function.
        """
        self.diagnostic_variables = dict()

        for k, v in kwargs.items():
            if type(v) is not Variable:
                raise ValueError("Keyword arguments must be of type Variable.")
            else:
                self.diagnostic_variables[k] = v
                self.__setattr__(k, self.diagnostic_variables[k])
