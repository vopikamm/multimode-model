"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

# from os import initgroups
# from numba.core.targetconfig import Option
import numpy as np
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Union, Type, Optional, Hashable, Any, Tuple
from types import ModuleType

from .config import config
from multimodemodel.api import (
    DomainBase,
    ParameterBase,
    StateBase,
    StateDequeBase,
    VariableBase,
    Array,
    MergeVisitorBase,
    SplitVisitorBase,
)
from .jit import sum_arr
from .grid import Grid, StaggeredGrid
from .coriolis import CoriolisFunc

xarray: Union[ModuleType, Type["xr_mockup"]]
try:
    import xarray as xr

    has_xarray = True
    xarray = xr

except ModuleNotFoundError:  # pragma: no cover
    has_xarray = False

    class xr_mockup:
        """Necessary for type hinting to work."""

        class DataArray:
            """Necessary for type hinting to work."""

            ...

        class Dataset:
            """Necessary for type hinting to work."""

            ...

    xarray = xr_mockup


def _set_attr(obj: object, name: str, value: Any, default: Any = None):
    if value is None:
        obj.__setattr__(name, default)
    else:
        obj.__setattr__(name, value)


@dataclass(frozen=True)
class Parameter(ParameterBase):
    """Class to organise all parameters.

    The parameters may be constant in space and/or time.
    Note that the objects `compute_f` method needs to be called after creation
    to provide the coriolis parameter on all subgrids of a staggered grid instance.

    Parameters
    ----------
    g : float, default=9.81
      Gravitational acceleration m/s^2
    H : float or Array, default=1000.0
      Depth of the fluid or thickness of the undisturbed layer in m
    rho_0 : float, default=1024.0
      Reference density of sea water in kg / m^3
    coriolis_func : Optional[CoriolisFunc], default=None
      Function used to compute the coriolis parameter on each subgrid
      of a staggered grid. It is called with the y coordinate of the respective grid.
    on_grid : StaggeredGrid, default=None
      StaggeredGrid object providing the necessary grid information if a
      parameter depends on space, such as the Coriolis parameter. Only
      required if such a parameter is part of the system to solve.

    Attributes
    ----------
    f: dict[str, numpy.ndarray]
      Mapping of the subgrid names (e.g. "u", "v", "eta") to the coriolis
      parameter on those grids
    """

    g: float = 9.81  # gravitational force m / s^2
    H: np.ndarray = np.array([1000.0])  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    a_h: float = 2000.0  # coefficient for horizontal eddie viscosity in m^2 / s
    a_v: float = 1e3  # coefficient for vertical eddie viscosity in m^2 / s
    b_h: float = -2e10  # coefficient for biharmonic mixing in m^4 / s
    k_h: np.ndarray = np.array([0.0])  # horizontal thermal diffusivity in m^2 / s
    k_v: np.ndarray = np.array([0.0])  # vertical thermal diffusivity in m^2 / s
    free_slip: bool = True  # lateral boundary conditions
    no_slip: bool = False  # lateral boundary conditions
    _f: dict[str, Array] = field(init=False)
    _id: int = field(init=False)

    def __init__(
        self,
        g: float = 9.80665,
        H: np.ndarray = np.array([1000.0]),
        rho_0: float = 1024.0,
        a_h: float = 2000.0,  # horizontal eddie viscosity in m^2 / s
        a_v: float = 1e3,  # horizontal eddie viscosity in m^2 / s
        b_h: float = -2e10,  # coefficient for biharmonic mixing in m^4 / s
        k_h: Optional[np.ndarray] = None,  # horizontal thermal diffusivity in m^2 / s
        k_v: Optional[np.ndarray] = None,  # vertical thermal diffusivity in m^2 / s
        free_slip: bool = True,  # lateral boundary conditions
        no_slip: bool = False,  # lateral boundary conditions
        coriolis_func: Optional[CoriolisFunc] = None,
        on_grid: Optional[StaggeredGrid] = None,
        f: Optional[dict[str, Array]] = None,
    ):
        """Initialize Parameter object."""
        _set_attr(super(), "_id", id(self))
        _set_attr(super(), "g", g)
        _set_attr(super(), "H", np.atleast_1d(H))
        _set_attr(super(), "rho_0", rho_0)
        _set_attr(super(), "a_h", a_h)
        _set_attr(super(), "a_v", a_v)
        _set_attr(super(), "b_h", b_h)
        _set_attr(super(), "k_h", k_h, np.array([0.0]))
        _set_attr(super(), "k_v", k_v, np.array([0.0]))

        if f is None:
            _set_attr(super(), "_f", self._compute_f(coriolis_func, on_grid))
        else:
            _set_attr(super(), "_f", f)

        """Set lateral boundary conditions."""
        _set_attr(super(), "no_slip", no_slip or (no_slip == free_slip))
        _set_attr(super(), "free_slip", not self.no_slip)

    def __hash__(self):
        """Return id of instance as hash."""
        return self._id

    @property
    def f(self) -> dict[str, Array]:
        """Getter of the dictionary holding the Coriolis parameter.

        Raises
        ------
        RuntimeError
          Raised when there is no Coriolis parameter computed.
        """
        if not self._f:
            raise RuntimeError(
                "Coriolis parameter not available. "
                "Parameters object must be created with both `coriolis_func` "
                "and `on_grid` argument."
            )
        return self._f

    @lru_cache(maxsize=config.lru_cache_maxsize)
    def split(self, splitter: SplitVisitorBase[np.ndarray]):
        """Split Parameter's spatially dependent data."""
        data = None
        try:
            data = self.f
        except RuntimeError:
            return splitter.parts * (self,)

        # Split array for each key, creating a new dictionary with the same keys
        # but holding lists of arrays
        new = {key: splitter.split_array(data[key]) for key in data}

        # Create list of dictionaries each holding just one part of splitted arrays
        out = [{key: new[key][i] for key in new} for i in range(splitter.parts)]

        return tuple(self._new_with_data(self, o) for o in out)

    @classmethod
    def _new_with_data(cls, template, data: dict[str, Array]):
        """Create instance of this class.

        Scalar parameters are copied from template.
        Dictionary of spatially varying parameters is set by data.

        Arguments
        ---------
        template: Parameter
            Some parameter object to copy scalar parameters from.
        data: dict[str, Array]
            Spatially varying parameters
        """
        # collect constant parameters
        kwargs = {
            f.name: getattr(template, f.name)
            for f in fields(template)
            if f.name not in ("_f", "_id")
        }
        kwargs["f"] = data
        return cls(**kwargs)

    @classmethod
    @lru_cache(maxsize=config.lru_cache_maxsize)
    def merge(cls, others: tuple["Parameter"], merger: MergeVisitorBase):
        """Merge Parameter's spatially varying data."""
        data = {}
        try:
            data = {
                key: merger.merge_array(tuple(o.f[key] for o in others))
                for key in others[0].f
            }
        except RuntimeError:
            pass

        return cls._new_with_data(others[0], data)

    def _compute_f(
        self, coriolis_func: Optional[CoriolisFunc], grids: Optional[StaggeredGrid]
    ) -> dict[str, Array]:
        """Compute the coriolis parameter for all subgrids.

        This method needs to be called before a rotating system
        can be set up.

        Arguments
        ---------
        grids: StaggeredGrid
          Grids on which the coriolis parameter shall be provided.

        Returns
        -------
        dict
          Mapping names of subgrids to arrays of the respective Coriolis parameter.
        """
        if coriolis_func is None or grids is None:
            return {}
        _f = {name: coriolis_func(grid.y) for name, grid in grids.items()}
        return _f

    def __eq__(self, other: Any) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, Parameter):
            return NotImplemented
        if self is other:
            return True
        return all(
            all((self._f[v] == other._f[v]).all() for v in self._f)
            if f.name == "_f"
            else getattr(self, f.name) == getattr(other, f.name)
            for f in fields(self)
            if f.name != "_id"
        )


@dataclass(frozen=True)
class MultimodeParameter(Parameter):
    """Class to organise mode dependent parameters.

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
    **kwargs:
      Additional arguments passed to the init function of the parents class


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

    rho: np.ndarray = np.array([])
    z: np.ndarray = np.array([])
    nmodes: int = 1
    Nsq: np.ndarray = np.array([])
    psi: np.ndarray = np.array([])
    dpsi_dz: np.ndarray = np.array([])
    c: np.ndarray = np.array([])
    P: np.ndarray = np.array([])
    Q: np.ndarray = np.array([])
    R: np.ndarray = np.array([])
    S: np.ndarray = np.array([])
    T: np.ndarray = np.array([])

    def __init__(
        self,
        rho: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        nmodes: Optional[int] = None,
        Nsq: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
        dpsi_dz: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        S: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initialize mode-dependent parameters from stratification."""
        super().__init__(**kwargs)
        _set_attr(super(), "rho", rho, np.array([]))
        _set_attr(super(), "z", z, np.array([]))
        _set_attr(super(), "nmodes", nmodes, 1)
        _set_attr(super(), "Nsq", Nsq, np.array([]))
        _set_attr(super(), "psi", psi, np.array([]))
        _set_attr(super(), "dpsi_dz", dpsi_dz, np.array([]))
        _set_attr(super(), "c", c, np.array([]))
        _set_attr(super(), "P", P, np.array([]))
        _set_attr(super(), "Q", Q, np.array([]))
        _set_attr(super(), "R", R, np.array([]))
        _set_attr(super(), "S", S, np.array([]))
        _set_attr(super(), "T", T, np.array([]))

        if z is None or rho is None or Nsq is None:
            print(
                "No valid stratification or vertical axis given. Fields cannot be initialized."
            )
            return

        # no vertical mode provided.
        if psi is None:
            if Nsq is None:
                Nsq_depth = np.zeros_like(self.z)
                Nsq_depth[1:] = (self.z[: self.z.size - 1] + self.z[1:]) / 2
                _set_attr(super(), "Nsq", self.N_squared(self.z, self.rho))

            psi, dpsi_dz, c = self.modal_decomposition(self.z, self.Nsq)

            _set_attr(super(), "psi", psi, np.array([]))
            _set_attr(super(), "dpsi_dz", dpsi_dz, np.array([]))
            _set_attr(super(), "c", c, np.array([]))
            self.__class__.__mro__[-1].__setattr__(self, "H", self.c**2 / self.g)

            P = self.compute_P()
            Q = self.compute_Q()
            R = self.compute_R()
            S = self.compute_S()
            T = self.compute_T()

            _set_attr(super(), "P", P, np.array([]))
            _set_attr(super(), "Q", Q, np.array([]))
            _set_attr(super(), "R", R, np.array([]))
            _set_attr(super(), "S", S, np.array([]))
            _set_attr(super(), "T", T, np.array([]))
        self.__class__.__mro__[-1].__setattr__(self, "_id", id(self))

    def __hash__(self):
        """Return id of instance as hash."""
        return self._id

    def __eq__(self, other: Any) -> bool:
        """Return true if other is identical or the same as self."""
        if not isinstance(other, MultimodeParameter):
            return NotImplemented
        if self is other:
            return True
        equals = []
        for f in fields(self):
            if f.name == "_f":
                equals.append(all((self._f[v] == other._f[v]).all() for v in self._f))
            elif type(getattr(self, f.name)) == np.ndarray:
                equals.append((getattr(self, f.name) == getattr(other, f.name)).all())
            elif f.name == "_id":
                pass
            else:
                equals.append(getattr(self, f.name) == getattr(other, f.name))

        return all(equals)

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
                ds[key] = (attributes_dict[key][0], param_copy)  # type: ignore
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
        factor = np.sqrt(np.abs(dz.sum() / ((pmodes**2.0) * dz).sum(axis=0)))
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

    def compute_P(self) -> Optional[np.ndarray]:
        """Compute the double-mode-tensor P for constant vertical mixing.

        The array elements correspond to:
            (1 / H) * int_{-H}^{0} dpsi_dz[k, :] * dpsi_dz[m, :] dz.
        """
        if self.dpsi_dz is None or self.z is None:
            return None
        tensor = np.empty((self.nmodes, self.nmodes))
        for m in range(self.nmodes):
            for k in range(self.nmodes):
                tensor[m, k] = -np.trapz(
                    self.dpsi_dz[:, m] * self.dpsi_dz[:, k],
                    self.z,
                )
        return tensor / abs(self.z[-1] - self.z[0])

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
                    tensor[n, m, k] = -np.trapz(
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
                    tensor[n, m, k] = -np.trapz(
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
                    tensor[n, m, k] = -np.trapz(
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
                    tensor[n, m, k] = -np.trapz(
                        self.dpsi_dz[:, k]
                        * d2psi_dz[:, m]
                        * self.dpsi_dz[:, n]
                        / self.Nsq,
                        self.z,
                    )
        return tensor * self.g / abs(self.z[-1] - self.z[0])


class Variable(VariableBase[np.ndarray, Grid]):
    """Variable class consisting of the data, a Grid instance and a time stamp.

    A Variable object contains the data for a single time slice of a variable as a Array,
    the grid object describing the grid arrangement and a single time stamp. The data attribute
    can take the value of :py:obj:`None` which is treated like an array of zeros when adding the
    variable to another variable.

    Variable implement summation with another Variable object, see :py:meth:`.Variable.__add__`.

    Parameters
    ----------
    data : Array, default=None
      Array containing a single time slice of a variable. If it is `None`, it will be interpreted
      as zero. To ensure a :py:class:`~numpy.ndarray` as return type, use the property :py:attr:`.safe_data`.
    grid: Grid
      Grid on which the variable is defined.
    time: np.datetime64
      Time stamp of the time slice.

    Raises
    ------
    ValueError
      Raised if `data.shape` does not match `grid.shape`.
    """

    _gtype = Grid

    @property
    def as_dataarray(self) -> xarray.DataArray:  # type: ignore
        """Return variable as :py:class:`xarray.DataArray`.

        The DataArray object contains a copy of (not a reference to) the `data` attribute of
        the variable. The horizontal coordinates are multidimensional arrays to support
        curvilinear grids and copied from the grids `x` and `y` attribute. Grid
        points for which the mask of the grid equals to 0 are converted to NaNs.

        Raises
        ------
        ModuleNotFoundError
          Raised if `xarray` is not present.
        """
        if not has_xarray:
            raise ModuleNotFoundError(  # pragma: no cover
                "Cannot convert variable to xarray.DataArray. Xarray is not available."
            )

        # copy to prevent side effects on self.data
        data = self.safe_data.copy()
        data[self.grid.mask == 0] = np.nan
        data = np.expand_dims(data, axis=0)

        coords: dict[Hashable, Any] = dict(
            x=(("j", "i"), self.grid.x),
            y=(("j", "i"), self.grid.y),
        )
        dims = ["j", "i"]

        if self.grid.ndim >= 3:
            coords["z"] = (("z",), self.grid.z)
            dims.insert(0, "z")

        dims.insert(0, "time")
        coords["time"] = (("time",), [self.time])

        return xarray.DataArray(  # type: ignore
            data=data,
            coords=coords,
            dims=dims,
        )

    @property
    def safe_data(self) -> Array:
        """Return `data` or, if it is `None`, a zero array of appropriate shape."""
        if self.data is None:
            return np.zeros(self.grid.shape)
        else:
            return self.data

    def copy(self):
        """Return a copy.

        `data` and `time` are deep copies while `grid` is a reference.
        """
        if self.data is None:
            data = None
        else:
            data = self.data.copy()
        return self.__class__(data, self.grid, self.time.copy())

    def _add_data(self, other_data: Optional[Array]) -> Optional[Array]:
        if self.data is None and other_data is None:
            new_data = None
        elif self.data is None:
            new_data = other_data.copy()  # type: ignore
        elif other_data is None:
            new_data = self.data.copy()
        else:
            new_data = sum_arr((self.data, other_data))
        return new_data

    def __eq__(self, other):
        """Return true if other is identical or the same as self."""
        if not isinstance(other, Variable):
            return NotImplemented
        if self is other:
            return True
        if (
            self.data is other.data and self.grid == other.grid
        ):  # captures both data attributes are None
            return True
        return all(
            (self.safe_data == other.safe_data).all()
            if f == "data"
            else getattr(self, f) == getattr(other, f)
            for f in self.__slots__
        )

    def _validate_init(self):
        """Validate after initialization."""
        if self.data is not None:
            if self.data.shape != self.grid.shape:
                raise ValueError(
                    f"Shape of data and grid missmatch. Got {self.data.shape} and {self.grid.shape}"
                )


class State(StateBase[Variable]):
    """State class.

    Combines the prognostic variables into a single object.

    The variables are passed as keyword arguments to :py:meth:`__init__`
    and stored in the dict :py:attr:`variables`.

    State objects can be added such that the individual variable objects are added.
    If an variable is missing in one state object, it is treated as zeros.

    For convenience, it is also possible to access the variables directly as atttributes
    of the state object.

    Parameters
    ---------
    `**kwargs` : dict
      Variables are given by keyword arguments.

    Raises
    ------
    ValueError
      Raised if a argument is not of type :py:class:`.Variable`.
    """

    _vtype = Variable


class StateDeque(StateDequeBase[State]):
    """Deque of State objects."""

    _stype = State


class Domain(DomainBase[State, Parameter]):
    """Domain classes.

    Domain objects keep references to the state of the domain, the
    history of the state (i.e. the previous evaluations of the rhs function),
    and the parameters necessary to compute the rhs function.
    """

    _stype = State
    _htype = StateDeque
    _ptype = Parameter
