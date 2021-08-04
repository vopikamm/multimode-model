"""General datastructures.

Dataclasses for building instances to hold parameters, dynamic variables
and their associated grids.
"""

import numpy as np
from dataclasses import dataclass
from .grid import Grid


@dataclass
class Parameters:
    """Class to organise all constant parameters."""

    f: float = 0.0
    g: float = 9.81  # gravitational force m/s^2
    beta: float = 2.0 / (24 * 3600)  # beta parameter 1/ms with f=f_0+beta *y
    H: float = 1000.0  # reference depth in m
    rho_0: float = 1024.0  # reference density in kg / m^3
    dt: float = 1.0  # time stepping in s
    t_0: float = 0.0  # starting time
    t_end: float = 3600.0  # end time
    write: int = 20  # how many states should be output
    r: float = 6371000.0  # radius of the earth


@dataclass
class Variable:
    """Variable class consisting of the data and a Grid instance."""

    data: np.array
    grid: Grid

    def __add__(self, other):
        """Add data of to variables."""
        if (
            # one is subclass of the other
            (isinstance(self, type(other)) or isinstance(other, type(self)))
            and self.grid is not other.grid
        ):
            raise ValueError("Try to add variables defined on different grids.")
        try:
            new_data = self.data + other.data
        except (TypeError, AttributeError):
            return NotImplemented
        return self.__class__(data=new_data, grid=self.grid)


@dataclass
class State:
    """State class.

    Combines the dynamical variables u,v, eta into one state object.
    """

    u: Variable
    v: Variable
    eta: Variable

    def __add__(self, other):
        """Add all variables of two states."""
        if not isinstance(other, type(self)) or not isinstance(self, type(other)):
            return NotImplemented
        try:
            u_new = self.u + other.u
            v_new = self.v + other.v
            eta_new = self.eta + other.eta
        except (AttributeError, TypeError):
            return NotImplemented
        return self.__class__(u=u_new, v=v_new, eta=eta_new)
