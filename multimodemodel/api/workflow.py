"""API of Workflow managers."""

from abc import ABC, abstractmethod
from math import ceil
from typing import Union

import numpy as np
from multimodemodel.api.core import DomainBase
from multimodemodel.api.integration import SolverBase

from multimodemodel.util import average_npdatetime64, str_to_date


def _to_datetime64(date: Union[str, np.datetime64]) -> np.datetime64:
    if isinstance(date, np.datetime64):
        return date
    return str_to_date(date)


class WorkflowBase(ABC):
    """Base class for Workflow managers."""

    domain: DomainBase
    solver: SolverBase

    @abstractmethod
    def run(self, steps: int) -> None:  # pragma: no cover
        """Run model forward for a number of steps.

        Arguments
        ---------
        steps: int
            Number of time steps to integrate.
        """
        ...

    def run_until(self, date: Union[str, np.datetime64]):
        """Run until specified date.

        The model will run at least until the specified date.
        The final date of the state will be the earlies date that is
        larger or equal to `date`. A ValueError is raised if `date` is before
        the date of the present state.

        Arguments
        ---------
        date: Union[str, np.datetime64]
            Date until the model runs. Either a ISO date string or a datetime64 object.
        """
        date = _to_datetime64(date)
        dt = self.solver.step
        t_state = average_npdatetime64(
            tuple(v.time for v in self.domain.state.variables.values())
        )

        # requested model run time in secods
        t_delta = (date - t_state) / np.timedelta64(1, "s")

        nt = int(ceil(t_delta / dt))

        if nt < 0:
            raise ValueError(
                "End date is before present date. End date: {date}, present date: {t_state}."
            )

        self.run(steps=nt)
