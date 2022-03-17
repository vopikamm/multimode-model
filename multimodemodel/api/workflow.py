"""API of Workflow managers."""

from abc import ABC, abstractmethod
from math import ceil
from typing import Optional, Union, Protocol

import numpy as np
from multimodemodel.api.core import DomainBase
from multimodemodel.api.integration import SolverBase

from multimodemodel.util import average_npdatetime64, str_to_date


class DiagnosticCallback(Protocol):
    """Protocol for diagnostic callback."""

    def __call__(self, domain: DomainBase) -> None:
        """Define signature of diagnostic callback functions."""
        ...


def _default_diag(domain: DomainBase) -> None:
    """Do nothing by default."""
    ...


def _to_datetime64(date: Union[str, np.datetime64]) -> np.datetime64:
    if isinstance(date, np.datetime64):
        return date
    return str_to_date(date)


class WorkflowBase(ABC):
    """Base class for Workflow managers."""

    domain: DomainBase
    solver: SolverBase

    def run(
        self, steps: int, diag: Optional[DiagnosticCallback] = None
    ) -> None:  # pragma: no cover
        """Run model forward for a number of steps.

        Before and after the iteration, `self.run_setup()` and
        `self.run_teardown()` is called. For each iteration,
        `self.run_next()`, which computes the domain state
        at the next time step, and `self.run_diag(diag)`, which runs
        the diagnostic function, are called.

        Arguments
        ---------
        steps: int
            Number of time steps to integrate.
        diag: Optional[(DomainBase) -> None]
            Function called with each new iteration result for
            online diagnostics purposes.
        """
        if diag is None:
            diag = _default_diag
        self._run_setup()
        for _ in range(steps):
            self._run_next()
            self._run_diag(diag)
        self._run_teardown()

    def _run_setup(self) -> None:
        """Set up the iteration."""
        ...

    @abstractmethod
    def _run_next(self) -> None:
        """Compute the state at the next time step."""
        ...

    def _run_diag(self, diag: DiagnosticCallback) -> None:
        """Perform diagnostics."""
        ...

    def _run_teardown(self) -> None:
        """Clean up and finalize after iteration."""
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
