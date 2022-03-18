"""Workflow manager classes."""

from collections import deque
from itertools import cycle
from typing import Any, Tuple, Type

from multimodemodel.api.core import DomainBase
from multimodemodel.api.integration import SolverBase
from multimodemodel.api.split import RegularSplitMergerBase
from multimodemodel.api.workflow import DiagnosticCallback, WorkflowBase
from multimodemodel.border import Border, Tail


class Workflow(WorkflowBase):
    """Run model on a single process.

    Arguments:
    ----------
    domain: DomainBase
        A domain object that contains the initial state. Using
        a domain object with initialized history will allow for
        a warm restart of a model simulation.
    solver: SolverBase
        The solver defining the problem to solve, the time stepping
        scheme and the time step.
    """

    def __init__(self, domain: DomainBase, solver: SolverBase):
        """Create a simple workflow on a single Python process."""
        self.domain = domain
        self.solver = solver

    def _run_next(self):
        """Run model forward for a single time step."""
        self.domain = self.solver.integrate(self.domain)

    def _run_diag(self, diag: DiagnosticCallback) -> None:
        """Run diagnostics on newly computed state."""
        diag(self.domain)


class DaskWorkflow(WorkflowBase):
    """Workflow distributed on a Dask cluster.

    Arguments:
    ----------
    domain: DomainBase
        A domain object that contains the initial state. Using
        a domain object with initialized history will allow for
        a warm restart of a model simulation.
    solver: SolverBase
        The solver defining the problem to solve, the time stepping
        scheme and the time step.
    client: distributed.Client
        Client to a Dask cluster.
    splitter: SplitVisitorBase
        An instance of a subclass of SplitVisitorBase used to split the domain
        into parts that are distributed across the workers of the client.
    """

    client: Any
    splitter: RegularSplitMergerBase
    domain_stack: deque
    border_stack: deque
    _domain_type: Type[DomainBase]

    def __init__(
        self,
        domain: DomainBase,
        solver: SolverBase,
        client,
        splitter: RegularSplitMergerBase,
    ):
        """Create a client orchestrated workflow."""
        self.domain = domain
        self._domain_type = domain.__class__
        self.solver = solver
        self.client = client
        self.splitter = splitter
        domains, borders = self._split_domain(border_width=2)
        self.domain_stack = deque([domains], maxlen=1)
        self.border_stack = deque([borders], maxlen=1)

    def _run_setup(self) -> None:
        """Set up iteration."""
        self._run_vars = {}
        self._run_vars["tailor"] = self.client.scatter([Tail()], broadcast=True)[0]
        self._run_vars["solver"] = self.client.scatter(
            [self.solver],
            broadcast=True,
        )[0]

    def _run_teardown(self) -> None:
        """Tear down after iteration."""
        del self._run_vars
        self.domain = self.client.submit(
            self._domain_type.merge, self.domain_stack[-1], self.splitter
        )

    def _run_next(self) -> None:
        """Compute state at next time step."""
        splitter = self.splitter
        client = self.client
        workers = self._get_cluster_worker()
        solver = self._run_vars["solver"]
        tailor = self._run_vars["tailor"]
        now_subdomains = self.domain_stack[-1]
        now_borders = self.border_stack[-1]

        new_borders = []
        new_subdomains = []
        for i, (s, worker_of_s) in enumerate(zip(now_subdomains, cycle(workers))):
            borders = client.submit(
                self._pint,
                solver,
                s,
                now_borders[i],
                now_borders[i - 1],
                now_borders[(i + 1) % (splitter.parts)],
                workers=worker_of_s,
            )
            new_s = client.submit(self._int, solver, s, workers=worker_of_s)
            new_subdomains.append(
                client.submit(self._stitch, tailor, new_s, borders, workers=worker_of_s)
            )
            new_borders.append(borders)

        self.domain_stack.append(new_subdomains)
        self.border_stack.append(new_borders)

    def _run_diag(self, diag: DiagnosticCallback) -> None:
        """Run diagnostics on newly computed states."""
        for s in self.domain_stack[-1]:
            diag(s)

    def _get_cluster_worker(self) -> Tuple[str, ...]:
        return tuple(self.client.scheduler_info()["workers"].keys())

    @staticmethod
    def _make_borders(base, width, dim):
        result = (
            Border.create_border(base, width, False, dim),
            Border.create_border(base, width, True, dim),
        )
        return result

    def _split_domain(self, border_width):
        """Split domain and create borders."""
        client = self.client
        domains = tuple(
            client.scatter(d, workers=w)
            for d, w in zip(
                self.domain.split(self.splitter), cycle(self._get_cluster_worker())
            )
        )

        borders = tuple(
            client.submit(self._make_borders, sub, border_width, self.splitter.dim[0])
            for sub in domains
        )
        return domains, borders

    @staticmethod
    def _pint(solver, domain, borders, left_neighbor_borders, right_neighbor_borders):
        result = (
            solver.integrate_border(
                border=borders[0],
                domain=domain,
                neighbor_border=left_neighbor_borders[1],
                direction=False,
            ),
            solver.integrate_border(
                border=borders[1],
                domain=domain,
                neighbor_border=right_neighbor_borders[0],
                direction=True,
            ),
        )
        return result

    @staticmethod
    def _int(solver, domain):
        new_domain = solver.integrate(domain)
        return new_domain

    @staticmethod
    def _stitch(tailor, domain, borders):
        return tailor.stitch(domain, borders)
