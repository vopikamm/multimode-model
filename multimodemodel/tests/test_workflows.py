# flake8: noqa
"""Test workflow managers."""
from itertools import count
from xmlrpc.client import NOT_WELLFORMED_ERROR
import numpy as np
import pytest
from multimodemodel.api import WorkflowBase
from multimodemodel.border import RegularSplitMerger
from multimodemodel.datastructure import Domain, Parameter, Variable, State
from multimodemodel.grid import Grid
from multimodemodel.util import str_to_date
from multimodemodel.integrate import Solver, euler_forward
from multimodemodel.workflow import DaskWorkflow, Workflow


class MockDaskClient:
    def __init__(self, n_workers):
        self.n_workers = n_workers

    def submit(self, func, *args, **kwargs):
        _ = kwargs.pop("workers", None)
        return func(*args, **kwargs)

    def scatter(self, iterable, **kwargs):
        return iterable

    def scheduler_info(self):
        return dict(workers={i: i for i in range(self.n_workers)})


@pytest.fixture()
def single_grid():
    shape = (50, 50)
    return Grid.cartesian(x=np.arange(shape[1]), y=np.arange(shape[0]))


@pytest.fixture()
def some_time():
    return str_to_date("2000-01-01")


@pytest.fixture()
def single_variable(single_grid, some_time):
    return Variable(None, grid=single_grid, time=some_time)


@pytest.fixture()
def single_variable_state(single_variable):
    return State(v=single_variable)


@pytest.fixture()
def parameter():
    return Parameter()


@pytest.fixture
def single_variable_domain(single_variable_state, parameter):
    return Domain(state=single_variable_state, parameter=parameter)


@pytest.fixture
def counting_solver():
    def rhs(state, param):
        """Simply increment by one."""
        inc_state = State(
            v=Variable(
                np.ones(state.v.grid.shape), grid=state.v.grid, time=state.v.time
            )
        )
        return inc_state

    return Solver(rhs=rhs, ts_schema=euler_forward, step=1.0)


@pytest.fixture(params=(0, 1))
def dim(request):
    return request.param


@pytest.fixture
def regular_splitter(dim):
    return RegularSplitMerger(parts=3, dim=(dim,))


class CountingWorkflow(WorkflowBase):
    def __init__(self, domain, solver):
        self.domain = domain
        self.solver = solver
        self.counter = 0

    def _run_next(self):
        self.counter += 1


def test_WorkflowBase_cannot_instantiate():
    """Test WorkflowBase cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class *"):
        _ = WorkflowBase()  # type: ignore


def test_WorkflowBase_run_until_runs_until_datetime64(
    single_variable_domain, counting_solver
):
    """Test WorkflowBase runs until specified datetime64."""

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    assert wf.counter == 0
    end_date = single_variable_domain.state.v.time + np.timedelta64(5, "s")
    wf.run_until(end_date)
    assert wf.counter == 5

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    assert wf.counter == 0
    end_date = single_variable_domain.state.v.time + np.timedelta64(5010, "ms")
    wf.run_until(end_date)
    assert wf.counter == 6

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    assert wf.counter == 0
    end_date = single_variable_domain.state.v.time
    wf.run_until(end_date)
    assert wf.counter == 0


def test_WorkflowBase_run_until_runs_until_str(single_variable_domain, counting_solver):
    """Test WorkflowBase runs until specified date str."""

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    assert wf.counter == 0
    end_date = "2000-01-01 00:00:05"
    wf.run_until(end_date)
    assert wf.counter == 5

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    assert wf.counter == 0
    end_date = "2000-01-01 00:00:05.5"
    wf.run_until(end_date)
    assert wf.counter == 6


def test_WorkflowBase_run_until_raises_on_end_date_in_past(
    single_variable_domain, counting_solver
):
    """Test WorkflowBase runs until specified datetime64."""

    wf = CountingWorkflow(single_variable_domain, counting_solver)
    end_date = single_variable_domain.state.v.time - np.timedelta64(5, "s")
    with pytest.raises(ValueError, match="End date is before present date.*"):
        wf.run_until(end_date)


def test_Workflow_run_runs_for_nsteps(single_variable_domain, counting_solver):

    workflow = Workflow(domain=single_variable_domain, solver=counting_solver)

    workflow.run(10)
    assert (workflow.domain.state.v.data == 10).all()

    workflow.run(5)
    assert (workflow.domain.state.v.data == 15).all()


def test_Workflow_diagnoses_each_step(single_variable_domain, counting_solver):

    workflow = Workflow(domain=single_variable_domain, solver=counting_solver)

    buffer = np.zeros(single_variable_domain.state.v.grid.shape)

    def diag(domain) -> None:
        buffer[:] += domain.state.variables["v"].safe_data

    n_steps = 10
    workflow.run(steps=n_steps, diag=diag)
    assert (buffer == sum(range(1, n_steps + 1))).all()


def test_DaskWorkflow_initializes(
    single_variable_domain, counting_solver, regular_splitter
):
    client = MockDaskClient(n_workers=3)
    workflow = DaskWorkflow(
        domain=single_variable_domain,
        solver=counting_solver,
        splitter=regular_splitter,
        client=client,
    )
    assert workflow._domain_type is single_variable_domain.__class__
    assert len(workflow.domain_stack) == len(workflow.border_stack) == 1
    assert (
        len(workflow.domain_stack[-1])
        == len(workflow.border_stack[-1])
        == regular_splitter.parts
    )


def test_DaskWorkflow_run_produces_correct_results(
    single_variable_domain, counting_solver, regular_splitter
):
    client = MockDaskClient(n_workers=3)
    workflow = DaskWorkflow(
        domain=single_variable_domain,
        solver=counting_solver,
        splitter=regular_splitter,
        client=client,
    )
    expected = single_variable_domain
    for _ in range(10):
        expected = counting_solver.integrate(expected)

    workflow.run(10)

    assert workflow.domain.state == expected.state


def test_DaskWorkflow_diag_all_subdomains(
    single_variable_domain, counting_solver, regular_splitter
):
    client = MockDaskClient(n_workers=3)
    workflow = DaskWorkflow(
        domain=single_variable_domain,
        solver=counting_solver,
        splitter=regular_splitter,
        client=client,
    )

    buffer = tuple(np.zeros(s.state.v.grid.shape) for s in workflow.domain_stack[-1])

    def diag(domain):
        buffer[domain.id][:] += domain.state.v.safe_data

    n_steps = 10
    workflow.run(steps=n_steps, diag=diag)
    assert all((b == sum(range(1, n_steps + 1))).all() for b in buffer)
