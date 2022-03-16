"""Test API contracts."""
# flake8: noqa
from numpy import split
import pytest

from multimodemodel.api import (
    GridBase,
    StaggeredGridBase,
    ParameterBase,
    StateBase,
    StateDequeBase,
    VariableBase,
    DomainBase,
    SolverBase,
    RHSFunction,
    SplitVisitorBase,
    MergeVisitorBase,
    BorderBase,
    TailorBase,
    BorderDirection,
)

from multimodemodel.api.split import Splitable

splitable_types = (
    ParameterBase,
    GridBase,
    StaggeredGridBase,
    VariableBase,
    StateBase,
    StateDequeBase,
    DomainBase,
    BorderBase,
)


@pytest.mark.parametrize("cls", splitable_types)
def test_splitable_interface_inherited_from(cls):
    """Test implementation of splitable interface."""
    assert issubclass(cls, Splitable)


# def test_MergeVisitorBase_axis_validation():
#     dims = (0, 1, 2, 3, -1, -2, -3, -4)
#     valid = (False, True, True, True, True, True, False, False)
#     for d, v in zip(dims, valid):
#         if not v:
#             with pytest.raises(
#                 ValueError,
#                 match="You cannot split along the first dimension.",
#             ):
#                 MergeVisitorBase._validate_axis(d)
#         else:
#             MergeVisitorBase._validate_axis(d)


# def test_SplitVisitorBase_axis_validation():
#     dims = (0, 1, 2, 3, -1, -2, -3, -4)
#     valid = (False, True, True, True, True, True, False, False)
#     for d, v in zip(dims, valid):
#         if not v:
#             with pytest.raises(
#                 ValueError,
#                 match="You cannot split along the first dimension.",
#             ):
#                 SplitVisitorBase._validate_axis(d)
#         else:
#             SplitVisitorBase._validate_axis(d)
