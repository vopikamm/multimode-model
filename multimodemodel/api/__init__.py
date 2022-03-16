"""API declarations.

All APIs, i.e. interfaces for classes, are declared here.
"""

# import public base classes
from .core import (
    GridShift,
    GridBase,
    StaggeredGridBase,
    ParameterBase,
    StateBase,
    StateDequeBase,
    VariableBase,
    DomainBase,
)

from .split import (
    SplitVisitorBase,
    MergeVisitorBase,
    RegularSplitMergerBase,
)

from .integration import (
    TimeSteppingFunctionBase,
    SolverBase,
)

from .border import (
    BorderDirection,
    BorderSplitterBase,
    BorderBase,
    BorderMergerBase,
    TailorBase,
)

from .workflow import WorkflowBase

# import public type aliases and bound TypeVars
from .core import (
    GridType,
    StaggeredGridType,
    ParameterType,
    VariableType,
    StateType,
    StateDequeType,
    DomainType,
)

from .border import BorderType, BorderSplitterType, BorderMergerType

from .typing import ArrayType, Array, Shape, DType

from .integration import RHSFunction
