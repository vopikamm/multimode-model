"""Types, Type Aliases and unbound TypeVars defined for this project."""

import numpy as np
from typing import TypeVar

ArrayType = TypeVar("ArrayType")

Array = np.ndarray
Shape = tuple[int, ...]
DType = float
