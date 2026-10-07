"""Shared array types and input/output helpers for the pricers.

Pricers accept anything array-like, compute on float64 arrays using NumPy
broadcasting, and return a Python float when the result is a scalar.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatOrArray: TypeAlias = float | FloatArray


def as_float_array(x: ArrayLike) -> FloatArray:
    """Convert an input to a float64 array (no copy if it already is one)."""
    return np.asarray(x, dtype=np.float64)


def as_bool_array(x: ArrayLike, name: str = "is_call") -> BoolArray:
    """Convert a bool or bool array; reject other dtypes.

    Strict on purpose: ``is_call=1`` or ``is_call="C"`` would otherwise be
    silently truthy, and a put priced as a call is a costly silent error.
    """
    arr = np.asarray(x)
    if arr.dtype != np.bool_:
        raise TypeError(f"{name} must be a bool or bool array, got dtype {arr.dtype}")
    return arr


def to_output(x: FloatArray) -> FloatOrArray:
    """Return a Python float for a 0-d result, otherwise the array unchanged."""
    return float(x) if x.ndim == 0 else x
