"""Tests for the shared input/output helpers in pricing_core._types."""

import numpy as np
import pytest

from pricing_core._types import as_bool_array, as_float_array, to_output


@pytest.mark.parametrize("bad", [1, 0, 1.0, "C", np.array([0, 1])])
def test_as_bool_array_rejects_non_bool(bad):
    # is_call=1 or "C" would otherwise be silently truthy: a put priced as a call.
    with pytest.raises(TypeError, match="is_call"):
        as_bool_array(bad)


@pytest.mark.parametrize("good", [True, False, np.True_, np.array([True, False])])
def test_as_bool_array_accepts_bool_and_bool_array(good):
    result = as_bool_array(good)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, good)


def test_as_float_array_no_copy_for_float64():
    grid = np.array([90.0, 100.0, 110.0])
    grid.setflags(write=False)  # like the shared test grid
    assert as_float_array(grid) is grid


def test_as_float_array_converts_scalars_and_ints():
    scalar = as_float_array(100)
    assert scalar.dtype == np.float64
    assert scalar.ndim == 0
    assert as_float_array([90, 100]).dtype == np.float64


def test_to_output_scalar_is_builtin_float():
    result = to_output(np.asarray(4.76))
    assert type(result) is float
    assert result == 4.76


@pytest.mark.parametrize("arr", [np.array([1.0]), np.ones(3), np.ones((2, 3))])
def test_to_output_array_unchanged(arr):
    # A size-1 array keeps its shape; only a 0-d result becomes a float.
    assert to_output(arr) is arr
