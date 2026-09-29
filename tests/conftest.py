"""Shared test fixtures: parameter grids for the analytic pricers."""

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

# Price is homogeneous of degree 1 in (S, K), so only moneyness K/S matters:
# S is fixed and K varies. Values are chosen to span the regular region of the
# pricer; the limits (T = 0, sigma = 0, deep ITM/OTM) are tested explicitly in
# their own tests, not through this grid.
_S = [100.0]
_K = [80.0, 90.0, 100.0, 110.0, 125.0]  # moneyness 0.8 to 1.25
_T = [0.05, 0.25, 1.0, 3.0]  # ~2.5 weeks to 3 years
_R = [-0.01, 0.0, 0.05]  # includes a negative rate (EUR/JPY/CHF regimes)
_Q = [0.0, 0.03]
_SIGMA = [0.05, 0.2, 0.5, 1.0]  # low-vol FX to high-vol crypto/single-name


@pytest.fixture(scope="session")
def bs_grid() -> dict[str, NDArray[np.float64]]:
    """Cartesian product of the values above, as flat 1-D arrays (480 points).

    Keys match the pricer argument names, so tests can call
    ``bs_price(**bs_grid, is_call=True)``. Arrays are read-only because the
    fixture is shared across the whole session.
    """
    names = ("S", "K", "T", "r", "q", "sigma")
    rows = list(itertools.product(_S, _K, _T, _R, _Q, _SIGMA))
    columns = np.array(rows, dtype=np.float64).T
    grid = dict(zip(names, columns, strict=True))
    for arr in grid.values():
        arr.setflags(write=False)
    return grid
