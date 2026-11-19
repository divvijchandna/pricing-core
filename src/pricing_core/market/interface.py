"""Market states for revaluation: spot, flat rates and an ATM vol term structure.

One MarketState holds n_states states at once (today, or thousands of scenarios),
so a book revalues in one broadcast call instead of a loop over states:

    spot, r, q   shape (n_states,)          r, q continuously compounded, flat
    tenors       shape (m,), shared         years, positive, strictly increasing
    atm_vol      shape (n_states, m)        annualised, decimal (0.10, not 10)

Vol at maturity T interpolates total variance w = sigma^2 T linearly in T between
the nodes w_j = sigma_j^2 T_j, with flat vol outside [T_1, T_m]:

    w(T)     = w_j + (T - T_j) / (T_{j+1} - T_j) * (w_{j+1} - w_j)
    sigma(T) = sqrt(w(T) / T)

Linear w means a constant forward variance (w_{j+1} - w_j) / (T_{j+1} - T_j)
between nodes, so w is non-decreasing in T whenever it is at the nodes: the
interpolation adds no calendar arbitrage. Linear interpolation in vol does not
have this property. No smile: every strike gets the ATM vol. Arbitrage checks on
the node vols themselves belong to market.surface.

A negative or NaN node vol gives NaN for every maturity whose interpolation
bracket [T_j, T_{j+1}] contains that node (also exactly at the neighbouring node,
where its weight is 0, since 0 * NaN = NaN). Never a silent |sigma| from squaring
and never an exception, so one bad scenario cannot stop a vectorised run. Shape
errors, which are programming errors, raise ValueError at construction.

Reference: Gatheral, The Volatility Surface (2006), ch. 1 (total implied
variance) and ch. 3 (no calendar arbitrage: w non-decreasing in T).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pricing_core._types import FloatArray, as_float_array


def _frozen_copy(x: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    """Read-only float64 copy broadcast to `shape`; the caller's array is untouched."""
    arr = np.array(np.broadcast_to(as_float_array(x), shape), dtype=np.float64)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False, init=False)
class MarketState:
    """n_states market states sharing one tenor grid. Arrays are read-only copies."""

    spot: FloatArray
    r: FloatArray
    q: FloatArray
    tenors: FloatArray
    atm_vol: FloatArray

    def __init__(
        self,
        spot: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        tenors: ArrayLike,
        atm_vol: ArrayLike,
    ) -> None:
        tenors_arr = as_float_array(tenors)
        if tenors_arr.ndim != 1 or tenors_arr.size == 0:
            raise ValueError("tenors must be a non-empty 1-D array")
        # NaN fails both comparisons, so NaN tenors are rejected too.
        if not (tenors_arr[0] > 0 and np.all(np.diff(tenors_arr) > 0)):
            raise ValueError("tenors must be positive and strictly increasing")
        m = tenors_arr.size

        per_state = {"spot": spot, "r": r, "q": q}
        shapes = []
        for name, x in per_state.items():
            ndim = np.ndim(x)
            if ndim > 1:
                raise ValueError(f"{name} must be a scalar or 1-D, got {ndim}-D")
            shapes.append(np.shape(x))
        vol_shape = np.shape(atm_vol)
        if len(vol_shape) not in (1, 2) or vol_shape[-1] != m:
            raise ValueError(
                f"atm_vol must have shape (m,) or (n_states, m) with m = {m}, "
                f"got {vol_shape}"
            )
        shapes.append(vol_shape[:-1])
        state_shape = np.broadcast_shapes(*shapes)  # ValueError if counts disagree
        n = state_shape[0] if state_shape else 1

        for name, x in per_state.items():
            object.__setattr__(self, name, _frozen_copy(x, (n,)))
        object.__setattr__(self, "tenors", _frozen_copy(tenors_arr, (m,)))
        object.__setattr__(self, "atm_vol", _frozen_copy(atm_vol, (n, m)))

    @property
    def n_states(self) -> int:
        return self.spot.shape[0]

    def vol(self, T: ArrayLike) -> FloatArray:
        """ATM vol at maturities T for every state, shape (n_states, *T.shape).

        Total-variance interpolation with flat vol extrapolation (module docstring).
        Always an array, also for scalar T, so per-state outputs keep one layout.
        """
        T_arr = as_float_array(T)
        tenors = self.tenors
        m = tenors.size
        # Clipping to [T_1, T_m] gives flat extrapolation, and t >= T_1 > 0 means
        # sqrt(w / t) never divides by zero (T = 0 is just the first node's vol).
        t = np.clip(T_arr, tenors[0], tenors[-1])
        lo = np.clip(np.searchsorted(tenors, t, side="right") - 1, 0, max(m - 2, 0))
        hi = np.minimum(lo + 1, m - 1)  # == lo only when there is a single tenor
        span = tenors[hi] - tenors[lo]
        lam = np.where(span > 0, (t - tenors[lo]) / np.where(span > 0, span, 1.0), 0.0)

        v = self.atm_vol
        w_nodes = np.where(v >= 0, v * v * tenors, np.nan)  # (n_states, m)
        w = (1.0 - lam) * w_nodes[:, lo] + lam * w_nodes[:, hi]
        return np.sqrt(w / t)
