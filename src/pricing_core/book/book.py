"""A book of vanilla positions, revalued across many market states in one call.

With k positions and n states, position terms become arrays of shape (k,) and
market inputs shape (n, 1), so every pricer call broadcasts to an (n, k) grid:

    V[i, j]     = bs_price(S_i, K_j, T_j - h, r_i, q_i, sigma_i(T_j - h), is_call_j)
    value[i]    = sum_j quantity_j * V[i, j]
    greek[i]    = sum_j quantity_j * greek(S_i, K_j, T_j, ...)

One vectorised call per pricer or Greek, whatever n and k: this is what lets the
risk engine revalue a book across thousands of VaR scenarios without a loop.

Horizon h (years): every option is aged to T - h and priced at vol(T - h), i.e.
the vol term structure is held fixed in maturity and each option rolls down it.
An option with T < h has expired by the horizon and is worth intrinsic at the
state's spot (T clipped to 0: cash settlement at the scenario spot).

Units: value in the underlying's currency per unit of quantity; Greeks in the
units of analytic.greeks (vega per 1.00 of vol, theta per year), times quantity.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from pricing_core._types import BoolArray, FloatArray
from pricing_core.analytic import greeks as g
from pricing_core.analytic.black_scholes import bs_price
from pricing_core.book.position import Position
from pricing_core.market.interface import MarketState


class BookGreeks(NamedTuple):
    """Book Greeks per state, each of shape (n_states,)."""

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray
    vanna: FloatArray
    volga: FloatArray


# Gamma, vega, vanna and volga are the same for calls and puts (no is_call).
_GREEKS: dict[str, tuple[Callable[..., object], bool]] = {
    "delta": (g.delta, True),
    "gamma": (g.gamma, False),
    "vega": (g.vega, False),
    "theta": (g.theta, True),
    "rho": (g.rho, True),
    "vanna": (g.vanna, False),
    "volga": (g.volga, False),
}


def _frozen(x: list[float] | list[bool], dtype: type) -> FloatArray | BoolArray:
    arr = np.array(x, dtype=dtype)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False, init=False)
class Book:
    """Vanilla positions on one underlying; held as a tuple, terms as arrays."""

    positions: tuple[Position, ...]
    _quantity: FloatArray = field(repr=False)
    _K: FloatArray = field(repr=False)
    _T: FloatArray = field(repr=False)
    _is_call: BoolArray = field(repr=False)

    def __init__(self, positions: Iterable[Position]) -> None:
        held = tuple(positions)  # a copy: later changes to the caller's list don't leak
        object.__setattr__(self, "positions", held)
        # Contract terms as read-only arrays, built once rather than per revaluation.
        for attr, dtype in (("quantity", float), ("K", float), ("T", float)):
            terms = _frozen([getattr(p, attr) for p in held], dtype)
            object.__setattr__(self, f"_{attr}", terms)
        object.__setattr__(self, "_is_call", _frozen([p.is_call for p in held], bool))

    def _inputs(self, states: MarketState, T: FloatArray) -> tuple[FloatArray, ...]:
        """(S, K, T, r, q, sigma) laid out to broadcast to (n_states, k)."""
        return (
            states.spot[:, None],
            self._K,
            T,
            states.r[:, None],
            states.q[:, None],
            states.vol(T),  # (n_states, k): each option at its own maturity
        )

    def revalue(self, states: MarketState, horizon: float = 0.0) -> FloatArray:
        """Book value in every state after `horizon` years, shape (n_states,)."""
        h = float(horizon)
        if not (np.isfinite(h) and h >= 0):
            raise ValueError(f"horizon must be finite and >= 0, got {horizon}")
        T = np.maximum(self._T - h, 0.0)
        values = bs_price(*self._inputs(states, T), is_call=self._is_call)
        return np.asarray(values) @ self._quantity

    def greeks(self, states: MarketState) -> BookGreeks:
        """Book Greeks in every state (no ageing), each of shape (n_states,)."""
        args = self._inputs(states, self._T)
        out = {}
        for name, (func, takes_is_call) in _GREEKS.items():
            kw = {"is_call": self._is_call} if takes_is_call else {}
            out[name] = np.asarray(func(*args, **kw)) @ self._quantity
        return BookGreeks(**out)
