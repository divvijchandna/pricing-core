"""Tests for the vectorised implied-vol solver (PLAN.md Week 1, Task 5).

Contract: implied_vol(price, S, K, T, r, q, *, is_call, max_iter=100) returns the
sigma with bs_price(S, K, T, r, q, sigma) = price, or NaN when no positive sigma
exists (price on or outside the no-arbitrage bounds), when an input is invalid,
or when the solver has not converged within max_iter. Never raises inside a
vectorised call.
"""

import numpy as np
import pytest

from pricing_core.analytic.black_scholes import black76_price, bs_price
from pricing_core.analytic.greeks import vega
from pricing_core.analytic.implied_vol import implied_vol

# Round-trip accuracy is limited by conditioning: d_sigma ~ d_price / vega, with
# two sources of price error:
#   - the solver, which works on the OTM price (the time value) and stops at
#     ~1e-12 relative to it: 1e-12 * otm_price / vega;
#   - rounding already in the input: an ITM price is only accurate to
#     ~eps * price, which no solver can undo: ~eps * price / vega.
# Tolerance: 100x each, plus a floor. A solver that skipped the OTM conversion
# and stopped at 1e-12 relative to the ITM price would fail deep ITM.
_EPS = np.finfo(np.float64).eps


def _otm_price(g):
    # OTM side: call if P_S <= P_K (F <= K), else put.
    otm_is_call = g["S"] * np.exp(-g["q"] * g["T"]) <= g["K"] * np.exp(-g["r"] * g["T"])
    return np.asarray(bs_price(**g, is_call=otm_is_call))


def _round_trip_tol(price, otm_price, vega_):
    return (1e-10 * otm_price + 100 * _EPS * np.abs(price)) / vega_ + 1e-12


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_on_grid(bs_grid, is_call):
    # Calls and puts on the whole grid: ITM inputs included, since the solver
    # converts them to the OTM side by parity. Points with negligible vega are
    # excluded: there the price barely depends on sigma.
    g = bs_grid
    price = np.asarray(bs_price(**g, is_call=is_call))
    v = np.asarray(vega(**g))
    ok = v > 1e-8 * g["S"]
    args = {k: g[k] for k in ("S", "K", "T", "r", "q")}
    sigma_hat = np.asarray(implied_vol(price, **args, is_call=is_call))
    err = np.abs(sigma_hat - g["sigma"])
    tol = _round_trip_tol(price, _otm_price(g), v)
    assert np.all(err[ok] <= tol[ok])
    assert ok.mean() > 0.9  # the filter must not hide most of the grid


def test_call_and_put_give_same_vol(bs_grid):
    # Same sigma for both by parity; the solver should agree with itself.
    g = bs_grid
    args = {k: g[k] for k in ("S", "K", "T", "r", "q")}
    iv_c = implied_vol(bs_price(**g, is_call=True), **args, is_call=True)
    iv_p = implied_vol(bs_price(**g, is_call=False), **args, is_call=False)
    v = np.asarray(vega(**g))
    ok = v > 1e-4 * g["S"]
    np.testing.assert_allclose(np.asarray(iv_c)[ok], np.asarray(iv_p)[ok], rtol=1e-8)


def test_hull_example_inverts_to_twenty_percent():
    price = bs_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, is_call=True)
    sigma = implied_vol(price, 42.0, 40.0, 0.5, 0.10, 0.0, is_call=True)
    assert sigma == pytest.approx(0.20, abs=1e-10)


def test_black76_via_q_equals_r():
    F, K, T, r = 20.0, 22.0, 0.5, 0.04
    price = black76_price(F, K, T, r, 0.35, is_call=False)
    assert implied_vol(price, F, K, T, r, r, is_call=False) == pytest.approx(
        0.35, abs=1e-10
    )


# --- no solution: NaN, never an exception -------------------------------------

_ARGS = {"S": 100.0, "K": 90.0, "T": 1.0, "r": 0.05, "q": 0.02}
_P_S = 100.0 * np.exp(-0.02)  # upper bound for calls
_P_K = 90.0 * np.exp(-0.05)
_CALL_LOWER = _P_S - _P_K  # ITM call, so the lower bound is positive


@pytest.mark.parametrize(
    "price",
    [
        _CALL_LOWER - 1e-3,  # below the lower bound: arbitrage
        _CALL_LOWER,  # exactly at the lower bound: sigma = 0, not positive
        _P_S,  # at the upper bound: sigma = infinity
        _P_S + 1.0,  # above the upper bound: arbitrage
        -1.0,
        np.nan,
    ],
    ids=["below_lower", "at_lower", "at_upper", "above_upper", "negative", "nan"],
)
def test_price_without_positive_vol_gives_nan(price):
    assert np.isnan(implied_vol(price, **_ARGS, is_call=True))


def test_put_bounds_give_nan():
    put_upper = _P_K  # K e^(-rT)
    assert np.isnan(implied_vol(put_upper + 0.5, **_ARGS, is_call=False))
    assert np.isnan(implied_vol(0.0, **_ARGS, is_call=False))  # OTM put lower = 0


def test_mixed_array_nan_mask():
    good = bs_price(**_ARGS, sigma=0.25, is_call=True)
    prices = np.array([good, _CALL_LOWER - 1.0, good, _P_S + 1.0, np.nan])
    result = np.asarray(implied_vol(prices, **_ARGS, is_call=True))
    np.testing.assert_array_equal(np.isnan(result), [False, True, False, True, True])
    np.testing.assert_allclose(result[[0, 2]], 0.25, atol=1e-10)


@pytest.mark.parametrize(
    "bad", [{"S": 0.0}, {"K": -1.0}, {"T": -0.5}, {"T": 0.0}, {"r": np.nan}]
)
def test_invalid_inputs_give_nan(bad):
    # T = 0: the price is pure intrinsic value, so no vol is implied.
    assert np.isnan(implied_vol(10.0, **{**_ARGS, **bad}, is_call=True))


# --- convergence ---------------------------------------------------------------


def test_converges_within_fixed_iterations(bs_grid):
    # A small cap must still give the full round trip on the regular grid:
    # Newton from a good start plus safeguarded bisection converges fast.
    g = bs_grid
    price = np.asarray(bs_price(**g, is_call=True))
    v = np.asarray(vega(**g))
    ok = v > 1e-8 * g["S"]
    args = {k: g[k] for k in ("S", "K", "T", "r", "q")}
    sigma_hat = np.asarray(implied_vol(price, **args, is_call=True, max_iter=50))
    err = np.abs(sigma_hat - g["sigma"])
    tol = _round_trip_tol(price, _otm_price(g), v)
    assert np.all(err[ok] <= tol[ok])


def test_stress_wide_random_sample():
    # Fixed-seed sample far wider than the grid: moneyness e^(+/-1), T from
    # ~4 days to ~3 years, sigma from 2% to 200%. Low-vol near-the-money cases
    # here need the bracket safeguard: from sigma* Newton creeps down a flat
    # tail and a rounding-sized step can land at sigma <= 0.
    rng = np.random.default_rng(20261009)
    n = 50_000
    S = np.full(n, 100.0)
    K = 100.0 * np.exp(rng.uniform(-1.0, 1.0, n))
    T = 10.0 ** rng.uniform(-2.0, 0.5, n)
    r = rng.uniform(-0.01, 0.08, n)
    q = rng.uniform(0.0, 0.05, n)
    sigma = 10.0 ** rng.uniform(-1.7, 0.3, n)
    is_call = rng.random(n) < 0.5
    g = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma}
    price = np.asarray(bs_price(**g, is_call=is_call))
    otm = _otm_price(g)
    v = np.asarray(vega(**g))
    args = {k: g[k] for k in ("S", "K", "T", "r", "q")}
    sigma_hat = np.asarray(implied_vol(price, **args, is_call=is_call))

    # 1. Every point with usable vega is solved to the round-trip tolerance.
    ok = v > 1e-8 * S
    tol = _round_trip_tol(price, otm, np.where(ok, v, 1.0))  # vega can underflow
    assert np.all(np.abs(sigma_hat - sigma)[ok] <= tol[ok])

    # 2. Never a wrong answer: any non-NaN sigma reprices to the input price.
    found = np.isfinite(sigma_hat)
    repriced = np.asarray(
        bs_price(**{**g, "sigma": np.where(found, sigma_hat, 0.2)}, is_call=is_call)
    )
    # Plus vega * 1e-12: in the far tails (prices ~1e-127) the price is so
    # sensitive to sigma that a sigma correct to ~1e-12 reprices ~1e-10 off.
    price_tol = 1e-10 * otm + 100 * _EPS * np.abs(price) + 1e-12 * v + 1e-300
    assert np.all(np.abs(repriced - price)[found] <= price_tol[found])


def test_unconverged_gives_nan_not_a_guess():
    price = bs_price(**_ARGS, sigma=0.8, is_call=False)
    assert np.isnan(implied_vol(price, **_ARGS, is_call=False, max_iter=0))


# --- API -----------------------------------------------------------------------


def test_scalar_in_float_out():
    price = bs_price(**_ARGS, sigma=0.3, is_call=True)
    assert type(implied_vol(price, **_ARGS, is_call=True)) is float


def test_is_call_required_and_keyword_only():
    with pytest.raises(TypeError):
        implied_vol(10.0, 100.0, 90.0, 1.0, 0.05, 0.02)
    with pytest.raises(TypeError):
        implied_vol(10.0, 100.0, 90.0, 1.0, 0.05, 0.02, True)
