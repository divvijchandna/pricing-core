"""Tests for bs_price and black76_price (PLAN.md Week 1, Task 3).

Written before the implementation. The same suite is run against the legacy
Streamlit formula in the Task 2 audit.
"""

import numpy as np
import pytest
from scipy.stats import norm

from pricing_core.analytic.black_scholes import black76_price, bs_price

# --- helpers -----------------------------------------------------------------


def _pv_forward_intrinsic(S, K, T, r, q, *, is_call):
    """Discounted forward intrinsic value.

    Call: max(S e^(-qT) - K e^(-rT), 0).  Put: max(K e^(-rT) - S e^(-qT), 0).
    """
    stock_leg = S * np.exp(-q * T)
    strike_leg = K * np.exp(-r * T)
    payoff = stock_leg - strike_leg if is_call else strike_leg - stock_leg
    return np.maximum(payoff, 0.0)


def _black76_reference(F, K, T, r, sigma, *, is_call):
    """Black (1976) written out directly, independent of bs_price.

    C = e^(-rT) [F N(d1) - K N(d2)],  P = e^(-rT) [K N(-d2) - F N(-d1)],
    d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T).
    """
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# --- model identities over the parameter grid --------------------------------


def test_put_call_parity(bs_grid):
    g = bs_grid
    call = bs_price(**g, is_call=True)
    put = bs_price(**g, is_call=False)
    rhs = g["S"] * np.exp(-g["q"] * g["T"]) - g["K"] * np.exp(-g["r"] * g["T"])
    # atol needed: rhs is ~0 near the forward, where a pure rtol is meaningless.
    np.testing.assert_allclose(call - put, rhs, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_black76_matches_independent_formula(bs_grid, is_call):
    g = bs_grid
    F, K, T, r, sigma = g["S"], g["K"], g["T"], g["r"], g["sigma"]
    expected = _black76_reference(F, K, T, r, sigma, is_call=is_call)
    actual = black76_price(F, K, T, r, sigma, is_call=is_call)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_no_arbitrage_bounds(bs_grid):
    g = bs_grid
    S, K, T, r, q = g["S"], g["K"], g["T"], g["r"], g["q"]
    call = bs_price(**g, is_call=True)
    put = bs_price(**g, is_call=False)
    slack = 1e-10  # rounding only; prices sit exactly on the bound deep ITM
    call_lo = _pv_forward_intrinsic(S, K, T, r, q, is_call=True)
    put_lo = _pv_forward_intrinsic(S, K, T, r, q, is_call=False)
    assert np.all(call >= call_lo - slack)
    assert np.all(call <= S * np.exp(-q * T) + slack)
    assert np.all(put >= put_lo - slack)
    assert np.all(put <= K * np.exp(-r * T) + slack)


# --- known answers (Hull, Options, Futures, and Other Derivatives) -----------
# Hull prints these to 2 d.p., so the tolerance is the rounding: +/- 0.005.


def test_hull_bsm_call():
    # "Black-Scholes-Merton model" chapter: S=42, K=40, T=0.5, r=10%, sigma=20%.
    assert bs_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, is_call=True) == pytest.approx(
        4.76, abs=0.005
    )


def test_hull_bsm_put():
    assert bs_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, is_call=False) == pytest.approx(
        0.81, abs=0.005
    )


def test_hull_black76_futures_put():
    # "Futures options and Black's model" chapter: crude-oil futures put,
    # F=20, K=20, T=4/12, r=9%, sigma=25%.
    assert black76_price(
        20.0, 20.0, 4 / 12, 0.09, 0.25, is_call=False
    ) == pytest.approx(1.12, abs=0.005)


# --- limits ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("K", "is_call", "expect_forward"),
    [
        (1.0, True, True),  # deep ITM call -> S e^(-qT) - K e^(-rT)
        (1e4, True, False),  # deep OTM call -> 0
        (1e4, False, True),  # deep ITM put  -> K e^(-rT) - S e^(-qT)
        (1.0, False, False),  # deep OTM put  -> 0
    ],
)
def test_deep_itm_and_otm(K, is_call, expect_forward):
    S, T, r, q, sigma = 100.0, 1.0, 0.05, 0.02, 0.2
    price = bs_price(S, K, T, r, q, sigma, is_call=is_call)
    if expect_forward:
        expected = _pv_forward_intrinsic(S, K, T, r, q, is_call=is_call)
        np.testing.assert_allclose(price, expected, rtol=1e-12)
    else:
        np.testing.assert_allclose(price, 0.0, atol=1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_zero_vol_is_discounted_forward_intrinsic(bs_grid, is_call):
    # Not max(S - K, 0): with sigma = 0, S_T = F with certainty.
    g = dict(bs_grid, sigma=np.zeros_like(bs_grid["sigma"]))
    expected = _pv_forward_intrinsic(
        g["S"], g["K"], g["T"], g["r"], g["q"], is_call=is_call
    )
    np.testing.assert_allclose(
        bs_price(**g, is_call=is_call), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("is_call", [True, False])
def test_zero_expiry_is_intrinsic(bs_grid, is_call):
    g = dict(bs_grid, T=np.zeros_like(bs_grid["T"]))
    S, K = g["S"], g["K"]
    expected = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    np.testing.assert_allclose(
        bs_price(**g, is_call=is_call), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("T", [0.25, 1.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_price_strictly_increasing_in_vol(T, is_call):
    # Vega = S e^(-qT) phi(d1) sqrt(T) > 0. The sigma range starts at 0.05, away
    # from sigma = 0, where the price flattens to its limit in floating point.
    K = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    sigma = np.linspace(0.05, 1.0, 20)[:, None]
    prices = bs_price(100.0, K, T, 0.05, 0.02, sigma, is_call=is_call)
    assert np.all(np.diff(prices, axis=0) > 0)


# --- API: scalars, broadcasting, is_call ------------------------------------


def test_scalar_inputs_return_float():
    call = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, is_call=True)
    fut = black76_price(100.0, 100.0, 1.0, 0.05, 0.2, is_call=True)
    for result in (call, fut):
        assert isinstance(result, float)
        assert np.ndim(result) == 0


def test_broadcasts_row_by_column():
    K = np.array([90.0, 100.0, 110.0])  # shape (3,)
    sigma = np.array([[0.1], [0.2]])  # shape (2, 1)
    prices = bs_price(100.0, K, 1.0, 0.05, 0.01, sigma, is_call=True)
    assert prices.shape == (2, 3)
    for i, s in enumerate(sigma[:, 0]):
        for j, k in enumerate(K):
            scalar = bs_price(100.0, k, 1.0, 0.05, 0.01, s, is_call=True)
            np.testing.assert_allclose(prices[i, j], scalar, rtol=1e-14)


def test_is_call_accepts_bool_array():
    K = np.array([90.0, 100.0, 110.0])
    flags = np.array([True, False, True])
    mixed = bs_price(100.0, K, 1.0, 0.05, 0.01, 0.2, is_call=flags)
    calls = bs_price(100.0, K, 1.0, 0.05, 0.01, 0.2, is_call=True)
    puts = bs_price(100.0, K, 1.0, 0.05, 0.01, 0.2, is_call=False)
    np.testing.assert_allclose(mixed, np.where(flags, calls, puts), rtol=1e-14)


def test_is_call_is_required_and_keyword_only():
    with pytest.raises(TypeError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2)  # missing is_call
    with pytest.raises(TypeError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True)  # positional is_call
