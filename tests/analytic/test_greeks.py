"""Tests for the analytic Greeks (PLAN.md Week 1, Task 4).

Units under test: delta per 1.00 of S, gamma per 1.00 of S squared, vega per 1.00
of vol, theta per year (theta = dV/dt = -dV/dT), rho per 1.00 of r, vanna per 1.00
of S and of vol, volga per 1.00 of vol squared.
"""

import numpy as np
import pytest
from scipy.stats import norm

from pricing_core.analytic.black_scholes import black76_price, bs_price
from pricing_core.analytic.greeks import (
    black76_rho,
    delta,
    gamma,
    rho,
    theta,
    vanna,
    vega,
    volga,
)
from pricing_core.conventions import per_day

# --- finite differences ------------------------------------------------------
# Bumps from CLAUDE.md: spot h = 1e-4 * S; vol, T and r h = 1e-4. Central
# differences. Error = truncation ~ (h / (S sigma sqrt(T)))^2 + roundoff ~ eps V / h,
# so the bump is too coarse where sigma sqrt(T) is small. rtol is the worst
# relative error measured on the grid (all at sigma = 0.05), times ~4 for margin.
# atol covers values near zero, where relative error is meaningless: small OTM
# deltas (~1e-4), and the zero crossings of vanna (d2 = 0) and volga
# (d1 d2 = 0), where the bumped value is roundoff noise, 4 eps V / h^2 ~ 1e-5
# for volga with V up to ~100. Real formula bugs (sign, missing e^(-qT), d1 for
# d2) give O(1) errors; the exact identities below check the formulas to ~1e-10.
H = 1e-4


def _price(g, is_call, **override):
    return np.asarray(bs_price(**{**g, **override}, is_call=is_call))


def _fd(g, name, is_call):
    S, s, T, r = g["S"], g["sigma"], g["T"], g["r"]
    hs = H * S
    P = lambda **o: _price(g, is_call, **o)  # noqa: E731
    if name == "delta":
        return (P(S=S + hs) - P(S=S - hs)) / (2 * hs)
    if name == "gamma":
        return (P(S=S + hs) - 2 * P() + P(S=S - hs)) / hs**2
    if name == "vega":
        return (P(sigma=s + H) - P(sigma=s - H)) / (2 * H)
    if name == "theta":
        return -(P(T=T + H) - P(T=T - H)) / (2 * H)
    if name == "rho":
        return (P(r=r + H) - P(r=r - H)) / (2 * H)
    if name == "vanna":
        return (
            P(S=S + hs, sigma=s + H)
            - P(S=S + hs, sigma=s - H)
            - P(S=S - hs, sigma=s + H)
            + P(S=S - hs, sigma=s - H)
        ) / (4 * hs * H)
    if name == "volga":
        return (P(sigma=s + H) - 2 * P() + P(sigma=s - H)) / H**2
    raise ValueError(name)


GREEKS = {
    # name: (function, takes is_call, rtol, atol)
    "delta": (delta, True, 2e-5, 1e-7),
    "gamma": (gamma, False, 5e-5, 1e-9),
    "vega": (vega, False, 4e-4, 1e-7),
    "theta": (theta, True, 1e-5, 1e-7),
    "rho": (rho, True, 1e-4, 1e-7),
    "vanna": (vanna, False, 6e-4, 1e-6),
    "volga": (volga, False, 3e-4, 2e-5),
}


def _greek(name, g, is_call):
    func, takes_flag = GREEKS[name][:2]
    return func(**g, is_call=is_call) if takes_flag else func(**g)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("name", list(GREEKS))
def test_greek_matches_central_difference(bs_grid, name, is_call):
    _, _, rtol, atol = GREEKS[name]
    expected = _fd(bs_grid, name, is_call)
    np.testing.assert_allclose(
        _greek(name, bs_grid, is_call), expected, rtol=rtol, atol=atol
    )


# --- exact identities (no bumps, so tight tolerances) -------------------------


def test_call_minus_put_relations(bs_grid):
    g = bs_grid
    S, K, T, r, q = g["S"], g["K"], g["T"], g["r"], g["q"]
    p_s, p_k = S * np.exp(-q * T), K * np.exp(-r * T)
    tol = {"rtol": 1e-12, "atol": 1e-10}
    np.testing.assert_allclose(
        delta(**g, is_call=True) - delta(**g, is_call=False), np.exp(-q * T), **tol
    )
    np.testing.assert_allclose(
        theta(**g, is_call=True) - theta(**g, is_call=False), q * p_s - r * p_k, **tol
    )
    np.testing.assert_allclose(
        rho(**g, is_call=True) - rho(**g, is_call=False), K * T * np.exp(-r * T), **tol
    )


@pytest.mark.parametrize("is_call", [True, False])
def test_black_scholes_pde(bs_grid, is_call):
    # Theta + (r - q) S Delta + 0.5 sigma^2 S^2 Gamma - r V = 0
    g = bs_grid
    S, r, q, s = g["S"], g["r"], g["q"], g["sigma"]
    residual = (
        theta(**g, is_call=is_call)
        + (r - q) * S * delta(**g, is_call=is_call)
        + 0.5 * s**2 * S**2 * gamma(**g)
        - r * _price(g, is_call)
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_vega_equals_s_squared_sigma_t_gamma(bs_grid):
    g = bs_grid
    expected = g["S"] ** 2 * g["sigma"] * g["T"] * gamma(**g)
    np.testing.assert_allclose(vega(**g), expected, rtol=1e-12, atol=1e-12)


def test_gamma_and_vega_non_negative(bs_grid):
    assert np.all(gamma(**bs_grid) >= 0)
    assert np.all(vega(**bs_grid) >= 0)


# --- units -------------------------------------------------------------------


def test_theta_per_day_is_one_day_decay():
    # per_day(theta) ~ V(T - 1 day) - V(T), to first order in 1/365.
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.2}
    one_day_later = {**args, "T": 1.0 - 1 / 365}
    decay = bs_price(**one_day_later, is_call=True) - bs_price(**args, is_call=True)
    assert per_day(theta(**args, is_call=True)) == pytest.approx(decay, rel=1e-2)


# --- Black-76 rho (F held fixed) ----------------------------------------------


@pytest.mark.parametrize("is_call", [True, False])
def test_black76_rho_is_minus_t_times_price(bs_grid, is_call):
    F, K, T, r, s = (bs_grid[k] for k in ("S", "K", "T", "r", "sigma"))
    price = black76_price(F, K, T, r, s, is_call=is_call)
    np.testing.assert_allclose(
        black76_rho(F, K, T, r, s, is_call=is_call), -T * price, rtol=1e-12
    )
    bumped = (
        black76_price(F, K, T, r + H, s, is_call=is_call)
        - black76_price(F, K, T, r - H, s, is_call=is_call)
    ) / (2 * H)
    np.testing.assert_allclose(
        black76_rho(F, K, T, r, s, is_call=is_call), bumped, rtol=1e-6, atol=1e-12
    )


# --- zero total volatility (sigma sqrt(T) = 0) ---------------------------------
# Value = max(P_S - P_K, 0): exact limits away from the forward; at the forward
# the kink makes gamma infinite, so gamma, vanna, volga and theta are NaN while
# delta, rho and vega take the midpoint (d1 = d2 = 0).


def test_zero_vol_away_from_forward():
    S, T, r, q = 100.0, 1.0, 0.05, 0.02
    p_s = S * np.exp(-q * T)
    for K, call_itm in ((80.0, True), (120.0, False)):
        a = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": 0.0}
        p_k = K * np.exp(-r * T)
        itm_flag = call_itm  # the ITM side: calls if call_itm, else puts
        sign = 1.0 if call_itm else -1.0
        assert delta(**a, is_call=itm_flag) == pytest.approx(sign * np.exp(-q * T))
        assert delta(**a, is_call=not itm_flag) == 0.0
        assert theta(**a, is_call=itm_flag) == pytest.approx(sign * (q * p_s - r * p_k))
        assert theta(**a, is_call=not itm_flag) == 0.0
        assert rho(**a, is_call=itm_flag) == pytest.approx(
            sign * K * T * np.exp(-r * T)
        )
        assert rho(**a, is_call=not itm_flag) == 0.0
        for f in (gamma, vega, vanna, volga):
            assert f(**a) == 0.0


def test_zero_vol_at_forward():
    # r = q makes P_S = P_K exactly when S = K, so the forward is hit exactly.
    a = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "q": 0.03, "sigma": 0.0}
    disc = np.exp(-0.03)
    assert delta(**a, is_call=True) == pytest.approx(0.5 * disc)
    assert delta(**a, is_call=False) == pytest.approx(-0.5 * disc)
    assert rho(**a, is_call=True) == pytest.approx(0.5 * 100.0 * disc)
    assert rho(**a, is_call=False) == pytest.approx(-0.5 * 100.0 * disc)
    assert vega(**a) == pytest.approx(100.0 * disc * norm.pdf(0.0))
    for f in (gamma, vanna, volga):
        assert np.isnan(f(**a))
    assert np.isnan(theta(**a, is_call=True))
    assert np.isnan(theta(**a, is_call=False))


def test_expiry_day_greeks():
    # T = 0, sigma > 0. ITM call: delta 1, theta = carry q S - r K, rest 0.
    itm = {"S": 110.0, "K": 100.0, "T": 0.0, "r": 0.05, "q": 0.02, "sigma": 0.2}
    assert delta(**itm, is_call=True) == 1.0
    assert theta(**itm, is_call=True) == pytest.approx(0.02 * 110.0 - 0.05 * 100.0)
    assert rho(**itm, is_call=True) == 0.0
    for f in (gamma, vega, vanna, volga):
        assert f(**itm) == 0.0
    # At the money on expiry: delta 1/2, theta -> -inf, gamma -> inf: NaN.
    atm = {**itm, "S": 100.0}
    assert delta(**atm, is_call=True) == pytest.approx(0.5)
    assert vega(**atm) == 0.0
    assert np.isnan(gamma(**atm))
    assert np.isnan(theta(**atm, is_call=True))


# --- invalid inputs and API ---------------------------------------------------


@pytest.mark.parametrize(
    "bad", [{"sigma": -0.1}, {"S": 0.0}, {"T": -1.0}, {"r": np.nan}]
)
@pytest.mark.parametrize("name", list(GREEKS))
def test_invalid_inputs_give_nan(name, bad):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2}
    assert np.isnan(_greek(name, {**args, **bad}, True))


@pytest.mark.parametrize("name", list(GREEKS))
def test_scalar_in_float_out(name):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2}
    assert type(_greek(name, args, True)) is float


@pytest.mark.parametrize("func", [delta, theta, rho])
def test_is_call_required_for_call_put_dependent_greeks(func):
    with pytest.raises(TypeError):
        func(100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
