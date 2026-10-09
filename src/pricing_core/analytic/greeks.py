"""Analytic Black-Scholes-Merton Greeks with a continuous yield q.

Notation as in black_scholes: N(x) normal CDF, n(x) normal density,
P_S = S e^(-qT), P_K = K e^(-rT), d1 and d2 from d1_d2.

Units (internal; see conventions for reporting units):
    delta  per 1.00 of S          gamma  per 1.00 of S, squared
    vega   per 1.00 of vol        theta  per year, theta = dV/dt = -dV/dT
    rho    per 1.00 of r          vanna  per 1.00 of S and per 1.00 of vol
    volga  per 1.00 of vol, squared

Gamma, vega, vanna and volga are identical for calls and puts, so they take no
is_call argument.

Zero total volatility (sigma sqrt(T) = 0, e.g. on expiry day): the value is
max(P_S - P_K, 0). Away from the forward every Greek takes its exact limit
(delta and rho are step functions; theta is the carry q P_S - r P_K or 0; gamma,
vega, vanna, volga are 0). At the forward (P_S = P_K) the value has a kink:
delta and rho take the midpoint (d1 = d2 = 0), vega its one-sided limit
P_S n(0) sqrt(T), and gamma, vanna, volga and theta are NaN (gamma is infinite
there, and at-the-money theta tends to -inf at expiry).

Invalid inputs give NaN, as in bs_price.

Black-76: every Greek here applies with S = F and q = r (delta is then dV/dF),
except rho, which must hold F fixed rather than S: use black76_rho.

References: Hull, Options, Futures, and Other Derivatives, "The Greek letters"
chapter; Haug, The Complete Guide to Option Pricing Formulas, ch. 2 (vanna,
volga).
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from pricing_core._types import (
    BoolArray,
    FloatArray,
    FloatOrArray,
    as_bool_array,
    as_float_array,
    to_output,
)
from pricing_core.analytic.black_scholes import (
    _d1_d2_valid,
    _sanitise,
    black76_price,
)

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _npdf(x: FloatArray) -> FloatArray:
    """Standard normal density n(x); n(+/-inf) = 0 without a warning."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


class _Inputs(NamedTuple):
    valid: BoolArray
    S: FloatArray
    K: FloatArray
    T: FloatArray
    r: FloatArray
    q: FloatArray
    sigma: FloatArray
    d1: FloatArray
    d2: FloatArray
    n_d1: FloatArray
    has_vol: BoolArray  # sigma sqrt(T) > 0
    at_forward: BoolArray  # sigma sqrt(T) = 0 and P_S = P_K
    # Safe stand-ins for the has_vol branch: equal to the real values where
    # has_vol, finite placeholders elsewhere, so the unused branch never
    # divides by zero or multiplies inf by 0.
    sigma_safe: FloatArray
    sqrt_t_safe: FloatArray
    d1_safe: FloatArray
    d2_safe: FloatArray


def _prepare(S, K, T, r, q, sigma) -> _Inputs:
    valid, S, K, T, r, q, sigma = _sanitise(S, K, T, r, q, sigma)
    d1, d2 = _d1_d2_valid(S, K, T, r, q, sigma)
    sqrt_t = np.sqrt(T)
    has_vol = sigma * sqrt_t > 0
    # With no vol, d1 is +inf, -inf, or exactly 0 at the forward.
    at_forward = ~has_vol & (d1 == 0.0)
    return _Inputs(
        valid=valid,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        d1=d1,
        d2=d2,
        n_d1=_npdf(d1),
        has_vol=has_vol,
        at_forward=at_forward,
        sigma_safe=np.where(has_vol, sigma, 1.0),
        sqrt_t_safe=np.where(has_vol, sqrt_t, 1.0),
        d1_safe=np.where(has_vol, d1, 0.0),
        d2_safe=np.where(has_vol, d2, 0.0),
    )


def _finish(x: FloatArray, p: _Inputs, *, singular_at_forward: bool) -> FloatOrArray:
    if singular_at_forward:
        x = np.where(p.at_forward, np.nan, x)
    return to_output(np.where(p.valid, x, np.nan))


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """dV/dS. Call: e^(-qT) N(d1). Put: -e^(-qT) N(-d1). Per 1.00 of S."""
    call = as_bool_array(is_call)
    p = _prepare(S, K, T, r, q, sigma)
    x = np.exp(-p.q * p.T) * np.where(call, ndtr(p.d1), -ndtr(-p.d1))
    return _finish(x, p, singular_at_forward=False)


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> FloatOrArray:
    """d2V/dS2 = e^(-qT) n(d1) / (S sigma sqrt(T)). Calls and puts. Per S^2."""
    p = _prepare(S, K, T, r, q, sigma)
    formula = np.exp(-p.q * p.T) * p.n_d1 / (p.S * p.sigma_safe * p.sqrt_t_safe)
    x = np.where(p.has_vol, formula, 0.0)
    return _finish(x, p, singular_at_forward=True)


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> FloatOrArray:
    """dV/dsigma = S e^(-qT) n(d1) sqrt(T). Calls and puts. Per 1.00 of vol.

    Per vol point: conventions.per_vol_point. Finite everywhere: with no vol
    it is 0 away from the forward and P_S n(0) sqrt(T) at the forward.
    """
    p = _prepare(S, K, T, r, q, sigma)
    x = p.S * np.exp(-p.q * p.T) * p.n_d1 * np.sqrt(p.T)
    return _finish(x, p, singular_at_forward=False)


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """dV/dt = -dV/dT, per year (per calendar day: conventions.per_day).

    Call: -S e^(-qT) n(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)
          + q S e^(-qT) N(d1)
    Put:  -S e^(-qT) n(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2)
          - q S e^(-qT) N(-d1)
    The first term is time decay; the rest is carry on the replicating
    portfolio. With no vol the decay term is 0, except at the forward (NaN).
    """
    call = as_bool_array(is_call)
    p = _prepare(S, K, T, r, q, sigma)
    p_s = p.S * np.exp(-p.q * p.T)
    p_k = p.K * np.exp(-p.r * p.T)
    decay = np.where(
        p.has_vol, -p_s * p.n_d1 * p.sigma_safe / (2.0 * p.sqrt_t_safe), 0.0
    )
    carry = np.where(
        call,
        -p.r * p_k * ndtr(p.d2) + p.q * p_s * ndtr(p.d1),
        p.r * p_k * ndtr(-p.d2) - p.q * p_s * ndtr(-p.d1),
    )
    return _finish(decay + carry, p, singular_at_forward=True)


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """dV/dr with S and q held fixed, per 1.00 of r.

    Call: K T e^(-rT) N(d2). Put: -K T e^(-rT) N(-d2). For Black-76 (F held
    fixed) use black76_rho instead.
    """
    call = as_bool_array(is_call)
    p = _prepare(S, K, T, r, q, sigma)
    x = p.K * p.T * np.exp(-p.r * p.T) * np.where(call, ndtr(p.d2), -ndtr(-p.d2))
    return _finish(x, p, singular_at_forward=False)


def vanna(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> FloatOrArray:
    """d2V/dS dsigma = d(delta)/dsigma = -e^(-qT) n(d1) d2 / sigma.

    Calls and puts. Per 1.00 of S and per 1.00 of vol. Zero where d2 = 0.
    """
    p = _prepare(S, K, T, r, q, sigma)
    formula = -np.exp(-p.q * p.T) * p.n_d1 * p.d2_safe / p.sigma_safe
    x = np.where(p.has_vol, formula, 0.0)
    return _finish(x, p, singular_at_forward=True)


def volga(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> FloatOrArray:
    """d2V/dsigma2 = d(vega)/dsigma = vega d1 d2 / sigma.

    Calls and puts. Per 1.00 of vol, squared. Zero where d1 d2 = 0, which is
    where the price switches from convex to concave in sigma.
    """
    p = _prepare(S, K, T, r, q, sigma)
    vega_ = p.S * np.exp(-p.q * p.T) * p.n_d1 * np.sqrt(p.T)
    formula = vega_ * p.d1_safe * p.d2_safe / p.sigma_safe
    x = np.where(p.has_vol, formula, 0.0)
    return _finish(x, p, singular_at_forward=True)


def black76_rho(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """dV/dr for Black-76 with the futures price F held fixed: -T V.

    Differs from rho(F, K, T, r, r, sigma): there q = r moves with r, which
    rho (S and q held fixed) does not capture. Here r only discounts the
    payoff, V = e^(-rT) [F N(d1) - K N(d2)], so dV/dr = -T V. Per 1.00 of r.
    """
    price = as_float_array(black76_price(F, K, T, r, sigma, is_call=is_call))
    return to_output(-as_float_array(T) * price)
