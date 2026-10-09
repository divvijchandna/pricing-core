"""Black-Scholes-Merton prices with a continuous yield q, and Black-76 via q = r.

Notation: N(x) is the standard normal CDF, n(x) the density. With the prepaid
forward of the stock P_S = S e^(-qT) and the present value of the strike
P_K = K e^(-rT):

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
       = ln(P_S/P_K) / (sigma sqrt(T)) + sigma sqrt(T) / 2
    d2 = d1 - sigma sqrt(T)
    C  = P_S N(d1) - P_K N(d2)
    P  = P_K N(-d2) - P_S N(-d1)

All inputs broadcast as NumPy arrays; T in years, r and q continuously
compounded, sigma annualised as a decimal (0.20, not 20).

Invalid inputs give NaN in that position, never an exception or a warning, so
one bad scenario cannot stop a vectorised run or pass for a real price. Invalid
means S <= 0, K <= 0, T < 0, sigma < 0, or any input that is NaN or +/-inf.

References: Black and Scholes (1973); Merton (1973) for the continuous yield;
Black (1976) for options on futures; Hull, Options, Futures, and Other
Derivatives, "Black-Scholes-Merton model" and "Futures options and Black's
model" chapters.
"""

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


def _sanitise(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> tuple[
    BoolArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
]:
    """Validity mask, plus inputs with invalid positions replaced by placeholders.

    The placeholders (S = K = T = sigma = 1, r = q = 0) are valid, so computing
    with them raises no warning; callers overwrite those positions with NaN.
    Comparisons with NaN are False, so NaN inputs are marked invalid without
    a warning.
    """
    S, K, T, r, q, sigma = (as_float_array(a) for a in (S, K, T, r, q, sigma))
    valid = (
        np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(q)
        & np.isfinite(sigma)
        & (S > 0)
        & (K > 0)
        & (T >= 0)
        & (sigma >= 0)
    )
    return (
        valid,
        np.where(valid, S, 1.0),
        np.where(valid, K, 1.0),
        np.where(valid, T, 1.0),
        np.where(valid, r, 0.0),
        np.where(valid, q, 0.0),
        np.where(valid, sigma, 1.0),
    )


def _d1_d2_valid(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """d1 and d2 for inputs already sanitised by _sanitise (see d1_d2)."""
    log_moneyness = np.log(S / K) + (r - q) * T  # ln(P_S / P_K)
    v = sigma * np.sqrt(T)  # total volatility to expiry
    has_vol = v > 0
    v_safe = np.where(has_vol, v, 1.0)  # unused where v = 0; avoids 0/0
    limit = np.where(
        log_moneyness > 0, np.inf, np.where(log_moneyness < 0, -np.inf, 0.0)
    )
    d1 = np.where(has_vol, log_moneyness / v_safe + 0.5 * v_safe, limit)
    d2 = np.where(has_vol, log_moneyness / v_safe - 0.5 * v_safe, limit)
    return d1, d2


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Black-Scholes d1 and d2, defined everywhere including sigma sqrt(T) = 0.

    Where sigma sqrt(T) > 0 the usual formulas apply. Where it is 0, d1 and d2
    take their limits as sigma sqrt(T) -> 0: +inf if P_S > P_K, -inf if
    P_S < P_K, and 0 at the forward (P_S = P_K, where d1 = -d2 = sigma sqrt(T)/2
    -> 0). No division by zero occurs in either case.

    Invalid inputs (see module docstring) give NaN. Returns float64 arrays
    (0-d for scalar inputs).
    """
    valid, S, K, T, r, q, sigma = _sanitise(S, K, T, r, q, sigma)
    d1, d2 = _d1_d2_valid(S, K, T, r, q, sigma)
    return np.where(valid, d1, np.nan), np.where(valid, d2, np.nan)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """European option price under Black-Scholes-Merton with continuous yield q.

        C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
        P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    q is the dividend yield for equities, the foreign rate for FX
    (Garman-Kohlhagen), or r for futures (see black76_price).

    Where sigma sqrt(T) = 0 the price is set explicitly to the discounted
    forward intrinsic value, max(S e^(-qT) - K e^(-rT), 0) for calls and
    max(K e^(-rT) - S e^(-qT), 0) for puts; at T = 0 this is max(S - K, 0) and
    max(K - S, 0).

    Args:
        S: spot price (> 0).
        K: strike (> 0).
        T: time to expiry in years (>= 0).
        r: risk-free rate, continuously compounded, per year.
        q: continuous yield, per year.
        sigma: volatility, annualised, as a decimal (>= 0).
        is_call: True for calls, False for puts; a bool or bool array.
            Anything else raises TypeError.

    Returns:
        Price in the same currency units as S and K, NaN where the inputs are
        invalid (see module docstring). A Python float when the broadcast
        result is a scalar, otherwise a float64 array.
    """
    call = as_bool_array(is_call)
    valid, S, K, T, r, q, sigma = _sanitise(S, K, T, r, q, sigma)

    p_s = S * np.exp(-q * T)  # prepaid forward of the stock
    p_k = K * np.exp(-r * T)  # present value of the strike
    d1, d2 = _d1_d2_valid(S, K, T, r, q, sigma)

    # N(-x) is computed directly, not as 1 - N(x), to keep precision in the tails.
    formula = np.where(
        call,
        p_s * ndtr(d1) - p_k * ndtr(d2),
        p_k * ndtr(-d2) - p_s * ndtr(-d1),
    )
    limit = np.where(call, np.maximum(p_s - p_k, 0.0), np.maximum(p_k - p_s, 0.0))
    has_vol = sigma * np.sqrt(T) > 0
    price = np.where(has_vol, formula, limit)
    return to_output(np.where(valid, price, np.nan))


def black76_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    is_call: bool | ArrayLike,
) -> FloatOrArray:
    """Black (1976) price of a European option on a futures or forward price F.

        C = e^(-rT) [F N(d1) - K N(d2)],  P = e^(-rT) [K N(-d2) - F N(-d1)]
        d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)

    Not a separate implementation: bs_price with S = F and q = r gives the same
    prepaid forward, F e^(-rT), and zero drift, so the formulas coincide. Same
    units, edge-case handling, invalid-input NaN and return type as bs_price.
    """
    return bs_price(F, K, T, r, q=r, sigma=sigma, is_call=is_call)
