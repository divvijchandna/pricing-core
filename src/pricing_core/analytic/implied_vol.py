"""Vectorised Black-Scholes implied volatility.

Notation as in black_scholes: P_S = S e^(-qT), P_K = K e^(-rT), x = ln(P_S/P_K).

Existence: the price is strictly increasing in sigma (vega > 0), from the lower
no-arbitrage bound at sigma = 0 to the upper bound as sigma -> inf. A price
strictly inside the bounds has exactly one implied vol; anything else has none
and returns NaN (a price exactly at the lower bound would be sigma = 0, which is
not a positive vol, so it is NaN too).

Method, per element:
1. Conditioning. d_sigma ~ d_price / vega, and vega is the same for a call and a
   put at the same strike, but an in-the-money price carries its intrinsic value,
   and with it a rounding error far larger than the time value that determines
   sigma. So the price is converted by put-call parity, C - P = P_S - P_K, to the
   out-of-the-money option (call if P_S <= P_K, else put), whose price is the
   time value alone. All solving happens on that side, with bounds (0, upper),
   upper = P_S for calls and P_K for puts.
2. Start. sigma* = sqrt(2 |x| / T) is where volga = vega d1 d2 / sigma changes
   sign: the price is convex in sigma below sigma* and concave above it, so
   Newton started at sigma* moves monotonically towards the root without
   overshooting. The price at sigma* also splits the bracket: if it is above the
   target, the root is below sigma* and Newton starts there; otherwise the root
   is above, where sigma* can be far too low near the money (sigma* -> 0 as
   x -> 0), so the start is the Brenner-Subrahmanyam estimate
   sqrt(2 pi / T) * price / P_S, kept inside the bracket.
3. Iterate. Newton, sigma_new = sigma - (C(sigma) - target) / vega, safeguarded
   by a bracket [lo, hi] (initially [0, 5]) that shrinks every iteration using
   the sign of C(sigma) - target. Where vega < 1e-8 S, or the Newton step would
   leave the bracket, bisection is used instead. The bracket guarantees
   convergence: from sigma* Newton can creep down a flat tail where a step of
   rounding size can land at sigma <= 0.
4. Stop when |C(sigma) - target| <= 1e-12 * target, or the bracket is a few ulps
   wide. Elements not converged within max_iter return NaN, never a guess.

Black-76: implied_vol(price, F, K, T, r, r, is_call=...), i.e. q = r.

References: Brenner and Subrahmanyam (1988), "A simple formula to compute the
implied standard deviation", Financial Analysts Journal 44(5); Jaeckel (2015),
"Let's Be Rational", Wilmott (the inflection point sigma* as a start); Press et
al., Numerical Recipes, 3rd ed., section 9.4 (Newton-Raphson with bisection,
rtsafe).
"""

import numpy as np
from numpy.typing import ArrayLike

from pricing_core._types import (
    FloatOrArray,
    as_bool_array,
    as_float_array,
    to_output,
)
from pricing_core.analytic.black_scholes import _sanitise, bs_price
from pricing_core.analytic.greeks import vega

SIGMA_MAX = 5.0  # upper end of the search bracket (500% vol)
PRICE_RTOL = 1e-12  # stop when |C(sigma) - target| <= PRICE_RTOL * target
VEGA_MIN = 1e-8  # bisect where vega < VEGA_MIN * S: Newton step unreliable
_EPS = np.finfo(np.float64).eps


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    *,
    is_call: bool | ArrayLike,
    max_iter: int = 100,
) -> FloatOrArray:
    """Black-Scholes-Merton implied volatility, annualised, as a decimal.

    Args:
        price: option price, in the same currency units as S and K.
        S, K, T, r, q: as in bs_price (T in years, rates continuously
            compounded).
        is_call: True for calls, False for puts; a bool or bool array. The
            price may be in or out of the money; it is converted by parity.
        max_iter: maximum number of Newton/bisection updates per element.

    Returns:
        sigma with bs_price(S, K, T, r, q, sigma) = price. NaN where no positive
        sigma exists (price on or outside the no-arbitrage bounds, or needing
        sigma > SIGMA_MAX), where an input is invalid (as in bs_price, plus
        T = 0 and a non-finite price), or where the solver has not converged
        within max_iter. Never raises for bad values. A Python float when the
        result is a scalar, otherwise a float64 array.
    """
    call = as_bool_array(is_call)
    price = as_float_array(price)
    valid, S, K, T, r, q, _ = _sanitise(S, K, T, r, q, 1.0)
    shape = np.broadcast_shapes(
        call.shape, price.shape, S.shape, K.shape, T.shape, r.shape, q.shape
    )
    call, price, S, K, T, r, q, valid = (
        np.broadcast_to(a, shape) for a in (call, price, S, K, T, r, q, valid)
    )
    valid = valid & np.isfinite(price) & (T > 0)
    T = np.where(valid, T, 1.0)  # placeholder: keeps sqrt and division safe

    # 1. Convert to the out-of-the-money side by parity: C - P = P_S - P_K.
    p_s = S * np.exp(-q * T)
    p_k = K * np.exp(-r * T)
    otm_call = p_s <= p_k
    target = np.where(
        call == otm_call,
        price,
        np.where(call, price - (p_s - p_k), price + (p_s - p_k)),
    )
    upper = np.where(otm_call, p_s, p_k)

    def otm_price(sigma):
        return np.asarray(bs_price(S, K, T, r, q, sigma, is_call=otm_call))

    valid = valid & (target > 0) & (target < upper)
    valid = valid & (target < otm_price(SIGMA_MAX))  # root inside the bracket
    target = np.where(valid, target, 0.5 * upper)  # placeholder, never returned

    # 2. Start: split the bracket at sigma*, then choose the starting point.
    sqrt_t = np.sqrt(T)
    sigma_star = np.sqrt(2.0 * np.abs(np.log(p_s / p_k)) / T)
    sigma_star = np.clip(sigma_star, 0.0, SIGMA_MAX)
    star_is_high = otm_price(sigma_star) >= target  # root <= sigma*
    lo = np.where(star_is_high, 0.0, sigma_star)
    hi = np.where(star_is_high, sigma_star, SIGMA_MAX)
    brenner = np.sqrt(2.0 * np.pi) * target / (p_s * sqrt_t)
    inside = (brenner > lo) & (brenner < hi)
    sigma = np.where(
        star_is_high,
        sigma_star,
        np.where(inside, brenner, 0.5 * (lo + hi)),
    )
    sigma = np.where(sigma > 0, sigma, 0.5 * (lo + hi))  # at the forward sigma* = 0

    # 3-4. Safeguarded Newton.
    done = ~valid
    converged = np.zeros(shape, dtype=bool)
    for iteration in range(max_iter + 1):
        f = otm_price(sigma) - target
        now = ~done & ((np.abs(f) <= PRICE_RTOL * target) | (hi - lo <= 4 * _EPS * hi))
        converged = converged | now
        done = done | now
        if done.all() or iteration == max_iter:
            break
        hi = np.where(f > 0, sigma, hi)
        lo = np.where(f < 0, sigma, lo)
        v = np.asarray(vega(S, K, T, r, q, sigma))
        usable = v > VEGA_MIN * S
        newton = sigma - f / np.where(usable, v, 1.0)
        bisect = ~usable | (newton <= lo) | (newton >= hi)
        sigma = np.where(done, sigma, np.where(bisect, 0.5 * (lo + hi), newton))

    return to_output(np.where(valid & converged, sigma, np.nan))
