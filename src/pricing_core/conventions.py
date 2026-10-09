"""Reporting-unit conventions for Greeks.

Greeks are computed internally per unit of the bumped quantity: theta per year,
vega (and vanna, volga) per 1.00 of vol. These helpers rescale to the units desks
quote. Each is the first-order price change for the reporting bump:
dV ~ (dV/dx) * dx, with dx = 1 day = 1/365 year, or dx = 1 vol point = 0.01.
"""

from numpy.typing import ArrayLike

from pricing_core._types import FloatOrArray, as_float_array, to_output

DAYS_PER_YEAR = 365.0
"""Calendar days: T counts calendar time, so an option decays over weekends too.
Trading-day theta would use 252 instead."""


def per_day(theta: ArrayLike) -> FloatOrArray:
    """Convert theta from per year to per calendar day: theta / 365.

    The sign is kept: theta is usually negative for a long option, but can be
    positive (e.g. a deep in-the-money European put).
    """
    return to_output(as_float_array(theta) / DAYS_PER_YEAR)


def per_vol_point(greek: ArrayLike) -> FloatOrArray:
    """Convert a vol sensitivity from per 1.00 of vol to per vol point: x / 100.

    Applies to vega, and to vanna and volga with respect to their vol leg.
    """
    return to_output(as_float_array(greek) / 100.0)
