"""Tests for the reporting-unit helpers in pricing_core.conventions."""

import numpy as np
import pytest

from pricing_core.conventions import DAYS_PER_YEAR, per_day, per_vol_point


def test_days_per_year_is_calendar():
    # Theta is per calendar day: T counts calendar time, so decay runs over weekends.
    assert DAYS_PER_YEAR == 365.0


def test_per_day_divides_by_365():
    assert per_day(-7.3) == pytest.approx(-0.02, rel=1e-15)


def test_per_day_keeps_sign_and_broadcasts():
    theta = np.array([-3.65, 0.0, 1.825])  # long option, zero, a positive theta
    np.testing.assert_allclose(per_day(theta), [-0.01, 0.0, 0.005], rtol=1e-15)


def test_per_vol_point_divides_by_100():
    # Vega per 1.00 of vol -> per vol point (0.01).
    assert per_vol_point(37.5) == pytest.approx(0.375, rel=1e-15)
    np.testing.assert_allclose(per_vol_point(np.array([10.0, 0.0])), [0.1, 0.0])


@pytest.mark.parametrize("func", [per_day, per_vol_point])
def test_scalar_in_float_out(func):
    assert type(func(1.0)) is float
