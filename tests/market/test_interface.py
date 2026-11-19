"""Tests for pricing_core.market.interface: MarketState and ATM vol interpolation."""

import numpy as np
import pytest

from pricing_core.market.interface import MarketState

TENORS = np.array([1 / 12, 0.25, 1.0])  # 1M, 3M, 1Y
VOLS = np.array([0.10, 0.08, 0.09])  # humped: tests must not assume monotone vols


def _state(n=1, **kw):
    args = dict(spot=1.10, r=0.04, q=0.02, tenors=TENORS, atm_vol=VOLS) | kw
    if n > 1:
        args["spot"] = np.full(n, args["spot"])
    return MarketState(**args)


def test_vol_at_nodes_is_the_node_vol():
    m = _state()
    np.testing.assert_allclose(m.vol(TENORS), VOLS[None, :], rtol=1e-15)


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.9])
def test_total_variance_is_linear_between_nodes(lam):
    # Between tenors T1 < T2, w(T) = sigma(T)^2 T is linear in T, not sigma itself.
    m = _state()
    T = (1 - lam) * TENORS[0] + lam * TENORS[1]
    w1, w2 = VOLS[0] ** 2 * TENORS[0], VOLS[1] ** 2 * TENORS[1]
    expected = np.sqrt(((1 - lam) * w1 + lam * w2) / T)
    assert m.vol(T)[0] == pytest.approx(expected, rel=1e-14)
    # And it is genuinely not linear interpolation in vol.
    assert m.vol(T)[0] != pytest.approx((1 - lam) * VOLS[0] + lam * VOLS[1], rel=1e-6)


def test_flat_vol_outside_the_tenor_range():
    # T = 0 falls in the flat region, so there is no 0/0 (warnings are errors).
    m = _state()
    np.testing.assert_allclose(m.vol(np.array([0.0, 0.01])), [[VOLS[0], VOLS[0]]])
    np.testing.assert_allclose(m.vol(np.array([2.0, 10.0])), [[VOLS[-1], VOLS[-1]]])


def test_vol_output_shape_is_states_by_maturities():
    m = _state(n=4)
    assert m.vol(np.array([0.1, 0.5, 2.0])).shape == (4, 3)
    assert m.vol(0.5).shape == (4,)


def test_each_state_uses_its_own_curve():
    vols = np.stack([VOLS, 2 * VOLS])
    m = _state(spot=np.array([1.1, 1.2]), atm_vol=vols)
    out = m.vol(np.array([0.2, 0.7]))
    # Scaling every node vol by 2 scales the interpolated vol by 2 (w scales by 4).
    np.testing.assert_allclose(out[1], 2 * out[0], rtol=1e-14)


def test_scalars_broadcast_to_n_states():
    m = _state(spot=np.array([1.0, 1.1, 1.2]), r=0.04, q=np.array([0.0, 0.01, 0.02]))
    assert m.n_states == 3
    for field in (m.spot, m.r, m.q):
        assert field.shape == (3,)
    assert m.atm_vol.shape == (3, 3)  # one curve shared by all states, broadcast


def test_all_scalars_is_one_state():
    m = _state()
    assert m.n_states == 1
    assert m.spot.shape == (1,) and m.atm_vol.shape == (1, 3)


def test_fields_are_float64():
    m = _state(spot=np.array([1, 2]), tenors=[1, 2, 3], atm_vol=[1, 1, 1])
    for field in (m.spot, m.r, m.q, m.tenors, m.atm_vol):
        assert field.dtype == np.float64


def test_arrays_are_read_only():
    # States are shared between scenarios: an in-place shock must not leak back.
    m = _state(n=2)
    with pytest.raises(ValueError):
        m.spot[0] = 2.0
    with pytest.raises(ValueError):
        m.atm_vol[0, 0] = 0.5


def test_input_arrays_are_not_frozen_or_aliased():
    spot = np.array([1.0, 1.1])
    m = _state(spot=spot)
    spot[0] = 9.0  # the caller's array stays writable and the state is unaffected
    assert m.spot[0] == 1.0


@pytest.mark.parametrize(
    "kw",
    [
        dict(spot=np.ones(3), r=np.ones(2)),  # state counts disagree
        dict(spot=np.ones(3), atm_vol=np.ones((2, 3))),  # vol rows vs state count
        dict(atm_vol=np.ones(2)),  # vol columns vs number of tenors
        dict(spot=np.ones((3, 1))),  # per-state fields must be at most 1-D
    ],
)
def test_mismatched_shapes_are_rejected(kw):
    with pytest.raises(ValueError):
        _state(**kw)


@pytest.mark.parametrize(
    "tenors",
    [
        np.array([0.25, 1 / 12, 1.0]),  # not increasing
        np.array([1 / 12, 1 / 12, 1.0]),  # repeated
        np.array([0.0, 0.25, 1.0]),  # zero tenor: total variance undefined as a node
    ],
)
def test_tenors_must_be_positive_and_strictly_increasing(tenors):
    with pytest.raises(ValueError):
        _state(tenors=tenors)


def test_negative_node_vol_gives_nan_not_its_absolute_value():
    # Squaring would silently turn -5% into +5%; a large vol-down scenario can
    # push a node below zero, and that state must show up as invalid.
    m = _state(atm_vol=np.array([0.10, -0.05, 0.09]))
    assert np.all(np.isnan(m.vol(np.array([TENORS[1], 0.5]))))


@pytest.mark.parametrize("bad", [-0.05, np.nan])
def test_bad_node_only_affects_brackets_that_contain_it(bad):
    # Bad 2Y node: maturities up to the 3M-1Y bracket never touch it.
    tenors = np.append(TENORS, 2.0)
    m = _state(tenors=tenors, atm_vol=np.array([0.10, 0.08, 0.09, bad]))
    assert np.all(np.isfinite(m.vol(np.array([0.0, 0.1, 0.25, 0.5]))))
    assert np.all(np.isnan(m.vol(np.array([1.5, 5.0]))))
