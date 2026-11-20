"""Tests for pricing_core.book.book: revaluing a book of vanillas across many states."""

import numpy as np
import pytest

from pricing_core.analytic import greeks as g
from pricing_core.analytic.black_scholes import bs_price
from pricing_core.book.book import Book
from pricing_core.book.position import Position
from pricing_core.market.interface import MarketState

TENORS = np.array([1 / 12, 0.25, 1.0])

# Mixed long/short, calls/puts, maturities on a node, between nodes and beyond.
POSITIONS = [
    Position(10.0, 1.10, 1 / 12, is_call=True),
    Position(-5.0, 1.05, 0.2, is_call=False),
    Position(3.0, 1.20, 0.6, is_call=True),
    Position(-2.0, 1.00, 2.0, is_call=False),
]
STATES = MarketState(
    spot=np.array([1.08, 1.10, 1.15]),
    r=np.array([0.04, 0.05, 0.03]),
    q=np.array([0.02, 0.02, 0.03]),
    tenors=TENORS,
    atm_vol=np.array([[0.10, 0.08, 0.09], [0.12, 0.11, 0.10], [0.07, 0.08, 0.09]]),
)
GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga")
_TAKES_IS_CALL = {"delta", "theta", "rho"}


def _one_by_one(func, states, positions, horizon=0.0, takes_is_call=True):
    """Reference: loop over states and positions with scalar pricer calls."""
    out = np.zeros(states.n_states)
    for i in range(states.n_states):
        for p in positions:
            T = p.T - horizon
            sigma = states.vol(T)[i]
            args = (states.spot[i], p.K, T, states.r[i], states.q[i], sigma)
            kw = {"is_call": p.is_call} if takes_is_call else {}
            out[i] += p.quantity * func(*args, **kw)
    return out


def test_value_is_sum_of_positions_priced_one_by_one():
    book = Book(POSITIONS)
    expected = _one_by_one(bs_price, STATES, POSITIONS)
    np.testing.assert_allclose(book.revalue(STATES), expected, rtol=1e-12)


@pytest.mark.parametrize("name", GREEK_NAMES)
def test_book_greeks_are_sums_of_position_greeks(name):
    func = getattr(g, name)
    expected = _one_by_one(
        func, STATES, POSITIONS, takes_is_call=name in _TAKES_IS_CALL
    )
    got = getattr(Book(POSITIONS).greeks(STATES), name)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_horizon_ages_T_and_rolls_down_the_vol_curve():
    # After h years every option has T - h left and is priced at vol(T - h), not
    # at today's vol(T): the term structure is held fixed in maturity.
    h = 1 / 365
    expected = _one_by_one(bs_price, STATES, POSITIONS, horizon=h)
    np.testing.assert_allclose(
        Book(POSITIONS).revalue(STATES, horizon=h), expected, rtol=1e-12
    )


def test_position_expiring_within_horizon_is_worth_intrinsic():
    # T < horizon: the option has expired by the horizon and settles at the
    # state's spot, so it is worth intrinsic, not NaN from a negative T.
    p = Position(4.0, 1.10, 0.5 / 365, is_call=True)
    value = Book([p]).revalue(STATES, horizon=1 / 365)
    np.testing.assert_allclose(value, 4.0 * np.maximum(STATES.spot - 1.10, 0.0))


def test_negative_horizon_is_rejected():
    with pytest.raises(ValueError):
        Book(POSITIONS).revalue(STATES, horizon=-1 / 365)


def test_long_call_short_put_is_a_forward():
    # Put-call parity at book level, independent of the pricer code paths:
    # C - P = S e^(-qT) - K e^(-rT) in every state, whatever the vol.
    K, T = 1.12, 0.4
    book = Book(
        [Position(1.0, K, T, is_call=True), Position(-1.0, K, T, is_call=False)]
    )
    s = STATES
    forward_pv = s.spot * np.exp(-s.q * T) - K * np.exp(-s.r * T)
    np.testing.assert_allclose(book.revalue(s), forward_pv, rtol=1e-12, atol=1e-15)


def test_identical_states_give_identical_values():
    one = MarketState(1.1, 0.04, 0.02, TENORS, STATES.atm_vol[0])
    many = MarketState(np.full(5, 1.1), 0.04, 0.02, TENORS, STATES.atm_vol[0])
    book = Book(POSITIONS)
    np.testing.assert_array_equal(book.revalue(many), np.repeat(book.revalue(one), 5))


def test_output_shapes():
    book = Book(POSITIONS)
    assert book.revalue(STATES).shape == (3,)
    for name in GREEK_NAMES:
        assert getattr(book.greeks(STATES), name).shape == (3,)


def test_empty_book_is_worth_zero():
    book = Book([])
    np.testing.assert_array_equal(book.revalue(STATES), np.zeros(3))
    for name in GREEK_NAMES:
        np.testing.assert_array_equal(getattr(book.greeks(STATES), name), np.zeros(3))


def test_bad_state_is_nan_and_does_not_affect_other_states():
    # A vol-down scenario pushes one state's node negative: that state is NaN
    # (visible as invalid), the others revalue normally.
    vols = STATES.atm_vol.copy()
    vols[1, :] = -0.01
    states = MarketState(STATES.spot, STATES.r, STATES.q, TENORS, vols)
    value = Book(POSITIONS).revalue(states)
    assert np.isnan(value[1])
    np.testing.assert_allclose(value[[0, 2]], Book(POSITIONS).revalue(STATES)[[0, 2]])


def test_positions_are_held_as_a_tuple():
    # The book must not change if the caller later mutates their list.
    positions = list(POSITIONS)
    book = Book(positions)
    positions.clear()
    assert book.positions == tuple(POSITIONS)
