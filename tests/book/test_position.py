"""Tests for pricing_core.book.position: one vanilla option contract and its size."""

import dataclasses

import numpy as np
import pytest

from pricing_core.book.position import Position


def test_fields_are_stored_as_float():
    p = Position(-2, 100, 1, is_call=True)
    assert (p.quantity, p.K, p.T) == (-2.0, 100.0, 1.0)
    assert all(type(x) is float for x in (p.quantity, p.K, p.T))


def test_negative_quantity_is_a_short():
    # Sign carries direction: the Book sums quantity * value, so shorts subtract.
    assert Position(-3.5, 100.0, 0.5, is_call=False).quantity == -3.5


def test_is_call_is_keyword_only():
    # Same convention as the pricers: (..., *, is_call), so a put can't be passed
    # positionally by mistake.
    with pytest.raises(TypeError):
        Position(1.0, 100.0, 1.0, True)


@pytest.mark.parametrize("bad", [1, 0, "C", None])
def test_is_call_must_be_a_bool(bad):
    # Strict like as_bool_array: is_call="C" or 1 would otherwise be truthy.
    with pytest.raises(TypeError):
        Position(1.0, 100.0, 1.0, is_call=bad)


def test_numpy_bool_is_accepted_and_stored_as_bool():
    # Positions built by iterating over a bool array receive np.bool_.
    p = Position(1.0, 100.0, 1.0, is_call=np.bool_(False))
    assert p.is_call is False


def test_expiring_today_is_allowed():
    # T = 0 is a valid contract (priced at intrinsic), not an error.
    assert Position(1.0, 100.0, 0.0, is_call=True).T == 0.0


@pytest.mark.parametrize(
    "field, bad",
    [
        ("K", 0.0),
        ("K", -1.0),
        ("K", np.nan),
        ("K", np.inf),
        ("T", -0.1),
        ("T", np.nan),
        ("T", np.inf),
        ("quantity", np.nan),
        ("quantity", np.inf),
    ],
)
def test_invalid_contract_terms_are_rejected(field, bad):
    # A contract is static data: a bad strike or expiry is a booking error, so it
    # fails loudly at construction instead of becoming NaN in every scenario.
    kw = dict(quantity=1.0, K=100.0, T=1.0) | {field: bad}
    with pytest.raises(ValueError):
        Position(**kw, is_call=True)


def test_is_immutable():
    p = Position(1.0, 100.0, 1.0, is_call=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.K = 90.0
