"""Tests for the Deribit public REST adapter.

No live network calls: parsing is tested on a saved slice of a real
get_book_summary_by_currency response (tests/fixtures/), and the fetch path is
tested with urlopen replaced by a stub.
"""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from pricing_core.market.adapters import deribit
from pricing_core.market.adapters.deribit import (
    OptionChain,
    fetch_option_chain,
    parse_book_summary,
    parse_instrument_name,
)

FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "deribit_book_summary_btc_option.json"
)


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


# --- instrument names -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expiry", "strike", "is_call"),
    [
        ("BTC-19OCT26-80000-C", datetime(2026, 10, 19, 8, tzinfo=UTC), 80000.0, True),
        ("BTC-6NOV26-70000-C", datetime(2026, 11, 6, 8, tzinfo=UTC), 70000.0, True),
        ("BTC-25JUN27-100000-P", datetime(2027, 6, 25, 8, tzinfo=UTC), 100000.0, False),
    ],
)
def test_parse_instrument_name(name, expiry, strike, is_call):
    # Deribit options expire at 08:00 UTC; the day can be one digit (6NOV26).
    assert parse_instrument_name(name) == (expiry, strike, is_call)


@pytest.mark.parametrize("bad", ["BTC-PERPETUAL", "BTC-19OCT26-80000-X", "BTC-19OCT26"])
def test_parse_instrument_name_rejects_non_options(bad):
    with pytest.raises(ValueError):
        parse_instrument_name(bad)


# --- book summary ------------------------------------------------------------------


def test_parse_book_summary_shapes_and_types(payload):
    chain = parse_book_summary(payload)
    n = len(payload["result"])
    assert isinstance(chain, OptionChain)
    assert len(chain.instrument) == n
    for arr in (
        chain.T,
        chain.strike,
        chain.forward,
        chain.mark_iv,
        chain.bid,
        chain.ask,
        chain.mark,
    ):
        assert arr.shape == (n,)
        assert arr.dtype == np.float64
    assert chain.is_call.dtype == np.bool_


def test_conventions_converted_at_the_boundary(payload):
    chain = parse_book_summary(payload)
    row = payload["result"][0]  # BTC-19OCT26-80000-C
    assert chain.instrument[0] == "BTC-19OCT26-80000-C"
    assert chain.mark_iv[0] == pytest.approx(row["mark_iv"] / 100)  # percent -> decimal
    assert chain.forward[0] == row["underlying_price"]  # forward for that expiry, USD
    assert chain.mark[0] == row["mark_price"]  # stays in BTC
    assert chain.strike[0] == 80000.0
    assert chain.is_call[0]
    # T: ACT/365 years from this row's creation_timestamp to 08:00 UTC on expiry.
    now = datetime.fromtimestamp(row["creation_timestamp"] / 1000, tz=UTC)
    expiry = datetime(2026, 10, 19, 8, tzinfo=UTC)
    expected_T = (expiry - now).total_seconds() / (365 * 86400)
    assert chain.T[0] == pytest.approx(expected_T, rel=1e-12)


def test_missing_bid_or_ask_is_nan(payload):
    chain = parse_book_summary(payload)
    names = list(chain.instrument)
    i = names.index("BTC-18OCT26-72000-P")  # no bid
    assert np.isnan(chain.bid[i])
    assert chain.ask[i] == 0.0001
    j = names.index("BTC-19OCT26-88000-C")  # no bid, no ask
    assert np.isnan(chain.bid[j])
    assert np.isnan(chain.ask[j])


def test_expired_rows_are_dropped(payload):
    # A row whose expiry is at or before its snapshot time has T <= 0: dropped.
    stale = dict(payload["result"][0], instrument_name="BTC-1JAN20-80000-C")
    chain = parse_book_summary({**payload, "result": [stale, *payload["result"]]})
    assert "BTC-1JAN20-80000-C" not in chain.instrument
    assert len(chain.instrument) == len(payload["result"])


def test_snapshot_time_is_latest_row_time(payload):
    chain = parse_book_summary(payload)
    latest = max(x["creation_timestamp"] for x in payload["result"])
    assert chain.snapshot_time == datetime.fromtimestamp(latest / 1000, tz=UTC)


def test_error_payload_raises():
    with pytest.raises(RuntimeError, match="Deribit"):
        parse_book_summary({"jsonrpc": "2.0", "error": {"code": 10001, "message": "x"}})


# --- fetch (network stubbed) ----------------------------------------------------------


def test_fetch_builds_public_url_and_parses(payload, monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout, context):
        seen["url"], seen["timeout"], seen["context"] = url, timeout, context
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(deribit.urllib.request, "urlopen", fake_urlopen)
    chain = fetch_option_chain("BTC", timeout=5.0)
    assert seen["url"] == (
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        "?currency=BTC&kind=option"
    )
    assert seen["timeout"] == 5.0
    assert seen["context"].verify_mode.name == "CERT_REQUIRED"  # never unverified
    assert len(chain.instrument) == len(payload["result"])
