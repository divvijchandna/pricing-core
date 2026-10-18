"""Deribit public REST adapter: option book summaries (no account or API key).

Deribit's conventions are converted here, at the adapter boundary:
- mark_iv is quoted in percent; converted to a decimal.
- Expiry comes from the instrument name (BTC-19OCT26-80000-C): 08:00 UTC on
  that date.
- T is ACT/365 years from each row's creation_timestamp to expiry. Rows with
  T <= 0 (already expired) are dropped.
- underlying_price is the forward (future or synthetic) for that expiry, in USD.
- bid, ask and mark stay in the base coin (BTC per option on 1 BTC). A missing
  bid or ask (empty side of the book) becomes NaN.

TLS: the optional extra `pricing-core[deribit]` installs certifi, used as the CA
bundle when present (python.org builds of Python on macOS ship without one).
Certificate verification is never disabled.

Reference: Deribit API v2, public/get_book_summary_by_currency.
"""

import json
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from pricing_core._types import BoolArray, FloatArray

BASE_URL = "https://www.deribit.com/api/v2"
_EXPIRY_HOUR_UTC = 8
_SECONDS_PER_YEAR = 365.0 * 86400.0
_EXPIRY_RE = re.compile(r"^(\d{1,2})([A-Z]{3})(\d{2})$")
# Explicit month map: datetime.strptime("%b") depends on the system locale.
_MONTHS = {
    m: i
    for i, m in enumerate(
        (
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ),
        start=1,
    )
}


@dataclass(frozen=True)
class OptionChain:
    """One snapshot of an option chain as aligned arrays, one entry per option.

    Prices (bid, ask, mark) in the base coin; forward and strike in USD; T in
    years (ACT/365); mark_iv as a decimal.
    """

    snapshot_time: datetime
    instrument: tuple[str, ...]
    expiry: tuple[datetime, ...]
    T: FloatArray
    strike: FloatArray
    is_call: BoolArray
    forward: FloatArray
    mark_iv: FloatArray
    bid: FloatArray
    ask: FloatArray
    mark: FloatArray


def parse_instrument_name(name: str) -> tuple[datetime, float, bool]:
    """'BTC-19OCT26-80000-C' -> (2026-10-19 08:00 UTC, 80000.0, True)."""
    parts = name.split("-")
    if len(parts) != 4 or parts[3] not in ("C", "P"):
        raise ValueError(f"Not a Deribit option instrument: {name!r}")
    match = _EXPIRY_RE.match(parts[1])
    if match is None or match.group(2) not in _MONTHS:
        raise ValueError(f"Unrecognised expiry in {name!r}")
    day, month, year = match.groups()
    expiry = datetime(
        2000 + int(year), _MONTHS[month], int(day), _EXPIRY_HOUR_UTC, tzinfo=UTC
    )
    return expiry, float(parts[2]), parts[3] == "C"


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def _float_or_nan(x: Any) -> float:
    return np.nan if x is None else float(x)


def parse_book_summary(payload: dict[str, Any]) -> OptionChain:
    """Convert a get_book_summary_by_currency response into an OptionChain."""
    if "result" not in payload:
        raise RuntimeError(f"Deribit API error: {payload.get('error')}")
    result = payload["result"]
    if not result:
        raise RuntimeError("Deribit API returned no instruments")

    rows = []
    for row in result:
        expiry, strike, is_call = parse_instrument_name(row["instrument_name"])
        seconds = (expiry - _ms_to_utc(row["creation_timestamp"])).total_seconds()
        T = seconds / _SECONDS_PER_YEAR
        if T <= 0:
            continue  # expired
        rows.append((row, expiry, strike, is_call, T))

    def column(values) -> FloatArray:
        return np.array(list(values), dtype=np.float64)

    return OptionChain(
        snapshot_time=_ms_to_utc(max(row["creation_timestamp"] for row in result)),
        instrument=tuple(r[0]["instrument_name"] for r in rows),
        expiry=tuple(r[1] for r in rows),
        T=column(r[4] for r in rows),
        strike=column(r[2] for r in rows),
        is_call=np.array([r[3] for r in rows], dtype=np.bool_),
        forward=column(r[0]["underlying_price"] for r in rows),
        mark_iv=column(r[0]["mark_iv"] / 100.0 for r in rows),
        bid=column(_float_or_nan(r[0]["bid_price"]) for r in rows),
        ask=column(_float_or_nan(r[0]["ask_price"]) for r in rows),
        mark=column(r[0]["mark_price"] for r in rows),
    )


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # optional extra: pricing-core[deribit]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_book_summary(currency: str = "BTC", *, timeout: float = 10.0) -> dict:
    """GET public/get_book_summary_by_currency for all options in `currency`."""
    query = urllib.parse.urlencode({"currency": currency, "kind": "option"})
    url = f"{BASE_URL}/public/get_book_summary_by_currency?{query}"
    with urllib.request.urlopen(url, timeout=timeout, context=_ssl_context()) as resp:
        return json.load(resp)


def fetch_option_chain(currency: str = "BTC", *, timeout: float = 10.0) -> OptionChain:
    """Live snapshot of every listed option in `currency` (network call)."""
    return parse_book_summary(fetch_book_summary(currency, timeout=timeout))
