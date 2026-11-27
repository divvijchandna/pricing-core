# pricing-core

[![CI](https://github.com/divvijchandna/pricing-core/actions/workflows/ci.yml/badge.svg)](https://github.com/divvijchandna/pricing-core/actions/workflows/ci.yml)

Vectorised option pricing and revaluation library in NumPy/SciPy. It is the shared
pricing layer for my other projects:

- [options-market-maker](https://github.com/divvijchandna/options-market-maker):
  quoting simulator for Deribit BTC options (coin-settled pricing, inventory skew)
- [options-pricing-dashboard](https://github.com/divvijchandna/options-pricing-dashboard):
  Streamlit calculator and heatmaps, with no pricing formulas of its own
- a derivatives risk engine (PnL attribution, VaR/ES and backtests, model
  validation), in development

Every pricer broadcasts over NumPy arrays, because a risk engine revalues a book
across thousands of scenarios and a Python loop per option would dominate the run.

## What is built

| Module | Contents |
|---|---|
| `analytic/black_scholes.py` | Black-Scholes-Merton with continuous yield `q`; Black-76 as `q = r`; exact limits at zero vol and at expiry |
| `analytic/greeks.py` | Delta, gamma, vega, theta, rho, vanna, volga; Black-76 rho |
| `analytic/implied_vol.py` | Vectorised implied vol: safeguarded Newton with a bisection fallback |
| `market/interface.py` | `MarketState`: many market states in one object; ATM vol term structure |
| `book/` | `Position` and `Book`: revalue a book and its Greeks across all states in one call |
| `market/adapters/deribit.py` | Deribit public REST API to an `OptionChain`, units converted at the boundary |
| `conventions.py` | Reporting units: theta per calendar day, Greeks per vol point |

Some measured results (Apple M4 Pro, current `main`):

- **Implied vol stress test.** 50,000 random options (moneyness e^±1, 4 days to 3
  years, vol 2% to 200%) solved in about 0.34 s in one call. Every option with
  usable vega (27,337 of the 50,000) is recovered; for the rest the price carries
  almost no information about vol, and the solver returns either NaN or a vol that
  reprices the input exactly, never a wrong answer.
- **Book revaluation.** 50 positions across 10,000 market states in about 20 ms.

## Example

```python
from pricing_core.book.book import Book
from pricing_core.book.position import Position
from pricing_core.market.interface import MarketState

book = Book([
    Position(10, 1.10, 1 / 12, is_call=True),   # long 10 1M 1.10 calls
    Position(-5, 1.05, 0.25, is_call=False),    # short 5 3M 1.05 puts
])
today = MarketState(
    spot=1.08, r=0.04, q=0.02,
    tenors=[1 / 12, 0.25, 1.0], atm_vol=[0.10, 0.08, 0.09],
)
book.revalue(today)              # array([0.03074166])
book.greeks(today).vega          # array([0.31216348]), per 1.00 of vol
book.revalue(today, horizon=1 / 365)   # value one day later, same market
```

Pass arrays for `spot`, `r`, `q` or `atm_vol` and the same calls return one value
per state.

## Design choices

- **NaN for bad market data, exceptions for bad code.** An invalid input (negative
  vol, a price outside no-arbitrage bounds) gives NaN in that position, never an
  exception inside a vectorised call, so one bad scenario cannot stop a run or pass
  for a real number. A mis-booked contract or mismatched array shapes raise at
  construction.
- **Edge cases are explicit.** At zero total volatility the price is the discounted
  forward intrinsic value, and each Greek takes its exact limit; values that are
  genuinely undefined (gamma exactly at the forward on expiry) are NaN.
- **Implied vol is solved where it is well conditioned.** In-the-money prices are
  converted to out-of-the-money ones by put-call parity first, since the intrinsic
  value adds rounding error far larger than the time value that determines vol.
  Newton starts at the inflection point of price in vol and is kept inside a
  shrinking bracket, so it always converges.
- **Vol term structure interpolates total variance σ²T, not vol.** Linear total
  variance means constant forward variance between tenors, so the interpolation
  adds no calendar arbitrage.
- **Units are stated, not assumed.** Vega is per 1.00 of vol and theta per year
  internally; conversion to per vol point and per day happens only for reporting.
- **Runtime dependencies are NumPy and SciPy only.** Data adapters are optional
  extras (`pip install "pricing-core[deribit]"`).

## Testing

219 tests, run in CI on Python 3.11 and 3.12 together with `ruff`. They check
prices against put-call parity, no-arbitrage bounds, the zero-vol and expiry limits
and textbook values; analytic Greeks against central finite differences; implied
vol by round trip; book values against a position-by-position loop; and the
Deribit adapter against a recorded API response (no live calls in CI). All warnings
are errors, so a silent divide-by-zero or overflow fails the suite.

## Roadmap

- **In progress:** scenario generation (spot and vol shocks), synthetic market data,
  Monte Carlo (exact GBM stepping, antithetic variates, standard errors) and
  closed-form barrier options.
- **Planned:** Heston (Fourier pricing and Monte Carlo), Crank-Nicolson PDE, SVI vol
  surfaces with arbitrage checks, a Bloomberg adapter, and a C++ port of the PDE
  solver and Monte Carlo path generator via pybind11.

## Install

Requires Python 3.11+.

```bash
pip install "pricing-core @ git+https://github.com/divvijchandna/pricing-core"
```

For development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -m "not slow"        # fast tests
pytest                      # full suite, including slow statistical tests
```

## Conventions

- Pricer signature: `(S, K, T, r, q, sigma, ..., *, is_call)`. Continuous
  compounding, `T` in years.
- `is_call` is a bool or bool array; other types are rejected.
- Scalar inputs return a float; array inputs return an array.

## References

Hull, *Options, Futures, and Other Derivatives*; Haug, *The Complete Guide to Option
Pricing Formulas* (vanna, volga); Gatheral, *The Volatility Surface* (total
variance); Jäckel (2015), "Let's Be Rational" (implied vol start point); Brenner and
Subrahmanyam (1988) (implied vol approximation).
