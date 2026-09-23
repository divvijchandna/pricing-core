# pricing-core

Vectorised option pricing library in NumPy/SciPy. It is the shared pricing layer for a
derivatives risk engine and an options market-making simulator.

> Status: pre-release scaffold. Pricers are added week by week; this README lists
> only what exists.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
ruff check .                # lint
ruff format --check .       # formatting
pytest -m "not slow"        # fast tests
pytest                      # full suite, including slow statistical tests
```

Tests run with all warnings turned into errors, so a silent divide-by-zero or overflow
fails the suite.

## Conventions

- Pricer signature: `(S, K, T, r, q, sigma, ..., *, is_call)`. Continuous compounding,
  `T` in years.
- All pricers broadcast over NumPy arrays; scalar inputs also work.
- Greek units are stated in every docstring: vega per 1.00 of vol, theta per year.
