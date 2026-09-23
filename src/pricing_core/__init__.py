"""pricing-core: vectorised option pricing in NumPy/SciPy.

Conventions (apply to every pricer in this package):
    - Signature order ``(S, K, T, r, q, sigma, ..., *, is_call)``.
    - Continuous compounding; ``T`` in years.
    - Inputs broadcast as NumPy arrays; scalar inputs also work.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pricing-core")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
