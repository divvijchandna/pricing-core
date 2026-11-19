"""One vanilla European option position: contract terms and signed size.

    quantity   signed number of options (negative = short)
    K          strike, > 0
    T          time to expiry in years, >= 0 (T = 0 prices at intrinsic)
    is_call    bool, keyword-only as in the pricers: (..., *, is_call)

Value of the position = quantity * option value; the Book sums these. There is no
contract multiplier: quantity is in units of the underlying's option.

A position is static booking data, so bad terms (K <= 0, T < 0, NaN or inf, a
non-bool is_call) raise at construction. This differs from the pricers and
MarketState, which return NaN for bad market inputs: a bad scenario must not stop
a vectorised run, but a mis-booked contract should fail once, loudly.
"""

import math
from dataclasses import KW_ONLY, dataclass

import numpy as np


@dataclass(frozen=True)
class Position:
    quantity: float
    K: float
    T: float
    _: KW_ONLY
    is_call: bool

    def __post_init__(self) -> None:
        # Strict like as_bool_array: 1 or "C" would otherwise be silently truthy.
        if not isinstance(self.is_call, bool | np.bool_):
            raise TypeError(f"is_call must be a bool, got {type(self.is_call)}")
        quantity, K, T = float(self.quantity), float(self.K), float(self.T)
        if not all(math.isfinite(x) for x in (quantity, K, T)):
            raise ValueError("quantity, K and T must be finite")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}")
        # Frozen dataclass: normalise the stored types through object.__setattr__.
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "is_call", bool(self.is_call))
