# pricing-core: design notes

Why each design decision was made: what was chosen, what else was considered, and the
reference behind it. Entries tagged `DRAFT` have not yet been rewritten in my own words.

## Entry format

```
## <Decision>: DRAFT
Chose: ...
Alternatives considered: ...
Why: ...
Interview answer (2 sentences): ...
Reference: <paper / book / section>
```

---

## Legacy audit: DRAFT
Chose: before porting, run the new `bs_price` / `black76_price` test suite against
the old Streamlit formula (`Black-Scholes-v1.0/black_scholes.py`, class
`BlackScholes`) through a temporary pytest plugin that swapped the implementation
without changing the tests. Scope: q = 0 only (the legacy code has no dividend
yield), so the shared grid was cut to its q = 0 half (240 points).
Black-76 was reached with q = 0 via S = F e^(-rT), which gives the same prepaid
forward S e^(-qT) = F e^(-rT) as the q = r route.
Alternatives considered: running at q != 0 and letting the tests fail (buries real
findings under repeated "no q" failures); reading the code and listing defects by
inspection (unverified, and it got one prediction wrong, see below).
Why: the same tests that accept the new pricer define what "correct" means; running
them on the old code turns a rewrite into a documented validation with evidence.
Interview answer (2 sentences): I ran my new pricer's test suite against my old
calculator first, which showed the core formula was correct but returned NaN at the
forward when T = 0 or sigma = 0, had no dividend yield, and warned on every limit
case. That is outcomes analysis in the model-validation sense: evidence from tests,
not inspection.
Reference: Federal Reserve / OCC SR 11-7, "Supervisory Guidance on Model Risk
Management" (2011), outcomes analysis; Hull, Options, Futures, and Other
Derivatives, "Black-Scholes-Merton model" chapter.

Result: 13 passed, 4 failed, 6 skipped (23 cases).

| # | Finding | Evidence | Why | Fix in pricing-core |
|---|---|---|---|---|
| 1 | NaN at the forward in the limits: T = 0 with S = K, and sigma = 0 with S = K e^(-rT) | legacy call/put = nan; correct value 0 | d1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma sqrt(T)) is 0/0 when the numerator is exactly 0; NaN propagates through N(d1) | Branch explicitly where sigma sqrt(T) = 0: return the discounted forward intrinsic max(S e^(-qT) - K e^(-rT), 0), with a safe denominator so the formula branch never divides by 0 |
| 2 | Divide-by-zero warning on every limit case (T = 0 or sigma = 0) | 4 tests fail: `RuntimeWarning: divide by zero`, black_scholes.py:16 | Away from the forward d1 = +/-inf and N(+/-inf) gives the right limit by accident, but only after dividing by zero | Same masking as #1; the suite runs with warnings as errors so this cannot regress silently |
| 3 | No dividend yield q | Cannot run q != 0 cases; deep ITM/OTM test skipped (its expected value uses q) | Formula hard-codes q = 0: S N(d1) instead of S e^(-qT) N(d1), and r instead of r - q in d1 | `q` is a required argument: `bs_price(S, K, T, r, q, sigma, *, is_call)` |
| 4 | No Black-76 | Not provided by the app; mathematically reachable via S = F e^(-rT), q = 0, and those tests pass | Scope, not a formula error | `black76_price(F, K, T, r, sigma, *, is_call)` delegates to `bs_price` with q = r |
| 5 | Class API with separate `call_option_price()` / `put_option_price()` methods; no `is_call` | `is_call` tests skipped (they would test adapter glue, not legacy code) | Object holds one set of inputs; a mixed call/put book needs two calls and a select | Pure function with keyword-only, required `is_call` (bool or bool array) |
| 6 | `requirements.txt` is UTF-16 LE with CRLF line endings, and pins a full `pip freeze` (50+ transitive packages) | `file` reports UTF-16; not a test failure | Likely written by a PowerShell `pip freeze >` redirect | options-pricing-dashboard lists direct dependencies only, UTF-8 |

What passed (the core formula is correct for q = 0): put-call parity, no-arbitrage
bounds, Black-76 via S = F e^(-rT) against an independent formula, all three Hull
known answers, strict monotonicity in sigma, scalar input returns a float,
(2,1) x (3,) broadcasting.

Predictions from reading the code that the tests corrected:
- Predicted "scalar only, not vectorised". Wrong: the methods are plain NumPy
  expressions, so arrays broadcast and the broadcasting test passed. Whether the old
  heatmap uses this or loops is a question for the dashboard refactor.
- Predicted "divides by zero in the limits" as the failure. Incomplete: the limits
  are right by accident (d1 -> +/-inf) except exactly at the forward, where they are
  NaN. The precise finding is #1.

## Verifying Black-76 without a self-referential test: DRAFT
Chose: two independent references for `black76_price`: (1) Black's formula written
out directly in the test, e^(-rT) [F N(d1) - K N(d2)], compared over the full grid
at rtol 1e-10; (2) Hull's worked crude-oil futures put (F = K = 20, T = 4/12,
r = 9%, sigma = 25%, printed 1.12) at +/- 0.005. Hull's BSM call/put (4.76 / 0.81)
anchor `bs_price` the same way.
Alternatives considered: PLAN.md's "bs_price(F, K, T, r, q=r, sigma) equals
Black-76", rejected because `black76_price` is implemented as exactly that call,
so the test compares the function with itself and cannot fail.
Why: the independent formula is precise and covers the whole grid but shares any
misunderstanding I have; the published value is coarse (2 d.p., one point) but is
external. Each covers the other's blind spot. Planted bugs confirmed it: a q = 0
wrapper fails both; a spot-intrinsic sigma = 0 limit fails the limit tests.
Interview answer (2 sentences): I test against an independent re-implementation
for precision across the grid and against published textbook values to catch errors
I would repeat in both, the same way crypto code is checked with differential tests
and NIST known-answer vectors. A test that compares a wrapper with the function it
wraps cannot fail, so it proves nothing.
Reference: Black (1976), "The pricing of commodity contracts", J. Financial
Economics 3; Hull, "Futures options and Black's model" chapter.

## Test tolerances: relative, plus absolute near zero: DRAFT
Chose: rtol 1e-12 for identities and limits; rtol 1e-10 for the Black-76 comparison
(the two formulas discount in a different order); an atol (1e-10 to 1e-12) wherever
the expected value can be ~0 (parity at the forward, OTM limits, zero intrinsic);
+/- 0.005 for Hull values (half the last printed decimal). Monotonicity tested on
sigma in [0.05, 1.0] only.
Alternatives considered: rtol only, as the CLAUDE.md convention says (meaningless
when the target is 0); a single loose tolerance everywhere (would hide real errors).
Why: each price is a handful of floating-point operations at ~1e-16 each, so 1e-12
has ample headroom yet catches any formula error. Below sigma ~ 0.05 at deep
moneyness the time value (~phi(d1) with |d1| ~ 9, i.e. ~1e-19) is below double
resolution next to a price of ~20, so the price is increasing mathematically but
flat numerically.
Interview answer (2 sentences): Tolerances follow from where the error comes from:
floating-point roundoff for closed-form identities, rounding of the published value
for textbook checks, and an absolute floor wherever the true value is zero because
relative error is undefined there. I also keep monotonicity tests away from regions
where doubles cannot resolve the change, rather than loosening the tolerance.
Reference: Higham, Accuracy and Stability of Numerical Algorithms (2nd ed.),
ch. 1-2 (floating-point error); numpy.testing.assert_allclose documentation.

## Black-76 as bs_price with q = r: DRAFT
Chose: `black76_price(F, K, T, r, sigma, *, is_call)` is one line,
`bs_price(F, K, T, r, q=r, sigma=sigma, is_call=is_call)`.
Alternatives considered: a separate Black-76 formula (two implementations to keep
consistent); a generalised "cost of carry b" pricer (b = r - q for stocks, 0 for
futures, r - r_f for FX), which is more general than anything Week 1 needs.
Why: Black-Scholes depends on the underlying only through its prepaid forward,
S e^(-qT). For a futures price that is F e^(-rT), which S = F, q = r produces, and
the drift r - q becomes 0. The PDE says the same: the (r - q) S V_S carry term
vanishes because a futures hedge costs nothing to hold. One formula means one place
for bugs, and every fix reaches both pricers.
Interview answer (2 sentences): Black-Scholes is really a formula in the prepaid
forward, so Black-76 is the same formula with the stock's prepaid forward S e^(-qT)
replaced by the future's F e^(-rT), which is exactly q = r. I implement it as a
one-line delegation and test it against Black's formula written out independently
plus Hull's worked example, because a test comparing the wrapper with itself could
never fail.
Reference: Black (1976), J. Financial Economics 3; Hull, "Futures options and
Black's model" chapter; Haug, The Complete Guide to Option Pricing Formulas, ch. 1
(generalised Black-Scholes).

## Zero total volatility: explicit branch for the price, true limits for d1/d2: DRAFT
Chose: where sigma sqrt(T) = 0, `bs_price` returns the discounted forward intrinsic
value max(S e^(-qT) - K e^(-rT), 0) (puts mirrored), which is also the lower
no-arbitrage bound; at T = 0 this is ordinary intrinsic value. `d1_d2` separately
returns the true limits: +inf if P_S > P_K, -inf if P_S < P_K, and 0 at the forward
(d1 = -d2 = sigma sqrt(T)/2 -> 0).
Alternatives considered: letting the division produce +/-inf and trusting
N(+/-inf) (the legacy approach: right by accident except NaN at the forward, and a
warning every time); returning max(S - K, 0) (wrong whenever r or q is non-zero).
Why: the limit is decided by moneyness against the forward, P_S vs P_K, not S vs K.
The explicit branch gives an exact, never-negative price; using the d1 = d2 = 0
limit instead would give 0.5 (P_S - P_K), which rounding can make slightly negative.
d1_d2 still carries the limits because the Greeks use d1/d2 directly: delta then
becomes the correct step function at sigma = 0, with 0.5 at the forward. A planted
bug (d1/d2 = NaN at the forward) failed only the d1_d2 test, not the price tests,
which is the explicit branch doing its job.
Interview answer (2 sentences): As sigma sqrt(T) goes to 0, d1 goes to plus or
minus infinity depending on whether the prepaid forward is above or below the
discounted strike, so the price tends to the discounted forward intrinsic value,
except exactly at the forward where d1 is 0/0. I branch explicitly to that limit
instead of relying on infinities, which removes the NaN my old calculator returned
there.
Reference: Hull, "Black-Scholes-Merton model" chapter (properties of the formula as
sigma -> 0 and T -> 0); legacy audit finding #1 above.

## Invalid inputs return NaN, never raise: DRAFT
Chose: S <= 0, K <= 0, T < 0, sigma < 0, or any NaN/inf input gives NaN in that
position of the output, with no exception and no warning. Other positions in the
same array are unaffected. `is_call` of the wrong type still raises TypeError.
Alternatives considered: raise ValueError on any invalid element (one bad scenario
aborts a 10,000-scenario revaluation); leave it unchecked (negative sigma fails the
`v > 0` test and is silently priced as zero vol).
Why: invalid values can come from my own pipeline, not just typos: a "vol -25
points" stress on a 20% vol gives sigma = -5%, and a Newton step can overshoot below
0. Silently pricing that as sigma = 0 would make a stress report look clean. NaN
cannot pass for a price and propagates into any sum, so it surfaces. A wrong type
for is_call is a programming error, so it stops the program instead. Matches the
IV-solver rule in CLAUDE.md (NaN outside bounds, never raise inside a vectorised
call).
Interview answer (2 sentences): In a vectorised pricer, an invalid input such as a
negative vol from an oversized stress shock returns NaN for that element only, so
the run completes and the bad scenario is visible rather than silently priced as
zero vol. I distinguish that from programming errors like a non-boolean is_call,
which raise immediately.
Reference: IEEE 754 NaN propagation; NumPy documentation, "Floating point error
handling" (numpy.errstate).

## is_call required, keyword-only, strictly boolean; scalars return float: DRAFT
Chose: `is_call` has no default, must be passed by keyword, and must be a bool or
bool array (`is_call=1`, `"C"`, `"P"` raise TypeError). A scalar result is returned
as a built-in Python float; array results stay float64 arrays.
Alternatives considered: `is_call=True` default; accepting any truthy value;
returning 0-d arrays or np.float64 for scalars.
Why: a put priced as a call is a silent, costly error, and `"P"` is truthy in
Python. Keyword-only makes every call site read `is_call=False` explicitly. A plain
float prints and serialises cleanly (NumPy 2 prints np.float64(4.76)); the cost is
that the caller's own arithmetic follows Python rules (1/0.0 raises).
Interview answer (2 sentences): I made the call/put flag mandatory, keyword-only
and strictly boolean, because in Python the string "P" is truthy and would silently
price a put as a call. Scalar inputs return a plain float and arrays return arrays,
so the same function serves a dashboard and a scenario engine.
Reference: Python docs, Built-in Types, "Truth Value Testing"; NumPy 2.0 release notes
(scalar representation, NEP 51).

## Normal CDF: scipy.special.ndtr, and N(-x) computed directly: DRAFT
Chose: `ndtr` for N(x); the put formula evaluates N(-d1), N(-d2) directly rather
than 1 - N(d1), 1 - N(d2).
Alternatives considered: `scipy.stats.norm.cdf` (same values, extra Python-level
overhead per call); `1 - N(x)`.
Why: for large positive x, N(x) rounds to 1.0 and 1 - N(x) becomes 0, losing a
small but real value. N(-x) is computed accurately in the tail, which keeps deep
OTM puts positive and correct.
Interview answer (2 sentences): I evaluate N(-x) directly because 1 - N(x)
suffers catastrophic cancellation in the tails, where N(x) is within an ulp of 1. I
call scipy.special.ndtr rather than scipy.stats.norm.cdf because it is the same
function without the distribution-object overhead, which matters when revaluing
thousands of scenarios.
Reference: Higham, Accuracy and Stability of Numerical Algorithms, ch. 1
(cancellation); SciPy documentation, scipy.special.ndtr.

## Element-wise special cases: np.where with safe placeholders: DRAFT
Chose: compute both branches for the whole array and select with `np.where`; before
any division, log or square root, replace the positions that will be discarded with
harmless valid values (`v_safe` = 1 where sigma sqrt(T) = 0; `_sanitise` puts
S = K = T = sigma = 1, r = q = 0 where inputs are invalid), then overwrite those
positions with the limit value or NaN.
Alternatives considered: a Python `if` per element (loses vectorisation);
`np.errstate` to silence the warnings (hides the symptom and still computes 0/0);
masked arrays (heavier, and every later operation must respect the mask).
Why: `np.where(cond, A, B)` evaluates all of A and all of B before selecting, so
the unused branch still runs. The same mask builds the placeholders and makes the
final selection, so a placeholder result can never reach the output.
Interview answer (2 sentences): NumPy's where evaluates both branches on every
element, so I substitute safe placeholder values wherever a branch will be discarded,
which keeps the computation warning-free without silencing warnings globally. The
same mask chooses the output, so the placeholder results are always overwritten by
the correct limit or by NaN.
Reference: NumPy documentation, numpy.where (both inputs are evaluated) and
numpy.errstate.

## Greeks API: one function per Greek, is_call only where it matters: DRAFT
Chose: `delta`, `gamma`, `vega`, `theta`, `rho`, `vanna`, `volga`, each with the
`bs_price` signature. `delta`, `theta` and `rho` take a required keyword-only
`is_call`; `gamma`, `vega`, `vanna` and `volga` take none. Internal units: per 1.00
of the bumped quantity, theta per year with theta = dV/dt = -dV/dT; reporting
conversions live in `conventions` (`per_day` = /365 calendar days, `per_vol_point`
= /100).
Alternatives considered: one function returning all Greeks (saves recomputing d1/d2
seven times, but no caller needs that yet); a uniform signature where every Greek
takes is_call (a flag that changes nothing misleads the reader).
Why: each Greek is a short formula with its own docstring and units and is tested
on its own. Gamma, vega, vanna and volga are equal for calls and puts because
C - P = S e^(-qT) - K e^(-rT) is linear in S and independent of sigma, so any
derivative that is second order in S or involves sigma cancels in C - P.
Interview answer (2 sentences): I expose one function per Greek in internal units,
per year for theta and per 1.00 of vol for vega, with separate helpers for desk
units such as per day and per vol point. The call/put flag appears only on delta,
theta and rho, because put-call parity makes the others identical for calls and
puts.
Reference: Hull, "The Greek letters" chapter; put-call parity (entry above).

## Black-76 Greeks: reuse the BS functions, except rho: DRAFT
Chose: Black-76 Greeks come from the BS functions with S = F and q = r (delta is then
dV/dF, theta holds F fixed), except rho, which has its own `black76_rho` = -T V.
Alternatives considered: a full set of black76_* functions (seven wrappers, six of
them one-line delegations).
Why: with q tied to r, `rho(F, K, T, r, r, sigma)` differentiates only the explicit
r while holding q fixed, which is the wrong partial derivative. In Black-76 r only
discounts, V = e^(-rT) [F N(d1) - K N(d2)], so with F fixed dV/dr = -T V. Tested
against -T V exactly and against a bumped black76_price.
Interview answer (2 sentences): Black-76 shares every Greek with Black-Scholes under
S = F, q = r, except rho, because tying q to r means the rate also moves the carry
term that the spot rho holds fixed. With the futures price held fixed, the rate only
discounts the payoff, so Black-76 rho is simply minus T times the price.
Reference: Black (1976); Hull, "Futures options and Black's model" chapter.

## Greeks at zero total volatility: DRAFT
Chose: where sigma sqrt(T) = 0 (zero vol, or expiry day), away from the forward each
Greek takes its exact limit: delta and rho are step functions, theta is the carry
q P_S - r P_K or 0, and gamma, vega, vanna, volga are 0. At the forward (P_S = P_K)
delta and rho take the midpoint (d1 = d2 = 0), vega its one-sided limit
P_S n(0) sqrt(T), and gamma, vanna, volga and theta return NaN.
Alternatives considered: NaN for every Greek at zero vol (an expired deep ITM
position would make book-level gamma totals NaN); +inf for gamma at the forward
(true, but inf - inf in later aggregation gives NaN anyway, less explicitly).
Why: the value max(P_S - P_K, 0) has a kink at the forward. Away from it, the
Greeks are well-defined and an expiring book needs them (delta 1 or 0 for hedging,
gamma 0). At the kink, gamma is a Dirac spike and at-the-money theta tends to -inf
at expiry, so no finite number is honest. The midpoint for delta is the average of
the left and right slopes, the usual convention.
Interview answer (2 sentences): On expiry day an option's value becomes
max(forward - strike, 0), so away from the strike the Greeks have clean limits such
as delta of exactly 1 or 0 and zero gamma, which keeps book-level aggregates usable.
Exactly at the strike the kink makes gamma infinite and theta unbounded, so those
return NaN while delta takes the midpoint, one half.
Reference: Hull, "The Greek letters" chapter (behaviour of gamma and theta as
T -> 0).

## Bump sizes and tolerances: truncation vs roundoff: DRAFT
Chose: kept the CLAUDE.md bumps (spot h = 1e-4 S; vol, T and r h = 1e-4, central
differences) and set each Greek's tolerance from data: rtol = 4 x the worst relative
error measured on the 480-point grid (1e-5 to 6e-4), plus an atol for values near
zero (1e-9 to 2e-5).
Alternatives considered: PLAN.md's rtol ~1e-6 (fails: measured errors reach 1e-4 at
sigma = 0.05); bumps scaled to the natural length, h_S = 1e-3 S sigma sqrt(T)
(about 1e-6 everywhere, but changes the convention and needs extra roundoff care at
small sigma); restricting the test to sigma sqrt(T) >= 0.1 (leaves low vol
untested).
Why: central-difference error = truncation h^2 |V'''| / 6 + roundoff eps |V| / h
(second derivatives: h^2 |V''''| / 12 + 4 eps |V| / h^2, eps = 2.2e-16). The price
curves over a spot range of about S sigma sqrt(T), so the relative truncation error
is about (h / (S sigma sqrt(T)))^2: 1e-8 at sigma sqrt(T) = 0.5 but 1e-5 at 0.025.
Every worst case was at sigma = 0.05. At the zero crossings of vanna (d2 = 0) and
volga (d1 d2 = 0) the bumped value is pure roundoff, 4 eps V / h^2 ~ 1e-5 for volga,
so only an absolute floor can work there. The resulting tolerances still sit ~1000x
below the O(1) errors that real formula bugs produce, and the exact identities
check the formulas far more tightly.
Interview answer (2 sentences): A finite-difference Greek has truncation error that
grows with the bump relative to how sharply the price curves, about S sigma sqrt(T)
in spot, and roundoff error that grows as the bump shrinks, so short-dated low-vol
options need either smaller bumps or looser tolerances. I measured the errors across
the grid and set tolerances from the data, with absolute floors where Greeks such as
vanna and volga cross zero.
Reference: Higham, Accuracy and Stability of Numerical Algorithms, ch. 1;
Glasserman, Monte Carlo Methods in Financial Engineering, ch. 7 (finite-difference
sensitivities, bias vs variance of the bump).

## Testing Greeks with both bumps and exact identities: DRAFT
Chose: two independent kinds of test. (1) Each Greek against a central difference
of bs_price (independent of my derivations, loose tolerance). (2) Exact identities
at ~1e-10: Delta_C - Delta_P = e^(-qT), Theta_C - Theta_P = q P_S - r P_K,
Rho_C - Rho_P = K T e^(-rT), the PDE Theta + (r - q) S Delta + 1/2 sigma^2 S^2
Gamma - r V = 0, and vega = S^2 sigma T Gamma. Plus signs (gamma, vega >= 0) and a
units check, per_day(theta) ~ V(T - 1 day) - V(T).
Alternatives considered: bumps only (cannot detect errors smaller than their
tolerance); identities only (blind to some bugs).
Why: each kind has blind spots the other covers. A planted bug using N(d1) instead
of N(d2) in rho still satisfies Rho_C - Rho_P = K T e^(-rT), because
N(x) + N(-x) = 1 for any x, so only the bumped test caught it. Conversely the PDE
and parity identities check delta, gamma, theta and the price against each other at
1e-10, far tighter than any bump. Planted bugs in delta (missing e^(-qT)) and theta
(sign of the q term) failed five or six tests each.
Interview answer (2 sentences): I test analytic Greeks two ways: against finite
differences of the price, which are independent but loose, and against exact
identities like put-call relations and the Black-Scholes PDE, which are tight but
can be satisfied by a wrong formula. A deliberately wrong rho passed the parity
identity and was caught only by the bump test, which is why I keep both.
Reference: Hull, "The Greek letters" chapter (relationship between delta, theta and
gamma); Wilmott, Paul Wilmott Introduces Quantitative Finance, ch. on the
Black-Scholes PDE.
