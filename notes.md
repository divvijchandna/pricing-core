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
