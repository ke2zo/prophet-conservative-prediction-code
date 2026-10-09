# Checks for Lemma 5.3 (closed form of Gamma) and Theorem 6.1 (rate)

- `closed_form_check.py` → `closed_form_check.txt`: the closed form Γ = c·θ(ε) − y1·g(ψ(r)) against the
  original definition at random points of F (and interval consistency); the gap formula
  1/β − 1/ρ̂ = θ(ε) + (I0 + I1 − ln(1/r)/S0)/W along the continuation branch.
- `corner_limits.py` → `corner_limits.txt`: 40-digit Euler–Lagrange points deep in the corner
  (c up to ~2e10), parametrized by y1; convergence of ε/y1², c·y1³, ω/y1², c^{2/3}·ω to the limits
  a∞, b∞, d∞, C∞ of Theorem 6.1 (ω = 1/β − 1/ρ).
- `indep_planar.py`, `indep_identity_check.py` → `indep_identity_check.txt`, `indep_corner.py` →
  `indep_corner.txt`: an independent mpmath implementation written from the note's definitions only
  (referee pass). It checks Lemma 5.3 (a)–(c) to ~1e-29 at random points, and follows the branch to
  y1 = 1e-7 (c ≈ 2e19) with Γ from its original definition: c^{2/3}·ω = 0.1581102 (limit 0.1581099),
  𝓑/y1² = −0.828146 (predicted −0.828142).

Both reuse the evaluator in `../certify` / `../certify/curve_float.npy` (run `../certify/curve_float.py`
first if the .npy is missing). Run from this folder.
- `near_one.py` → `near_one.txt`: Lemma D.1 (the formula for 1/J − 1 and r·τ − y0 > 0) at random points
  of F; the explicit path λ = 1 of the proof of Theorem 6.3 (ratio → 1 as c ↓ 1, slowly; evaluated with the
  30-digit `indep_planar.py`, since double precision loses digits for ℓ ≥ 20; also the quantities showing that
  the path lies in F at ℓ = 6); the second-order
  behaviour along the continuation branch against 1 − (1 + ln ℓ)/ℓ.
- `near_one_rates.py` → `near_one_rates.txt`: Theorem 6.4 (second order at α → 1). On the fibers r = e^{-ℓ},
  ℓ = 8, …, 40, the fiber is scanned over y1 ∈ (1e−9·y_max, y_max) (exactly one zero of Γ found), the zero is
  refined in 30 digits, and the six
  normalized rates of the theorem are recorded; the remainder of (ii) times ℓ³/ln ℓ is 0.45 → 0.42.
- `uniqueness_jacobians.py` → `uniqueness_jacobians.txt`: Remark 6.5 and Supplement (l) (uniqueness near the ends). The Jacobians
  of the two parts of the proof along the continuation branch (near c = 1 also across the window), their limits,
  and the injectivity quantity ||A^{-1}(DΥ − A)||: 0.33 at ℓ = 8; 0.50, 0.23, 0.03 at c ≈ 1.1e3, 2.4e4, 2e7.
