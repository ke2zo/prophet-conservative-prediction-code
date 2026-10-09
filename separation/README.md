# Checks for the separation theorem (manuscript Section 6.3, Theorem 6.6)

- `gap_check.py` → `gap_check.txt`: symbolic identities (A), (B) and the decomposition of 𝔅 (sympy);
  the two-point closed forms of E, L0, L1 against quadrature of their definitions (mpmath, 30 digits);
  the numerical constants of the proof with interval arithmetic (mpmath.iv); a brute-force grid over
  (c, S0) of the minimum over u* of the two-point ratio of the policy used in each regime, against the
  bound τ_c + Δ_c; a direct Monte Carlo simulation of the Poisson model with level marks for the
  wait-then-grab policy, against its closed form and against A(s).

Run from this folder: `python3 gap_check.py` (about 20 s).
