# Checks for unrestricted predictions (manuscript Section 7)

- `obstruction_exact.py` → `obstruction_exact.txt`: horizon n = 2. Both linear programs of the envelope
  lemma (primal: adversary's decompositions; dual: gambler's families of stopping rules), for all
  valid rules and for rules that depend on M only; the rounded solutions are verified in exact rational
  arithmetic. Obstruction instance (c = 4, F uniform on {1,2,4,8,32}): adversary optimum exactly
  349/371·E[M] (attained by the explicit three-label rule), every M-measurable rule ≥ 0.951482·E[M].
  Also the natural family ("optimal policy per slab") and its failure at c = 5.
- `adversary_lp.py` → `adversary_lp.txt`: the adversary's optimum at any horizon as one LP (Bellman
  inequalities on the prefix tree); `no_prediction_ratio`. Run directly, it reproduces the exact values of
  `obstruction_exact.py` (349/371, 353/371, and the c = 5 instance) and records the difference.
- `conjecture_sweep.py` → `conjecture_sweep.txt`: 3,784 random and structured instances, c ∈ {2,3,4,5},
  n ≤ 5.
- `conjecture_search.py` → `conjecture_search.txt`: local search minimizing the no-prediction ratio and the
  adversary's optimum.
- `band_finite_n.py` → `band_finite_n.txt`: finite-horizon band constants ρ_n(c) by local search (upper
  bounds only; for c = 3 superseded by `rhon_certify.py`).
- `finite_n_gap.py` → `finite_n_gap.txt`: local search for instances with atoms {0, 1, a, c·a} on which
  path-dependent rules push the gambler below ρ_n(c).
- `finite_n_search.py` → `finite_n_search_7.txt`, `finite_n_search_8.txt`: the local searches that produced the
  instances for n = 7, 8 (atoms {0, 1, a, 3a}; one n = 8 linear program takes about 24 s).
- `finite_n_exact.py` → `finite_n_exact.txt`: for those instances (c = 3, n = 3..8, hardcoded with the
  atoms and probabilities of the manuscript's table and printed in the output), upper bounds on the
  adversary's optimum in exact rational arithmetic, rounded up to six decimals (Proposition "Unrestricted rules
  at horizons 3 to 8"(b)), compared with the certified lower bounds on ρ_n(3); also the no-prediction values
  and the best rules depending on M only.

- `rhon_model.py`, `rhon_certify.py` → `rhon_certify.txt`: rigorous bounds on ρ_n(3) for n = 3..6
  (Proposition "Unrestricted rules at horizons 3 to 8"(a)): ρ_n(3) ∈ [0.873303, 0.873304], [0.858415, 0.858416],
  [0.853851, 0.853852], [0.847779, 0.847780]. Jensen averaging between the thresholds reduces to n cases (j = number of
  thresholds below 1) with at most n coordinates each; interval branch and bound with the natural extension,
  the mean-value form (interval forward-mode AD) and a Lagrangian relaxation for boxes that cross a
  constraint (it saves much work near the minimizers, which lie on the boundary between two cases).
  Self-tests first (model vs dynamic program, AD vs central differences, containment of value and gradient
  enclosures); then the certification (about two minutes), exact upper bounds from explicit laws, and
  controls: the bounds 0.8733036, 0.8584161, 0.8538516, 0.8477798 are refuted. `python3 rhon_certify.py 5
  0.853851` runs a single bound (`--nolag` disables the Lagrangian test).
  Horizons 7 and 8 (`rhon_certify_78.txt`): `python3 rhon_certify.py 78 4` (parallel driver, 4 processes;
  44.5 and 245 million boxes, about 10 and 75 minutes on a loaded laptop, 4 and 28 minutes on an idle one) certifies ρ_7(3) ∈ [0.845355, 0.845356]
  and ρ_8(3) ∈ [0.842265, 0.842266]; `python3 rhon_certify.py 78control` then refutes 0.8453560 and 0.8422660
  (sequentially, on the case containing the numerical minimizer). The n = 8 run needs several GB of memory.
  The log contains numpy warnings "invalid value encountered in multiply": near the corner W_{n-1} = c the
  interval arithmetic produces NaN (0·∞); every accept or discard test compares interval endpoints and fails
  on NaN, so such boxes are subdivided, and the certificate is unaffected.
- `rho3_certify.py` → `rho3_certify.txt`: an earlier, separate implementation for n = 3 only (same interval
  library and reduction; own coordinates in the last case, no Lagrangian test); certifies
  0.8733 ≤ ρ_3(3) ≤ 0.873305. `python3 rho3_certify.py --sanity`
  shows that the bound 0.87331 is refuted.

Run from this folder. `obstruction_exact.py` takes seconds and `finite_n_exact.py` under a minute,
`rhon_certify.py` about five minutes with the controls (n = 3..6); the searches take 10–30 minutes (n ≤ 6)
and up to two hours (n = 7, 8). Uses numpy, scipy (HiGHS) and mpmath (through ../certify/ivec.py).
