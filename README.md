# Computations for "Prophet Inequalities for I.I.D. Values with a Conservative Prediction: The Bounded-Support Kertz Constant"

This folder contains the code, the certificates and the logs of the computer-assisted parts of the manuscript,
and the numerical checks of the Supplement (the Supplement accompanies the journal submission). The computer-assisted
results are Theorem D (Theorem 5.5), Table 2, and Propositions 7.6 and 7.8. Everything else in the manuscript is
proved analytically; the checks of the Supplement (rows "Supplement …" below) are not used in any proof.

## Requirements

Python 3 with the packages in `requirements.txt` (the versions used for the saved outputs):

```
python3 -m pip install -r requirements.txt
```

The interval arithmetic is `certify/ivec.py`: IEEE double precision, every operation widened outward by one unit
in the last place, logarithms, exponentials and decimal constants from mpmath's interval arithmetic at 80 bits, rounded
outward to double precision and widened by one more unit in the last place (this absorbs any error of the 80-bit
endpoints up to a relative 2^-54, so these enclosures do not rely on mpmath's directed rounding being exact). `certify/tails.py`
checks its inequalities directly in mpmath's interval arithmetic at 80 bits, with relative margins above 1e-14. No
other library enters the certificates. A second implementation in Arb ball arithmetic (`certify/arb_eval.py`,
`certify/arb_check.py`; it needs `python-flint`, which is not in `requirements.txt`) repeats the tests of the
certification as a cross-check (the exclusion cover on a random sample); no proof uses it. SciPy is used by floating-point helpers, several checks and the linear programs of Section 7, not by the
certification.

## Reproducing

```
python3 reproduce.py                 # tier quick: 30 steps, about 40 minutes on one core
python3 reproduce.py --tier full     # adds 8 long steps: about 2.5 hours with 4 processes
python3 reproduce.py --list          # all steps; --only NAME ... runs selected steps
```

`reproduce.py` copies this folder to `_reproduce/`, runs every step there, and compares each regenerated output with
the saved one after removing run times; it prints PASS, DIFF (with the first differing line) or FAIL for every step
and writes `_reproduce/reproduce_report.txt`. The saved outputs are never overwritten. From a fresh copy of this
folder all 30 steps of the quick tier reproduce the saved outputs exactly (Python 3.14, macOS, the versions of
`requirements.txt`); the longest step is `asymptotics/corner_limits.py` (40-digit arithmetic, about 18 minutes).
The long steps of the tier `full` (the float curves, the four certification runs of Section 5.2 built from scratch,
the gradient check against 30 digits, and n = 7, 8) also reproduce every saved log, certificate and table, in about
2.5 hours with 4 processes. Two kinds of lines are ignored in these comparisons: the lines about building or re-validating
a tube (a run that finds a cached tube re-validates it instead of building it) and the width of the point enclosures in
`point_values.txt`, which depends on the floating-point starting point. The tier `search` adds the
local searches that found the instances of Section 7 (optional, hours, compared for information only: the searches
of `unrestricted/finite_n_search.py` for n = 7, 8 were stopped by hand and are not re-run).

Memory: the certification runs and the run for n = 7, 8 use `--procs` processes (default 4); the run for n = 8 needs
several GB. Run the long steps one at a time on a laptop.

## Map from the manuscript to the scripts

| manuscript | folder | scripts → outputs |
|---|---|---|
| Theorem 5.5 (Theorem D): tube, cover, tails | `certify/` | `run_certify.py` (with `tube.py`, `cover.py`, `cheap.py`, `tails.py`, `planar_iv.py`, `planar_box.py`) → `cert_*.log`, `cert_*.json.gz`; `glue_tubes.py` → `glue_tubes.txt` |
| Table 2 (point values) | `certify/` | `point_values.py` → `point_values.txt` |
| Supplement (j): self-checks of the certification code | `certify/` | `ivec.py`, `test_boxes.py`, `test_grads.py`, `test_grads_mp.py` → `test_*.txt` |
| Supplement (j): cross-check with Arb | `certify/` | `arb_check.py` (with `arb_eval.py`) → `arb_check.txt` |
| Proposition 7.6 (obstruction, exact) | `unrestricted/` | `obstruction_exact.py` → `obstruction_exact.txt` |
| Proposition 7.8(a): ρ_n(3), n = 3..8 | `unrestricted/` | `rhon_certify.py`, `rhon_model.py` → `rhon_certify.txt`, `rhon_certify_78.txt`; `rho3_certify.py` → `rho3_certify.txt` |
| Proposition 7.8(b): adversary bounds V_n | `unrestricted/` | `finite_n_exact.py` (with `adversary_lp.py`) → `finite_n_exact.txt` |
| Sections 7.4–7.5, Appendix C.2, Supplement (k) (evidence) | `unrestricted/` | `conjecture_sweep.py`, `conjecture_search.py`, `band_finite_n.py`, `finite_n_gap.py`, `finite_n_search.py` |
| Supplement (a)–(d): sanity checks of Section 4 | `.` | `firstvar_atom_check.py`, `free_minimization_check.py`, `d1_scan.py`, `d1_no_critical_points.py` |
| Supplement (e): uniqueness evidence | `uniqueness/` | `runmap.py`, `reduce2d.py`, `gamma_grid.py`, `dgamma.py`, `colcheck.py`, `contour.py`, `boundary_roots.py` |
| Supplement (f)–(h), (l): Lemma 5.3, Theorems 6.1, 6.3, 6.4, Remark 6.5 | `asymptotics/` | `closed_form_check.py`, `corner_limits.py`, `indep_planar.py`, `indep_identity_check.py`, `indep_corner.py`, `near_one.py`, `near_one_rates.py`, `uniqueness_jacobians.py` |
| Supplement (i): Theorem 6.6 | `separation/` | `gap_check.py` → `gap_check.txt` |

Each subfolder has a README with details. `shared/` holds two modules of an earlier, independent floating-point solver used by the sanity
checks. Run every script from its own folder; `reproduce.py` does this.

## Version accompanying the manuscript

Release `v1.0.1` is the code snapshot prepared for the Theoretical Computer Science submission.
The computation files were exported with a new public Git history; v1.0.1 corrects
the reproduction instructions and excludes Git metadata from the reproduction workspace.
The public repository contains computation code and supporting outputs only.

Before publication, six selected reproduction steps passed and matched the saved outputs:
`ivec_selftest`, `glue_tubes`, `point_values`, `gap_check`, `obstruction_exact`, and `finite_n_exact`.
These checks do not constitute a fresh run of the full certification cover.

The manuscript and its Supplement are supplied through the journal submission, not in this code repository.
No reuse license has been selected yet.
