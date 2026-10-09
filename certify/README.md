# Interval certification of rho = rho_EL on c in [1.0001, 3000]

Rigorous computation behind Section 5.2 (Theorem 5.5) of the manuscript `../../paper/main.tex`. (The working note
`../../notes/strict_monotonicity.tex` certified only [1.001, 1000] and is superseded by the manuscript.)
All enclosures use interval arithmetic: IEEE double with one-ulp outward widening after every
operation (`ivec.py`), logarithms, exponentials and decimal constants from `mpmath.iv` at 80 bits, rounded outward to
double precision and widened by one more ulp. The tail checks (`tails.py`) use `mpmath.iv` at 80 bits directly.

| file | role |
|---|---|
| `ivec.py` | vectorized interval arithmetic (numpy), self-test in `__main__` |
| `planar_iv.py` | thin/naive enclosures of the planar reduction; Taylor-model quadrature on a geometric mesh of [1e-7, 1] |
| `planar_box.py` | tight box enclosures: mean-value forms with analytic gradients, Gamma via E1(eps) |
| `cheap.py` | quadrature-free criteria of Lemma 5.6 (a)-(e) |
| `tails.py` | interval check of the tail inequalities of Lemma 5.7, cases (Z1)-(Z4) of the manuscript ((T1)-(T4) in the note) |
| `tube.py` | Krawczyk tube around the zero curve, gluing, dc/dr > 0, enclosures along the curve |
| `cover.py` | adaptive quadtree exclusion cover of the main rectangle (multiprocessing) |
| `run_certify.py` | driver: tube, then per piece rho_A, tails and cover |
| `point_values.py` | 2D Krawczyk for {Gamma = 0, c = c0}: enclosures of rho(c0), rounded outward to 11 decimals; checks that the upper enclosure is below rho_A of a piece containing c0, so the point lies on the certified curve (Theorem 5.5(ii)) |
| `curve_float.py`, `curve_float_ext.py` | floating-point continuation of the zero curve (input for tube cells; not part of the proof); the extended curve covers c in [1.00005, 11704] |
| `glue_tubes.py` | shows that the zero curves of two certified tubes coincide on their common r-range |
| `float_ref.py`, `planar_float.py` | independent floating-point evaluators, used only by the self-checks |
| `arb_eval.py`, `arb_check.py` | cross-check with a second implementation in Arb ball arithmetic at 128 bits (python-flint 0.9.0), which shares no code with the files above: Arb's rigorous integration instead of the Taylor models. It repeats the tail inequalities, the point values and every test on every tube cell, and excludes again a random sample of the leaves of the cover → `arb_check.txt`. Not used by the certification |
| `test_boxes.py`, `test_grads.py`, `test_grads_mp.py` | self-checks: containment against `float_ref`; gradients vs central differences; interval gradient enclosures vs the derivatives of the independent 30-digit implementation `../asymptotics/indep_planar.py` |

Reproduce. The range is certified in three runs, whose tubes are glued on their overlaps:

```
python3 curve_float.py; python3 curve_float_ext.py
PROCS=4 python3 -u run_certify.py 1.001 1000 1.003 1.01 1.05 1.25 2 5 20 50 100 300 > cert_1.001_1000.log  # ~18 min
CURVE=curve_float_ext.npy PROCS=4 python3 -u run_certify.py 1.0001 1.001 1.0003 > cert_1.0001_1.001.log # < 1 min
CURVE=curve_float_ext.npy PROCS=4 python3 -u run_certify.py 1000 3000 1500 2200 > cert_1000_3000.log # ~66 min, tube included
CURVE=curve_float_ext.npy TUBE=tube_0.721383_0.781416_1e-07.json MAXLEVEL=20 \
    PROCS=4 python3 -u run_certify.py 2200 3000 > cert_2200_3000.log                                  # ~22 min
python3 glue_tubes.py cert_1.001_1000.json.gz cert_1.0001_1.001.json.gz   > glue_tubes.txt
python3 glue_tubes.py cert_1.001_1000.json.gz cert_1000_3000.json.gz     >> glue_tubes.txt
python3 point_values.py > point_values.txt
python3 test_boxes.py 2026 > test_boxes.txt; python3 test_grads.py > test_grads.txt
python3 test_grads_mp.py 100 2026 > test_grads_mp.txt                                              # ~15 min
PROCS=4 python3 -u arb_check.py > arb_check.txt             # optional, needs python-flint; ~45 min
```

The run `1000 3000` certifies the pieces [1000, 1500] and [1500, 2200]; with the default 18 quadtree levels the
piece [2200, 3000] keeps 490 unresolved cells next to the curve near c = 3000 (where the tube cells are thinnest),
so that piece is repeated with the same tube, allowing levels up to 20 (`cert_2200_3000.log`, certified; the
cover finishes at level 19). A tube read from a cache file (`TUBE`, or `tube_*.json` from an earlier run) is re-validated
cell by cell before use, and the log says so. The saved logs come from one sequence of runs from scratch (1 h 47 min with
4 processes). `run_certify.py`
takes the environment variables CURVE, PROCS, TUBE and MAXLEVEL (see its docstring).

`curve_float.py` regenerates `curve_float.npy` up to rounding (relative differences below 1e-10, in derived columns
only); the certification does not depend on these digits: runs from the regenerated curve give the same logs.
`../reproduce.py --tier full` re-runs everything above from scratch (about 2.5 hours with 4 processes).

Outputs: `cert_*.log` (each ends with `CERTIFIED`, except `cert_1000_3000.log`, see above), `cert_*.json.gz` (tube
cells with enclosures of c and rho along the curve, pieces, statistics), `glue_tubes.txt`, `point_values.txt`
(19 widths from 1.0001 to 3000; each point is checked to lie on the certified curve), `test_*.txt`.

Software used for the runs: Python 3.14.0, NumPy 2.3.4, mpmath 1.3.0. The certification does not use SciPy;
SciPy 1.17.1 is used by the floating-point helpers (`float_ref.py`, `planar_float.py`), by several checks elsewhere
in `../`, and for the linear programs in `../unrestricted/`.
