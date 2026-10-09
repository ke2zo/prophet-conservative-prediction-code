# Shared modules

`el_shoot_solver.py` (an earlier, independent shooting solver for the Euler–Lagrange system) and
`rho_reduced2.py` (the reduced continuum evaluator `PWS`) are unchanged copies of the files of the same name in the
folder `verification/` of the working repository (commit a811a81). They are used by the checks
`../firstvar_atom_check.py`, `../free_minimization_check.py`, `../d1_scan.py` and `../uniqueness/runmap.py`
(through `../uniqueness/elmap.py`); none of the proofs uses them. Their `__main__` blocks are not run by the checks.
