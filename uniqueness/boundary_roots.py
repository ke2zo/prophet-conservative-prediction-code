"""Zeros of Gamma(., r) close to the boundary of the feasible set F, for r where the 360 x 360 grid of
gamma_grid.py sees no zero (the zero lies inside the last grid cell before the boundary).  For fixed r
we locate the upper end y1_max(r) of the feasible y1-interval by bisection, scan Gamma on points
y1 = y1_max (1 - 10^{-k}) with k up to 14, and refine the sign change by bisection.  Uses the float
evaluator of the planar reduction (../certify/planar_float.py).  Run from this folder."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from planar_float import evaluate

def feasible(y1, r):
    return evaluate(y1, r, derivs=False) is not None

def y1_max(r):
    lo, hi = 1e-9, 0.999999
    if not feasible(lo, r): return None
    if feasible(hi, r): return hi
    for _ in range(200):
        mid = math.sqrt(lo*hi)
        if feasible(mid, r): lo = mid
        else: hi = mid
    return lo

out = []
for r in (0.645, 0.7, 0.75, 0.7823, 0.85, 0.9, 0.95):
    top = y1_max(r)
    pts = [top*(1 - 10.0**(-k)) for k in range(1, 15)] + [top]
    vals = []
    for y1 in pts:
        e = evaluate(y1, r, derivs=False)
        if e is not None: vals.append((y1, e['Gamma']))
    root = None
    for (a, ga), (b, gb) in zip(vals, vals[1:]):
        if ga*gb < 0:
            for _ in range(100):
                m = 0.5*(a + b); gm = evaluate(m, r, derivs=False)['Gamma']
                if ga*gm <= 0: b = m
                else: a, ga = m, gm
            root = 0.5*(a + b); break
    if root is None:
        s = f"r={r}: y1_max={top:.8g}; no sign change of Gamma found near the boundary"
    else:
        e = evaluate(root, r, derivs=False)
        s = (f"r={r}: y1_max={top:.8g}; zero of Gamma at y1={root:.10g} (relative distance to the boundary "
             f"{(top - root)/top:.2e}); c={e['c']:.6g}, rho_EL={e['rho']:.6f}")
    print(s, flush=True); out.append(s)
open('boundary_roots.txt', 'w').write("\n".join(out) + "\n")
