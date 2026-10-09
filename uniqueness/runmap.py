import sys, math, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from el_shoot_solver import solve, ev
from elmap import full_solve, trace, outer, csolve, G
rho, r = solve(2.0)
rr = ev(r['M'], r['eps'], r['kappa'], r['S0'], 2.0, 200000)
base = (np.array([r['eps'], r['M'], r['kappa'], rr['y1p']]), r['S0'])
cs_up = np.exp(np.linspace(math.log(2.0), math.log(1e4), 70))[1:]
cs_dn = np.exp(np.linspace(math.log(2.0), math.log(1.02), 25))[1:]
sols = {2.0: full_solve(2.0, *base)}
for seq in (cs_up, cs_dn):
    prev_c = 2.0
    for c in seq:
        v, S0 = sols[prev_c]
        fs = full_solve(c, v, S0)
        if fs is None:   # sub-step in c
            ok = False
            for k in (2, 4, 8, 16):
                vv, SS = v, S0; good = True
                for j in range(1, k+1):
                    cj = prev_c*(c/prev_c)**(j/k)
                    fj = full_solve(cj, vv, SS)
                    if fj is None: good = False; break
                    vv, SS = fj
                if good: fs = (vv, SS); ok = True; break
            if not ok:
                print(f"c={c:.4g}: continuation of the full solution FAILED"); break
        sols[c] = fs; prev_c = c
print(f"{'c':>9} {'S0*':>7} {'rho':>9} {'atom@1':>7} {'range S0':>17} {'#zeros':>6} {'dDelta/dS0 at zero':>18} {'Delta left end':>14} {'Delta right end':>15}")
rows = []
for c in sorted(sols):
    v, S0s = sols[c]
    D, g, a1 = outer(v, S0s, c)
    left = trace(c, v, S0s, -1); right = trace(c, v, S0s, +1)
    path = [(S, w) for S, w in reversed(left)] + [(S0s, v)] + right
    Ss = np.array([p[0] for p in path]); Ds = np.array([outer(p[1], p[0], c)[0] for p in path])
    sgn = np.sign(Ds); zeros = int(np.sum(sgn[1:]*sgn[:-1] < 0) + np.sum(Ds == 0))
    i = int(np.argmin(np.abs(Ss - S0s)))
    dD = (Ds[min(i+1, len(Ds)-1)] - Ds[max(i-1, 0)])/(Ss[min(i+1, len(Ss)-1)] - Ss[max(i-1, 0)])
    rows.append((c, S0s, g, a1, Ss[0], Ss[-1], zeros, dD, Ds[0], Ds[-1]))
    print(f"{c:9.4g} {S0s:7.3f} {g:9.6f} {a1:7.4f} [{Ss[0]:6.3f},{Ss[-1]:7.3f}] {zeros:6d} {dD:18.4e} {Ds[0]:+14.4e} {Ds[-1]:+15.4e}", flush=True)
np.save('map_rows.npy', np.array(rows))
# comparison with the continuation branch of the planar reduction (another method and another code path)
from scipy.interpolate import CubicSpline
cur = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify', 'curve_float_ext.npy'))
cur = cur[np.argsort(cur[:, 2])]
sp = CubicSpline(np.log(cur[:, 2]), cur[:, 3])
dev = max(abs(float(sp(math.log(row[0]))) - row[2]) for row in rows)
print(f"comparison with the continuation branch (planar reduction): max |rho - rho_branch| over the {len(rows)} widths = {dev:.1e}")
