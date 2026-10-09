"""Self-check of the gradient formulas (planar_box.grads) that enter the mean-value forms and the
Krawczyk tests.  At random points (y1, r), the interval enclosures of the partial derivatives,
computed by planar_box.evaluate_box on the point box, must contain the derivatives of the
independent 30-digit implementation ../asymptotics/indep_planar.py (written from the definitions
only), obtained by the five-point central difference with relative step 1e-6 (error ~1e-20).
Points: half log-uniform in y1 in [1e-5, 0.6] and uniform in logit r in [-9, 3]; half within 10% in y1
of the zero curve (curve_float_ext.npy).  Run from this folder: python3 test_grads_mp.py [N] [seed]."""
import os, sys, numpy as np, mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'asymptotics'))
from indep_planar import reduced
from ivec import I
from planar_box import evaluate_box

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 2026)
curve = np.load('curve_float_ext.npy')
ys, rs = [], []
for k in range(N):
    if k % 2 == 0:
        ys.append(float(np.exp(rng.uniform(np.log(1e-5), np.log(0.6)))))
        rs.append(float(1/(1 + np.exp(-rng.uniform(-9, 3)))))
    else:
        row = curve[rng.integers(len(curve))]
        ys.append(float(row[1]*np.exp(rng.uniform(-0.1, 0.1)))); rs.append(float(row[0]))
V, G, ok, C = evaluate_box(I(np.array(ys)), I(np.array(rs)))
KEYS = ('S0', 'Ts', 'q', 'tau', 'kap', 'y0', 'yM', 'eps', 'K2', 'K3', 'M', 'c', 'Gamma', 'rho')

def val(R, key):
    return R[key]

tested = viol = 0
worst_out = mp.mpf(0); widths = []
for k in range(N):
    if not ok[k]:
        continue
    y1, r = mp.mpf(ys[k]), mp.mpf(rs[k])
    hs = (mp.mpf('1e-6')*y1, mp.mpf('1e-6')*min(r, 1 - r))
    base = reduced(y1, r)
    if base is None:
        continue
    stencil = {}
    good = True
    for j, h in enumerate(hs):
        for m in (-2, -1, 1, 2):
            R_ = reduced(y1 + m*h, r) if j == 0 else reduced(y1, r + m*h)
            if R_ is None:
                good = False; break
            stencil[(j, m)] = R_
        if not good:
            break
    if not good:
        continue
    tested += 1
    for key in KEYS:
        for j, h in enumerate(hs):
            f = lambda m: val(stencil[(j, m)], key)
            d = (8*(f(1) - f(-1)) - (f(2) - f(-2)))/(12*h)
            lo, hi = mp.mpf(float(G[key][j].lo[k])), mp.mpf(float(G[key][j].hi[k]))
            scale = max(abs(d), mp.mpf('1e-300'))
            out = max(lo - d, d - hi, 0)/scale
            if out > 0:
                viol += 1
                print(f"VIOLATION {key} d/d{'yr'[j]} at y1={ys[k]:.6e} r={rs[k]:.6e}: {mp.nstr(d, 20)} not in [{lo}, {hi}]")
            worst_out = max(worst_out, out)
            widths.append(float((hi - lo)/scale))
print(f"points tested: {tested} of {N}; partial derivatives checked: {tested*len(KEYS)*2}; "
      f"not enclosed: {viol}; median relative width of the enclosures: {np.median(widths):.1e}, "
      f"largest: {max(widths):.1e}")
