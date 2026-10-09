import math, numpy as np
from reduce2d import reduce
d = np.load('gamma_grid.npz'); ly, lr, G = d['ly'], d['lr'], d['G']
# sign of dGamma/d(log y1) on the grid (finite differences along rows of constant r)
dG = np.diff(G, axis=0)            # along ly
pos = np.nansum(dG > 0); neg = np.nansum(dG < 0)
print(f"dGamma/dlog y1 over the whole feasible grid: positive {int(pos)}, negative {int(neg)}")
# per column (fixed r): is Gamma monotone in y1?
mono = 0; nonmono = []
for j in range(len(lr)):
    g = G[:, j]; ok = np.isfinite(g); gg = g[ok]
    if len(gg) < 3: continue
    dd = np.diff(gg)
    if np.all(dd > 0) or np.all(dd < 0): mono += 1
    else: nonmono.append((math.exp(lr[j]), int(np.sum(dd > 0)), int(np.sum(dd < 0))))
print("columns (fixed r) with Gamma monotone in y1:", mono, "; non-monotone:", len(nonmono))
for x in nonmono[:12]: print("   r = %.3g: up %d down %d" % x)
