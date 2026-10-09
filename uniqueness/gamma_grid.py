import math, numpy as np
from reduce2d import reduce, psi
ly = np.linspace(math.log(1e-6), math.log(0.999), 360)      # log y1
lr = np.linspace(math.log(1e-6), math.log(0.999), 360)      # log r
Gm = np.full((len(ly), len(lr)), np.nan); Cm = np.full_like(Gm, np.nan); Rm = np.full_like(Gm, np.nan)
for i, a in enumerate(ly):
    for j, b in enumerate(lr):
        try:
            d = reduce(math.exp(a), math.exp(b))
        except Exception:
            d = None
        if d is not None and np.isfinite(d['Gamma']):
            Gm[i, j] = d['Gamma']; Cm[i, j] = d['c']; Rm[i, j] = d['rho']
np.savez('gamma_grid.npz', ly=ly, lr=lr, G=Gm, C=Cm, R=Rm)
valid = np.isfinite(Gm)
print("valid grid points:", int(valid.sum()), "of", Gm.size)
# zero crossings along r (for each y1 row) and along y1 (for each r column)
rows = []
for i in range(len(ly)):
    g = Gm[i]; idx = [j for j in range(len(lr)-1) if np.isfinite(g[j]) and np.isfinite(g[j+1]) and g[j]*g[j+1] < 0]
    rows.append(len(idx))
print("number of sign changes of Gamma along r, per y1-row: min", min(rows), "max", max(rows))
print("rows with k crossings:", {k: rows.count(k) for k in sorted(set(rows))})
