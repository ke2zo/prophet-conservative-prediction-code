import math, numpy as np
from scipy import optimize
from reduce2d import reduce, psi
d = np.load('gamma_grid.npz'); ly, lr, G = d['ly'], d['lr'], d['G']
cols = []
for j in range(len(lr)):
    g = G[:, j]; idx = [i for i in range(len(ly)-1) if np.isfinite(g[i]) and np.isfinite(g[i+1]) and g[i]*g[i+1] < 0]
    cols.append(len(idx))
print("sign changes of Gamma along y1, per r-column: ", {k: cols.count(k) for k in sorted(set(cols))})
# invalid-region diagnosis
reasons = {'target<=0': 0, 'target>=K1(1)': 0, 'numerical': 0}
from reduce2d import XI, WXI, Phi
for a in ly[::6]:
    for b in lr[::6]:
        y1, r = math.exp(a), math.exp(b)
        S0 = -math.log(y1*psi(r)); target = 1 + math.log(r)/S0
        q = S0/(1-r); tau = q*y1
        t = y1 + (1-1e-15-y1)*XI; K1max = (1-y1)*np.dot(WXI, 1/(tau + Phi(y1) - Phi(t)))
        if target <= 0: reasons['target<=0'] += 1
        elif K1max <= target: reasons['target>=K1(1)'] += 1
        elif reduce(y1, r) is None: reasons['numerical'] += 1
print("invalid points by reason (subsampled grid):", reasons)
# direct tracing of the zero curve: for each r on a fine grid, solve Gamma(y1, r) = 0 for y1
print(f"{'r':>9} {'#roots in y1':>12} {'y1 root(s)':>30} {'c':>10} {'rho':>9}")
out = []
for r in np.exp(np.linspace(math.log(1e-5), math.log(0.95), 60)):
    ys = np.exp(np.linspace(math.log(1e-7), math.log(0.9999), 1500))
    vals = []
    for y in ys:
        dd = reduce(y, r); vals.append(dd['Gamma'] if dd else np.nan)
    vals = np.array(vals); roots = []
    for i in range(len(ys)-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
            yr = optimize.brentq(lambda y: reduce(y, r)['Gamma'], ys[i], ys[i+1], xtol=1e-14)
            roots.append(yr)
    info = [(yr, reduce(yr, r)) for yr in roots]
    out.append((r, info))
    print(f"{r:9.3e} {len(roots):12d} {', '.join(f'{yr:.5f}' for yr,_ in info):>30} "
          f"{', '.join(f'{i[1]['c']:.4g}' for i in info):>10} {', '.join(f'{i[1]['rho']:.6f}' for i in info):>9}")
