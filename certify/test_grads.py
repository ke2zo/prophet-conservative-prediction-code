"""Self-check: the gradient formulas (planar_box.grads) against central differences of the thin
evaluator, at points spread over the certified range."""
import numpy as np
from ivec import I
import planar_iv as P
from planar_box import grads, assemble

def full(y, r):
    V = P.evaluate(I(y), I(r), derivs=True); assemble(V); return V

curve = np.load('curve_float.npy')
idx = np.linspace(0, len(curve) - 1, 25).astype(int)
pts = [(curve[k, 1]*s, curve[k, 0]) for k in idx for s in (0.9, 1.0, 1.1)] + [(0.05, 0.3), (0.3, 0.2), (0.01, 0.5), (0.002, 0.8)]
y = np.array([p[0] for p in pts]); r = np.array([p[1] for p in pts])
V = full(y, r); G = grads(V, I(y), I(r))
ok = V['ok']
worst = 0.0
for key in ('yM', 'DM', 'eps', 'K2', 'K3', 'M', 'P', 'c', 'Gamma', 'rho'):
    for j, var in enumerate('yr'):
        h = 1e-6*(y if var == 'y' else r)
        Vp = full(y + h, r) if var == 'y' else full(y, r + h)
        Vm = full(y - h, r) if var == 'y' else full(y, r - h)
        fd = (Vp[key].mid - Vm[key].mid)/(2*h)
        an = G[key][j].mid
        scale = np.maximum(np.abs(fd), np.abs(V[key].mid)/np.maximum(y if var == 'y' else r, 1e-300)*1e-3)
        rel = np.where(ok, np.abs(fd - an)/scale, 0.0)
        worst = max(worst, float(rel.max()))
print(f"points: {int(ok.sum())}; max relative mismatch (analytic vs central differences): {worst:.1e}")
