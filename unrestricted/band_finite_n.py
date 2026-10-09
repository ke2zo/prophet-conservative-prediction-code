"""Finite-horizon band constants rho_n(c) = inf over band laws F (support in {0} u [1,c]) of
OPT_n(F)/E[M_n], estimated from above by local search over laws with an atom at 0 and k atoms in
[1, c] (positions and probabilities free).  On a band law the adversary's optimum equals OPT_n(F)
(Lemma lem:disint(a) and the inert binary rule), so rho_n(c) is also the value of the finite-horizon
analogue of Conjecture conj:general restricted to band laws."""
import math, sys
import numpy as np
from scipy.optimize import minimize
from adversary_lp import no_prediction_ratio

def decode(z, k, c):
    pos = 1 + (c - 1)/(1 + np.exp(-z[:k]))
    w = z[k:]; p = np.exp(w - w.max()); p = p/p.sum()
    return [0.0] + list(pos), list(p)

def rho_n(n, c, k=4, starts=12, seed=0):
    rng = np.random.default_rng(seed)
    best = (9, None)
    f = lambda z: no_prediction_ratio(*decode(z, k, c), n)
    for _ in range(starts):
        z0 = np.concatenate([rng.normal(0, 2, k), rng.normal(0, 1, k + 1)])
        r = minimize(f, z0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-12})
        r = minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-13})
        if r.fun < best[0]: best = (r.fun, r.x)
    return best[0], decode(best[1], k, c)

if __name__ == '__main__':
    out = []
    for n in (2, 3, 4, 5):
        for c in (2.0, 3.0, 4.0, 5.0):
            v, (vals, probs) = rho_n(n, c)
            s = (f"n={n} c={c:g}: rho_n(c) <= {v:.6f}   at vals={np.round(vals, 4).tolist()} "
                 f"probs={np.round(probs, 4).tolist()}")
            print(s, flush=True); out.append(s)
    open('band_finite_n.txt', 'w').write("\n".join(out) + "\n")
