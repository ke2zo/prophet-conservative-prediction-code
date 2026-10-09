"""Does the finite-horizon analogue of the conjecture hold?  At horizon n, isotone rules give exactly
rho_n(c) (the band constant at horizon n; proof of Theorem A with rho_n in place of rho).  We minimize
the exact adversary optimum over all valid randomized rules (adversary_lp.py) over laws with atoms
{0, 1, a, c*a} spanning two overlapping windows, and compare with rho_n(c) (band_finite_n.py)
and with rho(c)."""
import math, sys, time
import numpy as np
from scipy.optimize import minimize
from adversary_lp import adversary_optimum, no_prediction_ratio
from band_finite_n import rho_n

RHO = {3.0: 0.82471663643}
out = []
def say(s):
    print(s, flush=True); out.append(s)

def decode(z, c):
    """atoms 0, 1, a, c*a with a in (1, c): the top atom sits exactly at the upper end of the window
    [a, ca] of the middle atom, so that the slabs {1, a} and {a, ca} overlap in M = a."""
    a = math.exp(math.log(c)/(1 + math.exp(-z[0])))
    vals = [0.0, 1.0, a, c*a]
    w = z[1:]; p = np.exp(w - w.max()); p = p/p.sum()
    return vals, list(p)

def f(z, c, n):
    v, p = decode(z, c)
    try:
        return adversary_optimum(v, p, c, n)
    except AssertionError:
        return 9.0

t0 = time.time()
ns = [int(x) for x in sys.argv[1:]] or [3, 4, 5, 6]
for c in (3.0,):
    for n in ns:
        rn, _ = rho_n(n, c, k=4, starts=10, seed=n)
        best = (9, None)
        rng = np.random.default_rng(100*n + int(c))
        # seed: the n = 3, c = 3 minimizer (a = 2.2774; p = .438, .128, .343, .092)
        z_seed = np.array([math.log(1/(math.log(c)/math.log(min(2.2774, 0.9*c)) - 1)),
                           math.log(.438), math.log(.128), math.log(.343), math.log(.092)])
        starts = [z_seed] + [rng.normal(0, 1, 5) for _ in range(3)]
        for z0 in starts:
            r = minimize(f, z0, args=(c, n), method='Nelder-Mead',
                         options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-8})
            if r.fun < best[0]: best = (r.fun, r.x)
        v, p = decode(best[1], c)
        nop = no_prediction_ratio(v, p, n)
        say(f"c={c:g} n={n}: rho_n(c) <= {rn:.5f}; min adversary optimum over {{0,1,a,ca}}: {best[0]:.5f} "
            f"(minus rho_n: {best[0] - rn:+.5f}; minus rho(c): {best[0] - RHO[c]:+.5f}); no-prediction ratio there "
            f"{nop:.5f}; a={v[2]:.4f} probs={np.round(p, 4).tolist()}  [{time.time() - t0:.0f}s]")
open('finite_n_gap.txt', 'w').write("\n".join(out) + "\n")
