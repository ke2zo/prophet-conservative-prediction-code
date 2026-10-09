"""Instances for horizons n = 7, 8 (Proposition prop:finite(b)): Nelder-Mead over laws with atoms {0, 1, a, 3a}
minimizing the adversary's optimum (adversary_lp.py), from several starts in parallel.  Not part of any proof:
the instances it finds are checked exactly by finite_n_exact.py.
  python3 finite_n_search.py 7 700 8                          -> finite_n_search_7.txt (starts: the n = 3..6 instances
                                                                 and perturbations of the n = 6 one)
  python3 finite_n_search.py 8 220 2 "2.40,0.655,0.15,0.163,0.032" "2.37408,0.62082,0.15329,0.18662,0.03928"
                                                              -> finite_n_search_8.txt (one start finished; the second
                                                                 was stopped after two hours)
Arguments: n, maximal number of Nelder-Mead iterations, number of processes, optional starts "a,p0,p1,p2,p3"."""
import sys, math, time, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings('ignore')
C = 3.0

def decode(z):
    a = math.exp(math.log(C)/(1 + math.exp(-z[0])))
    w = np.asarray(z[1:]); p = np.exp(w - w.max()); p = p/p.sum()
    return [0.0, 1.0, a, C*a], list(p)

def encode(a, p):
    t = math.log(C)/math.log(a)          # 1 + exp(-z0) = log C / log a
    return np.array([-math.log(t - 1)] + [math.log(x) for x in p])

def f(z, n):
    from adversary_lp import adversary_optimum
    v, p = decode(z)
    try:
        return adversary_optimum(v, p, C, n)
    except Exception:
        return 9.0

def run(args):
    n, z0, maxiter, tag = args
    from scipy.optimize import minimize
    t0 = time.time()
    r = minimize(f, z0, args=(n,), method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-9})
    v, p = decode(r.x)
    return tag, r.fun, v[2], p, r.nfev, time.time() - t0

if __name__ == '__main__':
    n = int(sys.argv[1]); maxiter = int(sys.argv[2]); nproc = int(sys.argv[3])
    known = {3: (2.0753, [0.3819, 0.1779, 0.3552, 0.0851]), 4: (2.6133, [0.4492, 0.2126, 0.2776, 0.0605]),
             5: (2.7097, [0.4848, 0.1707, 0.2835, 0.061]), 6: (2.2977, [0.5745, 0.1538, 0.2234, 0.0483])}
    extra = [tuple(map(float, s.split(':')[0:1])) for s in []]
    starts = [] if len(sys.argv) > 4 else [(f"from n={m}", encode(a, p)) for m, (a, p) in known.items()]
    for s in sys.argv[4:]:                        # extra starts "a,p0,p1,p2,p3"
        vals = [float(x) for x in s.split(',')]
        starts.append((f"given {s}", encode(vals[0], vals[1:])))
    rng = np.random.default_rng(n)
    z6 = encode(*known[6])
    while len(starts) < nproc:
        starts.append((f"perturbed n=6 #{len(starts)}", z6 + rng.normal(0, 0.25, 5)))
    with Pool(nproc) as pool:
        for tag, fun, a, p, nfev, dt in pool.imap_unordered(run, [(n, z, maxiter, t) for t, z in starts]):
            print(f"n={n} [{tag}] adversary optimum {fun:.7f}  a={a:.5f} probs={np.round(p, 5).tolist()}  "
                  f"({nfev} LPs, {dt:.0f}s)", flush=True)
