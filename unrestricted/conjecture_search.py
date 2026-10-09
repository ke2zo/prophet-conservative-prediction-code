"""Adversarial search for counterexamples to the conjecture g = g_iso at small horizons.

The Jensen baseline (Lemma lem:disint(a)) already proves the conjectured bound on every instance
whose no-prediction ratio OPT_n/E[M_n] is at least rho(c). So we search where it matters: we
minimize, over laws F with k positive atoms plus an atom at 0 (values and probabilities free; consecutive
positive atoms at most a factor c^2 apart),
  (1) the no-prediction ratio OPT_n(F)/E[M_n]                  -> shows the baseline fails,
  (2) the exact adversary optimum over all valid randomized rules / E[M_n]   (adversary_lp.py),
and compare both with certified values of rho(c). Local search: Nelder-Mead from random starts,
then from the minimizer of (1)."""
import math, sys, time
import numpy as np
from scipy.optimize import minimize
from adversary_lp import adversary_optimum, no_prediction_ratio

RHO = {2.0: 0.85925809025, 3.0: 0.82471663643, 4.0: 0.80767753027, 5.0: 0.79730252726}
rng = np.random.default_rng(1)
out = []
def say(s):
    print(s, flush=True); out.append(s)

def decode(z, k, c):
    """z -> (vals, probs): values 0 < 1 = v1 < v2 < ..., plus an atom at 0. Consecutive positive values
    differ by a factor exp(exp(z_i)), capped at c^2 (so the support spans at most 2(k-1) windows;
    beyond that the prediction separates the atoms and the LP becomes ill-conditioned)."""
    gaps = np.minimum(np.exp(np.clip(z[:k - 1], -8, 4)), 2*math.log(c))
    vals = np.concatenate([[0.0], np.exp(np.cumsum(np.concatenate([[0.0], gaps])))])
    w = z[k - 1:]
    p = np.exp(w - w.max()); p = p/p.sum()
    p = 1e-9 + p; p = p/p.sum()
    return list(vals), list(p)

def f_nopred(z, k, n, c):
    v, p = decode(z, k, c)
    return no_prediction_ratio(v, p, n)

def f_adv(z, k, n, c):
    v, p = decode(z, k, c)
    try:
        return adversary_optimum(v, p, c, n)
    except AssertionError:
        return 9.0

def main():
  t0 = time.time()
  settings = [(2, 2.0, 5), (2, 3.0, 5), (3, 2.0, 5), (3, 3.0, 5), (3, 4.0, 5), (4, 2.0, 4), (4, 3.0, 4),
              (4, 5.0, 4), (5, 2.0, 4)]
  if len(sys.argv) > 1: settings = settings[:int(sys.argv[1])]
  for n, c, k in settings:
      best_np = (9, None); best_adv = (9, None)
      starts = [rng.normal(0, 1, 2*k) for _ in range(4)]
      for z0 in starts:
          r = minimize(f_nopred, z0, args=(k, n, c), method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-9})
          if r.fun < best_np[0]: best_np = (r.fun, r.x)
      for z0 in starts[:3] + [best_np[1]]:
          r = minimize(f_adv, z0, args=(k, n, c), method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-8})
          if r.fun < best_adv[0]: best_adv = (r.fun, r.x)
      vN, pN = decode(best_np[1], k, c); vA, pA = decode(best_adv[1], k, c)
      adv_at_np = f_adv(best_np[1], k, n, c)
      say(f"n={n} c={c:g} k={k}+0: min no-prediction ratio {best_np[0]:.5f} (rho(c) = {RHO[c]:.5f}, "
          f"diff {best_np[0] - RHO[c]:+.4f}); adversary optimum there {adv_at_np:.5f}; "
          f"min adversary optimum found {best_adv[0]:.5f} (diff {best_adv[0] - RHO[c]:+.4f})  [{time.time() - t0:.0f}s]")
      say(f"      argmin no-pred: vals={np.round(vN, 4).tolist()} probs={np.round(pN, 4).tolist()}")
      say(f"      argmin adv: vals={np.round(vA, 4).tolist()} probs={np.round(pA, 4).tolist()}")
  open('conjecture_search.txt', 'w').write("\n".join(out) + "\n")

if __name__ == '__main__':
    main()
