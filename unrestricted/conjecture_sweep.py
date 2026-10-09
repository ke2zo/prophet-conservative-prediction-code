"""Evidence for the conjecture g = g_iso: exact adversary optima over all valid randomized rules
(adversary_lp.py) on random and structured multi-window instances, against certified values of
rho(c) (lower ends of the enclosures in ../certify/point_values.txt and point_values.py 4).
Also records how often path-dependent rules beat every M-measurable rule strictly."""
import math, sys, time
import numpy as np
from adversary_lp import adversary_optimum, no_prediction_ratio

RHO = {2.0: 0.85925809025, 3.0: 0.82471663643, 4.0: 0.80767753027, 5.0: 0.79730252726}
rng = np.random.default_rng(20260923)
out = []
def say(s):
    print(s, flush=True); out.append(s)

def random_instance(c, n):
    k = int(rng.integers(3, 7 if n <= 3 else 6))
    w = rng.uniform(1.2, 3.5)                        # support spans c^w
    vals = np.sort(np.exp(rng.uniform(0, w*math.log(c), k)))
    vals = vals/vals[0]
    probs = rng.dirichlet(np.full(k, 0.6))
    if rng.random() < 0.3:
        vals = np.concatenate([[0.0], vals]); probs = np.concatenate([[rng.uniform(0.1, 0.7)], probs])
    return list(vals), list(probs/np.sum(probs))

def structured_instances(c, n):
    k = 6 if n <= 3 else 5
    for theta in (0.3, 0.5, 0.7):
        for p0 in (0.0, 0.3, 0.6):
            vals = [c**(j/2) for j in range(k)]
            probs = np.array([theta**j for j in range(k)]); probs = (1 - p0)*probs/probs.sum()
            if p0 > 0:
                yield [0.0] + vals, [p0] + list(probs)
            else:
                yield vals, list(probs)

NR = {2: 400, 3: 300, 4: 150, 5: 60}
summary = []
t0 = time.time()
for n in (2, 3, 4, 5):
    for c in (2.0, 3.0, 4.0, 5.0):
        worst = (9, None); worst_np = 9; nstrict = 0; ntot = 0
        insts = list(structured_instances(c, n)) + [random_instance(c, n) for _ in range(NR[n])]
        for vals, probs in insts:
            v = adversary_optimum(vals, probs, c, n)
            vm = adversary_optimum(vals, probs, c, n, mmeas=True)
            npr = no_prediction_ratio(vals, probs, n)
            ntot += 1
            if vm > v + 1e-7: nstrict += 1
            if v - RHO[c] < worst[0]: worst = (v - RHO[c], (vals, probs, v, vm, npr))
            worst_np = min(worst_np, npr - RHO[c])
        vals, probs, v, vm, npr = worst[1]
        say(f"n={n} c={c:g}: {ntot} instances ({ntot - NR[n]} structured); min(adv.opt/E[M] - rho(c)) = {worst[0]:+.4f}; "
            f"path-dependent strictly below every M-measurable rule on {nstrict}/{ntot}; "
            f"min(no-prediction ratio - rho(c)) = {worst_np:+.4f}  [{time.time() - t0:.0f}s]")
        say(f"      worst instance: vals={np.round(vals, 4).tolist()} probs={np.round(probs, 4).tolist()} "
            f"adv={v:.5f} M-meas={vm:.5f} no-pred={npr:.5f}")
        summary.append((n, c, ntot, worst[0], nstrict, worst_np))
open('conjecture_sweep.txt', 'w').write("\n".join(out) + "\n")
