"""Finite horizons: instances on which path-dependent rules push the gambler below rho_n(3), the
worst case of isotone rules at horizon n = 3, ..., 8 (Proposition prop:finite(b)).

Upper bound on the adversary's optimum in EXACT rational arithmetic: the LP split (adversary_lp.py),
rounded to rationals and renormalized, splits P into classes whose values of M lie in a window [w, c w]
(checked exactly).  Merging the classes that lie in a common slab and announcing one label per slab is a
valid rule (Lemma lem:disint(b)); since hatV is subadditive, its value is at most sum over classes of
hatV(nu), which is computed by backward induction over the prefix tree in Fractions and rounded UP to six
decimals.  The values for rules depending on M only are floating-point LP optima.  It is compared with the certified lower bound on
rho_n(3) from rhon_certify.py."""
import itertools
from fractions import Fraction as Fr
import numpy as np
from adversary_lp import adversary_optimum, no_prediction_ratio
import math
from rhon_certify import LOWER, LOWER_78
LOWER = {**LOWER, **LOWER_78}

def exact_value(vals, probs, n, dec):
    """dec: dict slab -> dict realization(tuple of atom indices) -> mass (Fractions).
    Returns sum_S hatV(nu_S) by exact backward induction."""
    k = len(vals)
    tot = Fr(0)
    for S, nu in dec.items():
        # W(h) for prefixes; nu(h) = sum over extensions
        pm = {}
        for x, w in nu.items():
            for L in range(1, n + 1):
                pm[x[:L]] = pm.get(x[:L], Fr(0)) + w
        def mass(h):
            return pm.get(h, Fr(0))
        memo = {}
        def W(h):
            if h in memo: return memo[h]
            stop = vals[h[-1]]*mass(h)
            if len(h) == n:
                val = stop
            else:
                val = max(stop, sum((W(h + (i,)) for i in range(k)), Fr(0)))
            memo[h] = val
            return val
        tot += sum((W((i,)) for i in range(k)), Fr(0))
    return tot

def main(c, n, vals, probs):
    vals = [Fr(v) for v in vals]; probs = [Fr(p) for p in probs]
    s = sum(probs); probs = [p/s for p in probs]
    X = list(itertools.product(range(len(vals)), repeat=n))
    P = {x: np.prod([probs[i] for i in x], dtype=object) for x in X}
    M = {x: max(vals[i] for i in x) for x in X}
    EM = sum(P[x]*M[x] for x in X)
    lp, split, slabs = adversary_optimum([float(v) for v in vals], [float(p) for p in probs], float(c), n,
                                         return_split=True)
    # round the split, renormalize per realization, check validity, evaluate exactly
    dec = {}
    for x in X:
        ks = [(x, j) for j in range(len(slabs)) if (x, j) in split]
        th = [Fr(max(split[kk], 0.0)).limit_denominator(10**6) for kk in ks]
        if sum(th) == 0: th = [Fr(1)] + [Fr(0)]*(len(th) - 1)
        tt = sum(th); th = [t/tt for t in th]
        for (xx, j), t in zip(ks, th):
            if t == 0: continue
            dec.setdefault(j, {})[x] = dec.get(j, {}).get(x, Fr(0)) + P[x]*t
    # validity, exactly: in every class the values of M lie in [w, c w] for w := the least of them
    # (or the class is {M = 0}); then announcing w on the class is a valid rule
    for j, nu in dec.items():
        ms = {M[x] for x in nu}
        assert ms == {0} or (min(ms) > 0 and max(ms) <= c*min(ms)), "invalid class"
    assert all(sum((dec[j].get(x, Fr(0)) for j in dec), Fr(0)) == P[x] for x in X), "not a decomposition"
    ub = exact_value(vals, probs, n, dec)/EM
    ub6 = Fr(math.ceil(ub*10**6), 10**6)                 # rounded up
    nopred = no_prediction_ratio([float(v) for v in vals], [float(p) for p in probs], n)
    mmeas = adversary_optimum([float(v) for v in vals], [float(p) for p in probs], float(c), n, mmeas=True)
    lower = Fr(LOWER[n])
    assert ub6 < lower
    print(f"c={c} n={n} atoms={[str(v) for v in vals]} probs={[str(p) for p in probs]}")
    print(f"  adversary optimum: LP {lp:.7f}; exact upper bound from the rounded split {float(ub):.7f} <= {float(ub6):.6f}")
    print(f"  no-prediction ratio {nopred:.7f}; best rule depending on M only {mmeas:.6f}")
    print(f"  certified rho_n(3) >= {LOWER[n]};  deficit >= {float(lower - ub6):.6f}")
    return ub, ub6, lower, nopred, mmeas

if __name__ == '__main__':
    out = []
    for (c, n, vals, probs) in [
        (3, 3, [0, 1, Fr('2.0753'), 3*Fr('2.0753')], [Fr('0.3819'), Fr('0.1779'), Fr('0.3552'), Fr('0.0851')]),
        (3, 4, [0, 1, Fr('2.6133'), 3*Fr('2.6133')], [Fr('0.4492'), Fr('0.2126'), Fr('0.2776'), Fr('0.0605')]),
        (3, 5, [0, 1, Fr('2.7097'), 3*Fr('2.7097')], [Fr('0.4848'), Fr('0.1707'), Fr('0.2835'), Fr('0.061')]),
        (3, 6, [0, 1, Fr('2.2977'), 3*Fr('2.2977')], [Fr('0.5745'), Fr('0.1538'), Fr('0.2234'), Fr('0.0483')]),
        (3, 7, [0, 1, Fr('2.3741'), 3*Fr('2.3741')], [Fr('0.6208'), Fr('0.1533'), Fr('0.1866'), Fr('0.0393')]),
        (3, 8, [0, 1, Fr('2.4050'), 3*Fr('2.4050')], [Fr('0.6369'), Fr('0.1338'), Fr('0.1901'), Fr('0.0392')]),
    ]:
        ub, ub6, lower, nopred, mmeas = main(c, n, vals, probs)
        out.append(f"c={c} n={n} a={float(vals[2]):g} probabilities of 0, 1, a, 3a proportional to "
                   f"{', '.join(f'{float(p):g}' for p in probs)}: exact upper bound on the adversary optimum {float(ub):.7f} <= {float(ub6):.6f}; "
                   f"no-prediction ratio {nopred:.7f}; best rule depending on M only {mmeas:.6f}; certified "
                   f"rho_n(c) >= {LOWER[n]} (rhon_certify.py); deficit >= {float(lower - ub6):.6f}")
    open('finite_n_exact.txt', 'w').write("\n".join(out) + "\n")
