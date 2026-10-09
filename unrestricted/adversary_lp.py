"""Exact adversary optimum over all valid randomized prediction rules, for finitely supported F and
any horizon n, as ONE linear program (no column generation).

For a slab S (a set of values of M admissible for one label) and an unnormalized measure nu on
realizations, the gambler's optimal value hatV(nu) is the least W satisfying the Bellman
inequalities on the prefix tree:
    W(h) >= x_k * nu(h)                (stop on the k-th value x_k of the prefix h, |h| = k)
    W(h) >= sum_i W(h i)               (continue; |h| < n)
    hatV(nu) = sum_i W((i)).
nu_S(h) = sum_{x extends h} P(x) theta(x, S) is linear in the adversary's split theta, so
    min_theta sum_S hatV(nu_S)
is a single LP. Its optimum is the adversary optimum (Lemma lem:disint(b): one label per slab).
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def slabs_of(Mvals, c):
    pos = sorted(m for m in set(Mvals) if m > 0)
    crit = sorted(set(pos + [m/c for m in pos]))
    cands = []
    for i, w in enumerate(crit):
        cands.append(w)
        if i + 1 < len(crit): cands.append(0.5*(w + crit[i + 1]))
    out = []
    for w in cands:
        s = frozenset(m for m in pos if w*(1 - 1e-12) <= m <= c*w*(1 + 1e-12))
        if s and s not in out: out.append(s)
    if any(m == 0 for m in Mvals): out.append(frozenset([0.0]))
    return out


def adversary_optimum(vals, probs, c, n, mmeas=False, return_split=False):
    """Returns (adversary optimum / E[M]). mmeas=True restricts the split theta to depend on M only.
    With return_split=True also returns the optimal split {(realization index tuple, slab): theta}
    (realizations as tuples of atom indices; slabs as frozensets of values of M, unnormalized)."""
    vals = [float(v) for v in vals]; probs = np.asarray(probs, float); probs = probs/probs.sum()
    # the ratio is scale invariant; normalize the values by E[M] for the conditioning of the LP
    order = np.argsort(vals); vs = np.asarray(vals)[order]; Fc = np.cumsum(probs[order])
    em = float(np.sum(vs*(Fc**n - np.concatenate([[0.0], Fc[:-1]])**n)))
    vals = [v/em for v in vals]
    k = len(vals)
    X = list(itertools.product(range(k), repeat=n))
    P = np.array([np.prod([probs[i] for i in x]) for x in X])
    M = np.array([max(vals[i] for i in x) for x in X])
    EM = float(P @ M)
    slabs = slabs_of(list(M), c)
    # theta variables
    groups = sorted(set(M)) if mmeas else list(range(len(X)))
    gof = (lambda xi: M[xi]) if mmeas else (lambda xi: xi)
    Mg = (lambda g: g) if mmeas else (lambda g: M[g])
    th = [(g, j) for g in groups for j, S in enumerate(slabs) if Mg(g) in S]
    thidx = {t: i for i, t in enumerate(th)}
    nvar = len(th)
    # prefix variables per slab (only prefixes of realizations that can carry the slab)
    Widx = {}
    for j, S in enumerate(slabs):
        for xi, x in enumerate(X):
            if M[xi] in S:
                for L in range(1, n + 1):
                    h = x[:L]
                    if (j, h) not in Widx:
                        Widx[(j, h)] = nvar; nvar += 1
    rows, cols, data, rhs = [], [], [], []
    r = 0
    # stop constraints: x_k * nu_j(h) - W(j,h) <= 0
    ext = {}
    for xi, x in enumerate(X):
        for L in range(1, n + 1):
            ext.setdefault(x[:L], []).append(xi)
    for (j, h), wi in Widx.items():
        xk = vals[h[-1]]
        for xi in ext[h]:
            t = (gof(xi), j)
            if t in thidx and xk*P[xi] != 0:
                rows.append(r); cols.append(thidx[t]); data.append(xk*P[xi])
        rows.append(r); cols.append(wi); data.append(-1.0); rhs.append(0.0); r += 1
    # continue constraints: sum_i W(j, h i) - W(j, h) <= 0
    for (j, h), wi in Widx.items():
        if len(h) < n:
            for i in range(k):
                ci = Widx.get((j, h + (i,)))
                if ci is not None:
                    rows.append(r); cols.append(ci); data.append(1.0)
            rows.append(r); cols.append(wi); data.append(-1.0); rhs.append(0.0); r += 1
    A_ub = coo_matrix((data, (rows, cols)), shape=(r, nvar)).tocsr()
    # equalities: sum_j theta(g, j) = 1
    er, ec, ed = [], [], []
    for gi, g in enumerate(groups):
        for j in range(len(slabs)):
            t = (g, j)
            if t in thidx:
                er.append(gi); ec.append(thidx[t]); ed.append(1.0)
    A_eq = coo_matrix((ed, (er, ec)), shape=(len(groups), nvar)).tocsr()
    cost = np.zeros(nvar)
    for (j, h), wi in Widx.items():
        if len(h) == 1: cost[wi] = 1.0
    bounds = [(0, None)]*len(th) + [(None, None)]*(nvar - len(th))
    res = linprog(cost, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=np.ones(len(groups)),
                  bounds=bounds, method='highs')
    assert res.status == 0, res.message
    if return_split:
        split = {}
        for (g, j), i in thidx.items():
            if mmeas:
                for xi in range(len(X)):
                    if M[xi] == g: split[(X[xi], j)] = res.x[i]
            else:
                split[(X[g], j)] = res.x[i]
        return res.fun/EM, split, [frozenset(m*em for m in S) for S in slabs]
    return res.fun/EM


def no_prediction_ratio(vals, probs, n):
    """OPT_n(F)/E[M_n] without prediction (backward recursion)."""
    vals = np.asarray(vals, float); probs = np.asarray(probs, float); probs = probs/probs.sum()
    W = 0.0
    for _ in range(n):
        W = float(np.sum(probs*np.maximum(vals, W)))
    order = np.argsort(vals); v = vals[order]; p = probs[order]
    Fc = np.cumsum(p)
    EM = float(np.sum(v*(Fc**n - np.concatenate([[0.0], Fc[:-1]])**n)))
    return W/EM


if __name__ == '__main__':
    # Reproduces the exact values of obstruction_exact.py; output saved in adversary_lp.txt.
    from fractions import Fraction as Fr
    out = []
    def say(t): print(t); out.append(t)
    X, P = [1, 2, 4, 8, 32], [1]*5
    for mm, ex, lab in ((False, Fr(349, 371), 'all valid rules'), (True, Fr(353, 371), 'rules depending on M only')):
        v = adversary_optimum(X, P, 4.0, 2, mmeas=mm)
        say(f"obstruction instance (c = 4, n = 2), {lab}: LP {v:.12f}; exact {ex} = {float(ex):.12f}; "
            f"|difference| = {abs(v - float(ex)):.1e}")
    v = adversary_optimum([0, 5.4443, 18.6798, 26.0801, 43.5019], [.2630, .6181, .1048, .0029, .0111], 5.0, 2)
    say(f"natural-family breach (c = 5, n = 2), all valid rules: LP {v:.12f}; exact 0.893586142 to 9 digits "
        f"(obstruction_exact.txt); |difference| = {abs(v - 0.893586142):.1e}")
    for mm, lab in ((False, 'all valid rules'), (True, 'rules depending on M only')):
        say(f"obstruction instance at horizon n = 3, {lab}: LP {adversary_optimum(X, P, 4.0, 3, mmeas=mm):.9f}")
    open('adversary_lp.txt', 'w').write("\n".join(out) + "\n")
