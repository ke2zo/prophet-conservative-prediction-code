"""Unrestricted prediction rules at horizon n = 2: exact adversary optima with rational certificates.

For a finitely supported F the adversary's problem is the finite game of Lemma lem:envelope:
the adversary splits each realization x among the slabs S containing it, the gambler picks a
(randomized) stopping rule per slab. At n = 2 the deterministic stopping rules are 'accept X1 iff
X1 in A' (then accept X2), A a subset of the support.

For each instance we solve, with scipy's HiGHS, both sides of the game
  primal:  min over decompositions  sum_S hatV(nu_S)
  dual:    max over families (q_S)  E[ min_{S contains X} r_{q_S}(X) ]
and then VERIFY the bounds in exact rational arithmetic: the rounded primal solution is a valid
decomposition whose value is computed exactly (upper bound on the optimum), and the rounded dual
solution is a valid family whose envelope is computed exactly (lower bound). The same is done
for the M-measurable relaxation (labels that depend on the realization only through M).
"""
import itertools, sys
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


class Instance:
    def __init__(self, support, probs, c, n=2):
        assert n == 2
        self.sup = [Fr(x) for x in support]
        self.p = [Fr(x) for x in probs]
        tot = sum(self.p); self.p = [q/tot for q in self.p]
        self.c = Fr(c)
        self.X = list(itertools.product(range(len(self.sup)), repeat=2))
        self.P = {x: self.p[x[0]]*self.p[x[1]] for x in self.X}
        self.M = {x: max(self.sup[x[0]], self.sup[x[1]]) for x in self.X}
        self.EM = sum(self.P[x]*self.M[x] for x in self.X)
        # slabs: distinct nonempty sets {m in Mvals : w <= m <= c w}, w > 0; plus {0} if 0 occurs
        mv = sorted(set(self.M.values()))
        pos = [m for m in mv if m > 0]
        crit = sorted(set(pos + [m/self.c for m in pos]))
        cands = []
        for i, w in enumerate(crit):
            cands.append(w)
            if i + 1 < len(crit): cands.append((w + crit[i + 1])/2)
        slabs = []
        for w in cands:
            s = frozenset(m for m in pos if w <= m <= self.c*w)
            if s and s not in slabs: slabs.append(s)
        if 0 in mv: slabs.append(frozenset([Fr(0)]))
        self.slabs = slabs
        self.Mvals = mv
        # policies: accept X1 iff X1 in A (A subset of support indices)
        k = len(self.sup)
        self.pols = [frozenset(i for i in range(k) if (b >> i) & 1) for b in range(2**k)]
    def r(self, A, x):
        return self.sup[x[0]] if x[0] in A else self.sup[x[1]]
    def hatV(self, nu):
        """exact max over policies of sum_x r(A,x) nu(x); nu: dict x -> Fraction"""
        return max(sum(self.r(A, x)*w for x, w in nu.items()) for A in self.pols)
    def value_of_decomposition(self, dec):
        """dec: dict slab -> dict x -> mass (exact). Checks validity and returns sum hatV / E[M]."""
        tot = {x: Fr(0) for x in self.X}
        for S, nu in dec.items():
            for x, w in nu.items():
                assert w >= 0 and (w == 0 or self.M[x] in S), "invalid decomposition"
                tot[x] += w
        assert all(tot[x] == self.P[x] for x in self.X), "not a decomposition of P"
        return sum(self.hatV(nu) for nu in dec.values())/self.EM


def solve(inst, mmeas=False):
    """LP for the adversary optimum (mmeas: labels depend on M only). Returns float value, and exact
    rational upper/lower bounds from the rounded primal/dual solutions."""
    S_list = inst.slabs; pols = inst.pols
    # primal variables: theta[g, S] for g a realization (or an M-value if mmeas) and S containing M(g)
    groups = inst.Mvals if mmeas else inst.X
    Mof = (lambda g: g) if mmeas else (lambda g: inst.M[g])
    th = [(g, j) for g in groups for j, S in enumerate(S_list) if Mof(g) in S]
    nth = len(th); nv = len(S_list)
    idx = {k: i for i, k in enumerate(th)}
    members = {g: [x for x in inst.X if inst.M[x] == g] if mmeas else [g] for g in groups}
    A_ub, b_ub = [], []
    for j in range(nv):
        for A in pols:
            row = np.zeros(nth + nv)
            for (g, jj), i in idx.items():
                if jj == j:
                    row[i] = float(sum(inst.P[x]*inst.r(A, x) for x in members[g]))
            row[nth + j] = -1.0
            A_ub.append(row); b_ub.append(0.0)
    A_eq, b_eq = [], []
    for g in groups:
        row = np.zeros(nth + nv)
        for (gg, jj), i in idx.items():
            if gg == g: row[i] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    cost = np.concatenate([np.zeros(nth), np.ones(nv)])
    res = linprog(cost, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)]*(nth + nv), method='highs')
    assert res.status == 0
    val = res.fun/float(inst.EM)
    # exact upper bound: round theta, renormalize per group, evaluate
    theta = {}
    for g in groups:
        ks = [(g, j) for (gg, j) in th if gg == g]
        vals = [Fr(max(res.x[idx[k]], 0.0)).limit_denominator(10**6) for k in ks]
        s = sum(vals); vals = [v/s for v in vals]
        for k, v in zip(ks, vals): theta[k] = v
    dec = {}
    for (g, j), t in theta.items():
        for x in members[g]:
            dec.setdefault(S_list[j], {})
            dec[S_list[j]][x] = dec[S_list[j]].get(x, Fr(0)) + inst.P[x]*t
    ub = inst.value_of_decomposition(dec)
    # dual: max sum_g t_g  s.t.  t_g <= sum_A q_{S,A} sum_{x in g} P(x) r(A,x)  (S containing M(g)),
    #       sum_A q_{S,A} = 1, q >= 0.  Variables: q (nv*len(pols)), t (len(groups)).
    nq = nv*len(pols); ng = len(groups)
    A_ub, b_ub = [], []
    for gi, g in enumerate(groups):
        for j, S in enumerate(S_list):
            if Mof(g) not in S: continue
            row = np.zeros(nq + ng)
            row[nq + gi] = 1.0
            for a, A in enumerate(pols):
                row[j*len(pols) + a] = -float(sum(inst.P[x]*inst.r(A, x) for x in members[g]))
            A_ub.append(row); b_ub.append(0.0)
    A_eq, b_eq = [], []
    for j in range(nv):
        row = np.zeros(nq + ng); row[j*len(pols):(j + 1)*len(pols)] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    cost = np.concatenate([np.zeros(nq), -np.ones(ng)])
    res2 = linprog(cost, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                   bounds=[(0, None)]*nq + [(None, None)]*ng, method='highs')
    assert res2.status == 0
    q = {}
    for j in range(nv):
        vals = [Fr(max(res2.x[j*len(pols) + a], 0.0)).limit_denominator(10**6) for a in range(len(pols))]
        s = sum(vals); q[j] = [v/s for v in vals]
    def rq(j, x): return sum(q[j][a]*inst.r(A, x) for a, A in enumerate(pols))
    lb = sum(min(sum(inst.P[x]*rq(j, x) for x in members[g]) for j, S in enumerate(S_list) if Mof(g) in S)
             for g in groups)/inst.EM
    return val, ub, lb, dec, q


def natural_family_envelope(inst):
    """pi_S := optimal policy of P restricted to slab S (ties: accept); envelope E[min_{S contains X} r]."""
    best = {}
    for S in inst.slabs:
        nu = {x: inst.P[x] for x in inst.X if inst.M[x] in S}
        vals = [(sum(inst.r(A, x)*w for x, w in nu.items()), -len(A), A) for A in inst.pols]
        best[S] = max(vals)[2]
    env = sum(inst.P[x]*min(inst.r(best[S], x) for S in inst.slabs if inst.M[x] in S) for x in inst.X)
    return env/inst.EM


def report(name, inst, rho):
    say(f"=== {name}: support {[str(s) for s in inst.sup]}, probs {[str(p) for p in inst.p]}, c = {inst.c}, n = 2")
    say(f"    E[M] = {float(inst.EM):.6f}; slabs (sets of M-values): {[sorted(float(m) for m in S) for S in inst.slabs]}")
    v, ub, lb, dec, q = solve(inst)
    say(f"    adversary optimum over all valid randomized rules: LP {v:.7f}; exact bounds [{float(lb):.9f}, {float(ub):.9f}]")
    v2, ub2, lb2, _, _ = solve(inst, mmeas=True)
    say(f"    optimum over M-measurable (fractional) rules:      LP {v2:.7f}; exact bounds [{float(lb2):.9f}, {float(ub2):.9f}]")
    # deterministic M-measurable rules
    choices = [[S for S in inst.slabs if m in S] for m in inst.Mvals]
    bestdet = None
    for pick in itertools.product(*choices):
        dec2 = {}
        for m, S in zip(inst.Mvals, pick):
            for x in inst.X:
                if inst.M[x] == m:
                    dec2.setdefault(S, {})[x] = inst.P[x]
        val = inst.value_of_decomposition(dec2)
        if bestdet is None or val < bestdet: bestdet = val
    say(f"    optimum over deterministic M-measurable rules (exact, {np.prod([len(ch) for ch in choices])} rules): {float(bestdet):.9f}")
    nat = natural_family_envelope(inst)
    say(f"    envelope of the natural family (optimal policy per slab): {float(nat):.7f};  rho(c) = {rho}")
    return lb, ub, lb2, ub2, bestdet, nat


# ---------------------------------------------------------------- the obstruction instance
inst = Instance([1, 2, 4, 8, 32], [1]*5, 4)
lb, ub, lb2, ub2, bestdet, nat = report("obstruction instance", inst, "0.80767753027... (certified: ../certify/point_values.py 4)")
# the explicit three-label rule: [1,4]: M=1, or M in {2,4} and X1 = 1;  [2,8]: M in {2,4} and X1 >= 2,
# or M = 8 and X1 < 8;  [8,32]: M = 32, or M = 8 and X1 = 8.
S = {tuple(sorted(float(m) for m in s)): s for s in inst.slabs}
A_, B_, C_ = S[(1.0, 2.0, 4.0)], S[(2.0, 4.0, 8.0)], S[(8.0, 32.0)]
dec = {A_: {}, B_: {}, C_: {}}
for x in inst.X:
    m = inst.M[x]; x1 = inst.sup[x[0]]
    if m == 32 or (m == 8 and x1 >= 8): dec[C_][x] = inst.P[x]
    elif m == 1 or (m in (2, 4) and x1 < 2): dec[A_][x] = inst.P[x]
    else: dec[B_][x] = inst.P[x]
v3 = inst.value_of_decomposition(dec)
say(f"    explicit three-label rule: value {v3} = {float(v3):.9f} (exact)")
mm = {}
for lab, nu in dec.items():
    for x in nu: mm.setdefault(inst.M[x], set()).add(lab)
say(f"    labels used per value of M: { {float(m): len(v) for m, v in sorted(mm.items())} }  (>1 means not M-measurable)")
say(f"    separation: exact adversary optimum <= {float(v3):.9f} < {float(lb2):.9f} <= every M-measurable rule; "
    f"gap {float(lb2 - v3):.6f}")
say(f"    the explicit rule is optimal to within {float(v3 - lb):.2e} (exact lower bound {float(lb):.9f})")

# ---------------------------------------------------------------- explicit certificates (printed in the paper)
idx = {v: i for i, v in enumerate(inst.sup)}
def Aset(*vals): return frozenset(idx[Fr(v)] for v in vals)
def key(S): return tuple(sorted(int(m) for m in S))
def rq(q, x): return sum(p*inst.r(a, x) for a, p in q.items())
qa = {(1,): {Aset(): 1}, (1, 2): {Aset(): 1}, (8,): {Aset(): 1},
      (1, 2, 4): {Aset(4): 1}, (2, 4): {Aset(4): 1}, (2, 4, 8): {Aset(4): 1},
      (4, 8): {Aset(4, 8): 1}, (8, 32): {Aset(32): 1}, (32,): {Aset(32): 1}}
env_a = sum(inst.P[x]*min(rq(qa[key(S)], x) for S in inst.slabs if inst.M[x] in S) for x in inst.X)/inst.EM
qb = dict(qa)
qb[(2, 4, 8)] = {Aset(4): Fr(13, 17), Aset(4, 8): Fr(4, 17)}
qb[(4, 8)] = {Aset(4): Fr(13, 17), Aset(4, 8): Fr(4, 17)}
qb[(8,)] = {Aset(1, 2, 8): 1}
env_b = sum(min(sum(inst.P[x]*rq(qb[key(S)], x) for x in inst.X if inst.M[x] == m)
                for S in inst.slabs if m in S) for m in inst.Mvals)/inst.EM
say(f"    explicit deterministic family (a): envelope = {env_a} (exact)")
say(f"    explicit family (b), rules depending on M only: envelope = {env_b} (exact); best deterministic "
    f"M-measurable rule = {bestdet}")
assert env_a == Fr(349, 371) and env_b == Fr(353, 371) and bestdet == Fr(353, 371)

# ---------------------------------------------------------------- the natural-family breach (c = 5)
inst5 = Instance([0, Fr('5.4443'), Fr('18.6798'), Fr('26.0801'), Fr('43.5019')],
                 [Fr('0.2630'), Fr('0.6181'), Fr('0.1048'), Fr('0.0029'), Fr('0.0111')], 5)
report("natural-family breach", inst5, "0.797303 (certified, Theorem 5.5)")

open('obstruction_exact.txt', 'w').write("\n".join(out) + "\n")
