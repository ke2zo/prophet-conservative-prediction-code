"""Checks for the separation theorem  rho(c) >= c/(2c-1) + (1-1/c)^2/400  (Theorem thm:gap).

1. Symbolic: the identities (A), (B), the decomposition of B-frak, and the formula for B1 - tau.
2. Two-point closed forms of E, L0, L1 against direct quadrature of their definitions.
3. The numerical constants of the proof, with interval arithmetic (mpmath.iv).
4. Brute force over (c, S0): the minimum over u* of the two-point ratio of the policy used
   by the proof, against tau_c + Delta_c (c - 1 log-uniform, S0 clustered at the regime boundary).
5. Monte Carlo in the Poisson model: the level policy Pi(u1, t1) run through the thinning
   rule of Lemma lem:levelpol reproduces the closed form L1, and L1 <= A(s).
"""
import math, random
import numpy as np
import sympy as sp
import mpmath as mp

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ---------------------------------------------------------------- 1. symbolic identities
z, S = sp.symbols('z S0', positive=True)
E1 = sp.exp(-1)
c = (z - 1)/(z - 2)
tau = 1 - 1/z
y0 = sp.exp(-S)
xi = sp.exp(sp.log(z)/S - 1)
p1 = 1 - xi
q2 = xi*(1 - 1/z)
# q2 from its definition: e^{-lambda t1}(1 - e^{-S0(1-t1)}) with lambda = 1, t1 = 1 - ln z/S0
t1 = 1 - sp.log(z)/S
say("q2 definition - xi*tau :", sp.simplify(sp.exp(-t1)*(1 - sp.exp(-S*(1 - t1))) - q2))
rhat = (p1*c + q2*(1 + (c - 1)/S))/(c*(1 - y0))
A_rhs = (1 + (z - 1)*sp.exp(-S) - xi*(2 - 1/S))/(z*(1 - y0))
say("identity (A) residual  :", sp.simplify(rhat - tau - A_rhs))
dC = c - y0 - (c - 1)*E1
rC = ((p1 + q2) + (c - 1)*(1 - E1)*(p1 + q2/S))/dC
Bfrak = (2 - E1)*z - 2 - (1 - E1)*(z - 1)/S
B_rhs = ((z - 1 - E1) + (z - 1)*(z - 2)*sp.exp(-S) - xi*Bfrak)/(z*(z - 2)*dC)
say("identity (B) residual  :", sp.simplify(rC - tau - B_rhs))
say("B-frak decomposition   :", sp.simplify(Bfrak - ((1 - E1)*(2 - 1/S) + (z - 2)*((2 - E1) - (1 - E1)/S))))
cc = sp.symbols('c', positive=True)
yy = sp.symbols('y0', positive=True)
B1 = (1 + (cc - 1)*(1 - yy)/S)/cc
zc = (2*cc - 1)/(cc - 1)
say("B1 - tau formula       :", sp.simplify(B1 - cc/(2*cc - 1) - (cc - 1)/cc*((1 - yy)/S - 1/zc)))
say("c, c-1, (c-1)/c in z   :", sp.simplify(zc.subs(cc, c) - z), sp.simplify(c - 1 - 1/(z - 2)),
    sp.simplify((c - 1)/c - 1/(z - 1)))
# the four endpoint values r_A, r_B, r_C, r_D and the dominations used in the proof
rA = (p1*c + q2*(1 + (c - 1)*(2 - sp.E*y0)/S))/(c*(1 - y0))
rB = (p1*c + q2*(1 + (c - 1)/S))/dC
rD = (p1 + q2)/(1 - y0)

# ---------------------------------------------------------------- 2. two-point closed forms
mp.mp.dps = 30
def two_point_forms(cv, S0, us, u1, t1v):
    """E, L0, L1 for Q = 1 + (c-1) 1[u > u*], from the definitions (quadrature)."""
    y0v = mp.e**(-S0)
    Q = lambda u: 1 + (cv - 1)*(1 if u > us else 0)
    Ed = mp.quad(Q, [y0v, us, 1])
    I = lambda lo: mp.quad(lambda v: Q(v)/v, [lo, max(lo, us), 1]) if us > lo else mp.quad(lambda v: Q(v)/v, [lo, 1])
    L0d = (1 - y0v)/S0*I(y0v)
    lam = -mp.log(u1)
    p1v = 1 - mp.e**(-lam*t1v); q2v = mp.e**(-lam*t1v)*(1 - mp.e**(-S0*(1 - t1v)))
    L1d = p1v/lam*I(u1) + q2v/S0*I(y0v)
    a = -mp.log(us)
    Ec = cv - y0v - (cv - 1)*us
    L0c = (1 - y0v)*(1 + (cv - 1)*a/S0)
    L1c = p1v*(1 + (cv - 1)*min(a, lam)/lam) + q2v*(1 + (cv - 1)*a/S0)
    return max(abs(Ed - Ec), abs(L0d - L0c), abs(L1d - L1c))
rng = random.Random(1)
worst = 0
for _ in range(300):
    cv = mp.mpf(1 + 20*rng.random()); S0 = mp.mpf(0.1 + 8*rng.random())
    y0v = mp.e**(-S0)
    us = y0v + (1 - y0v)*mp.mpf(rng.random())
    u1 = y0v + (1 - y0v)*mp.mpf(rng.random()); t1v = mp.mpf(rng.random())
    worst = max(worst, two_point_forms(cv, S0, us, u1, t1v))
say(f"two-point closed forms vs quadrature, 300 random cases: max error {mp.nstr(worst, 3)}")

# ---------------------------------------------------------------- 3. constants (interval arithmetic)
iv = mp.iv; iv.dps = 30
e = iv.e
say("1 - 4 e^{-3/2}           =", iv.mpf(1) - 4*iv.exp(-iv.mpf(1.5)), " (> 0.107)")
say("3(1-ln2) e^{3/2}         =", 3*(1 - iv.log(2))*iv.exp(iv.mpf(1.5)), " (> 4)")
dA = 1 - iv.exp(iv.log(2) + iv.mpf(4)/3*iv.exp(-iv.mpf(1.5)) - 1)
say("delta_A                  =", dA, " (> 0.009)")
say("(2e-1) e^{4/(3e)} - e^2  =", (2*e - 1)*iv.exp(4/(3*e)) - e**2, " (< 0)")
qB = (2 - 1/e)*iv.exp(4/(3*e) - 1)
say("q_B                      =", qB, " (< 1)")
say("(1-1/e) delta_A          =", (1 - 1/e)*dA, " (>= 0.00588)")
say("max_z (ln z - 1/2)/z at z = e^{3/2}: value e^{-3/2} =", iv.exp(-iv.mpf(1.5)))
say("min_{z>0} (3z/4 - ln z) = 1 - ln(4/3) =", 1 - iv.log(iv.mpf(4)/3), " (> 0, so S0 > ln z)")
# the four final polynomial inequalities on z >= 2 (each side is linear or convex, checked at z = 2 with slopes)
say("regime 1: 400(z-1)^2 - z at z=2:", 400 - 2, "; 14.2(z-1) - z at z=2:", 14.2 - 2, "(slopes 400(2z-2)-1 > 0, 13.2 > 0)")
say("regime 2: 3.6(z-1)^2 - z at z=2:", 3.6 - 2, "; 2.35(z-1) - z at z=2:", 2.35 - 2, "(slopes > 0)")
say("400*0.107/3 =", 400*0.107/3, "; 400*0.009 =", 400*0.009, "; 400*0.00588 =", 400*0.00588)

# ---------------------------------------------------------------- 4. brute force over (c, S0)
def ratio_min(cv, S0, regime):
    """min over u* in [y0, 1] of L/E on two-point instances, for the policy of the regime.
    Grid in a = ln(1/u*) in [0, S0]: dense near 0 (linear and logarithmic), the kink a = 1, and a = S0."""
    y0v = math.exp(-S0)
    a = np.concatenate([np.linspace(0, min(S0, 12.0), 3001), np.geomspace(1e-9, S0, 3001), [1.0, S0]])
    a = a[(a >= 0) & (a <= S0)]
    us = np.exp(-a)
    Ev = cv - y0v - (cv - 1)*us
    if regime == 1:
        L = (1 - y0v)*(1 + (cv - 1)*a/S0)
    else:
        zz = (2*cv - 1)/(cv - 1)
        t1v = 1 - math.log(zz)/S0
        p1v = 1 - math.exp(-t1v); q2v = math.exp(-t1v)*(1 - math.exp(-S0*(1 - t1v)))
        L = p1v*(1 + (cv - 1)*np.minimum(a, 1.0)) + q2v*(1 + (cv - 1)*a/S0)
    return float(np.min(L/Ev))
worst_margin = (1e9, None)
nviol = 0; ntot = 0
for cm1 in np.geomspace(1e-5, 1e5, 121):             # c - 1 log-uniform, so that c near 1 is well sampled
    cv = 1 + cm1
    zz = (2*cv - 1)/(cv - 1)
    Delta = (1 - 1/cv)**2/400
    target = cv/(2*cv - 1) + Delta
    S0s = np.concatenate([np.geomspace(1e-3, 1e7, 400),
                          [0.75*zz*(1 + e) for e in (-1e-2, -1e-4, -1e-6, -1e-9, 0.0, 1e-9, 1e-6, 1e-4, 1e-2)]])
    for S0 in S0s:
        regime = 1 if 4*S0 < 3*zz else 2
        r = ratio_min(cv, S0, regime)
        ntot += 1
        m = (r - target)/Delta
        if r < target - 1e-15: nviol += 1
        if m < worst_margin[0]: worst_margin = (m, (cv, S0, regime, r, target))
phi1 = lambda x: (1 - math.exp(-x))/x
say(f"brute force: {ntot} (c,S0) points (c-1 log-uniform in [1e-5,1e5], S0 up to 1e7 and clustered at the "
    f"regime boundary 3z/4), {nviol} violations; smallest (ratio - bound)/Delta_c = {worst_margin[0]:.3f} at "
    f"c-1={worst_margin[1][0]-1:.3g}, S0={worst_margin[1][1]:.6g}, regime {worst_margin[1][2]}")
say(f"limit c -> infinity at S0 = 3z/4 from below: 400(phi_1(3/2) - 1/2) - 1 = {400*(phi1(1.5) - 0.5) - 1:.4f}")

# ---------------------------------------------------------------- 5. Monte Carlo in the Poisson model
def mc_check(levels, rates, u1, t1v, N=4000000, seed=7):
    """Instance with atoms at 'levels' (in [1,c]) with Poisson rates 'rates'. Direct simulation of the
    Poisson model: K ~ Poi(S0) arrivals at uniform times with values drawn from rates/S0; each arrival
    gets an independent level, log-uniform on the flat (y(x_j-), y(x_j)] of its value. The level
    policy accepts the first arrival with (t < t1 and level >= u1) or t >= t1."""
    levels = np.asarray(levels, float); rates = np.asarray(rates, float)
    S0 = rates.sum()
    tail = np.concatenate([np.cumsum(rates[::-1])[::-1][1:], [0.0]])   # s(x_j) = sum_{i>j} rates_i
    y_hi = np.exp(-tail); y_lo = np.exp(-(tail + rates))                 # flat of x_j: (y_lo, y_hi]
    lam = -math.log(u1)
    rng = np.random.default_rng(seed)
    tot = 0.0; tot2 = 0.0; B = 20000; Kmax = 40
    for _ in range(N//B):
        K = rng.poisson(S0, B)
        t = rng.random((B, Kmax)); t[np.arange(Kmax)[None, :] >= K[:, None]] = np.inf
        j = rng.choice(len(levels), (B, Kmax), p=rates/S0)
        v = np.exp(np.log(y_lo[j]) + rng.random((B, Kmax))*(np.log(y_hi[j]) - np.log(y_lo[j])))
        ok = np.isfinite(t) & (((t < t1v) & (v >= u1)) | (t >= t1v))
        tt = np.where(ok, t, np.inf)
        k = np.argmin(tt, axis=1); got = np.isfinite(tt[np.arange(B), k])
        val = levels[j[np.arange(B), k]]*got
        tot += val.sum(); tot2 += (val**2).sum()
    Nt = N//B*B
    Lmc = tot/Nt
    se = math.sqrt(max(tot2/Nt - Lmc**2, 0)/Nt)
    def intQ(lo):
        return float(sum(levels[i]*max(0.0, math.log(y_hi[i]/max(y_lo[i], lo))) for i in range(len(levels))))
    p1v = 1 - math.exp(-lam*t1v); q2v = math.exp(-lam*t1v)*(1 - math.exp(-S0*(1 - t1v)))
    L1 = p1v/lam*intQ(u1) + q2v/S0*intQ(math.exp(-S0))
    def R(w):
        return float(sum(rates[i]*max(0.0, levels[i] - w) for i in range(len(levels))))
    def T(a, K=4000):
        ws = np.linspace(0, a, K + 1); f = 1/np.array([R(w) for w in ws])
        return float(np.sum((f[1:] + f[:-1])/2)*(a/K))
    lo, hi = 0.0, levels.max()*(1 - 1e-9)
    for _ in range(50):
        mid = (lo + hi)/2
        (lo, hi) = (mid, hi) if T(mid) < 1 else (lo, mid)
    return Lmc, se, L1, lo
for levels, rates, u1, t1v in [((1.0, 2.5, 4.0), (0.9, 0.5, 0.3), 0.4, 0.5),
                               ((1.0, 1.7, 3.0, 6.0), (1.5, 0.4, 0.3, 0.2), 1/math.e, 0.7)]:
    Lmc, se, L1, A = mc_check(levels, rates, u1, t1v)
    say(f"MC levels={levels} rates={rates} u1={u1:.4f} t1={t1v}: simulated {Lmc:.5f} (s.e. {se:.5f}), "
        f"closed form {L1:.5f}, A(s) = {A:.5f}")

open('gap_check.txt', 'w').write("\n".join(out) + "\n")
