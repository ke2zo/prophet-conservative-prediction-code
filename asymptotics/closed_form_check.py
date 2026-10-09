"""Checks of Lemma 5.3 (closed form of Gamma) and of the gap formula (c), and the scaling of the
terms along the continuation branch.  Uses the thin interval evaluator of ../certify (midpoints)."""
import os, sys, math, numpy as np
from scipy import integrate, optimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from ivec import I, ilog, iexp
import planar_iv as P
from planar_box import assemble

phi = lambda t: t*(1 - math.log(t)) if t > 0 else 0.0
K = optimize.brentq(lambda K: integrate.quad(lambda t: 1/(K - 1 + phi(t)), 0, 1, limit=200,
                    epsabs=1e-15, epsrel=1e-15)[0] - 1, 1.2, 1.6, xtol=1e-15)
print(f"Kertz: K = 1/beta = {K:.13f}, beta = {1/K:.13f}")

# (b) Gamma = c theta(eps) - y1 g(psi(r)) at random points of F
rng = np.random.default_rng(5)
y = np.exp(rng.uniform(math.log(1e-4), math.log(0.6), 4000)); r = 1/(1 + np.exp(-rng.uniform(-9, 3, 4000)))
V = P.evaluate(I(y), I(r)); assemble(V); ok = V['ok']
theta = 1.0 - (1.0 + V['eps'])*iexp(-V['eps']); g = V['psi'] - 1.0 - ilog(V['psi'])
alt = V['c']*theta - I(y)*g
d = np.abs(V['Gamma'].mid - alt.mid)[ok]
inter = (np.maximum(V['Gamma'].lo, alt.lo) <= np.minimum(V['Gamma'].hi, alt.hi))[ok]
print(f"(b) {int(ok.sum())} random points of F: max |Gamma - (c theta - y1 g)| = {d.max():.1e}; "
      f"interval enclosures intersect everywhere: {bool(inter.all())}")

# (c) gap formula and scaling along the branch
curve = np.load('../certify/curve_float.npy')
print("(c) along the continuation branch:")
print(f"{'c':>8} {'y1':>8} {'1-r':>7} {'eps':>8} {'gap':>9} {'gap/y1^2':>8} {'eps/y1^2':>8} "
      f"{'B/y1^2':>7} {'W':>6} {'formula-gap':>11}")
for cc in (2, 5, 10, 30, 100, 300, 1000, 1500):
    k = np.argmin(abs(curve[:, 2] - cc)); rr, y1 = curve[k, 0], curve[k, 1]
    V = P.evaluate(I(np.array([y1])), I(np.array([rr]))); assemble(V)
    eps = V['eps'].mid[0]; yM = math.exp(-eps); delta = V['b'].mid[0]; S0 = V['S0'].mid[0]
    gap = K - 1/V['rho'].mid[0]
    I0 = integrate.quad(lambda t: 1/(K - 1 + phi(t)), 0, y1, epsabs=1e-15)[0]
    I1 = integrate.quad(lambda t: 1/(K - 1 + phi(t)), yM, 1, epsabs=1e-15)[0]
    W = integrate.quad(lambda t: 1/((delta + phi(t))*(K - 1 + phi(t))), y1, yM, limit=200, epsabs=1e-15)[0]
    th = 1 - (1 + eps)*math.exp(-eps); B = I0 + math.log(rr)/S0
    print(f"{V['c'].mid[0]:8.2f} {y1:8.5f} {1-rr:7.4f} {eps:8.2e} {gap:9.3e} {gap/y1**2:8.4f} {eps/y1**2:8.4f} "
          f"{B/y1**2:7.3f} {W:6.4f} {th + (I0 + I1 + math.log(rr)/S0)/W - gap:+11.1e}")
