"""Rigorous enclosures of rho(c) at given widths c0: a 2D Krawczyk test for the system
      Gamma(y1, r) = 0,   c(y1, r) = c0
on a small box X around a float solution.  If K(X) is inside int X, the system has exactly one solution in X,
an Euler-Lagrange point at width c0.  The script then checks that the upper enclosure of rho_hat on K(X) is
below the threshold rho_A = 1/(1+eta) of a piece [a, b] of the certification containing c0 (thresholds read
from the certification runs cert_*.json.gz).  By Theorem 5.5(ii) every Euler-Lagrange point at width c0 off the curve C has
J > rho_A, so the isolated point is the point of C at width c0, and rho(c0) = rho_hat there by Theorem 5.5(i)."""
import sys, gzip, json, numpy as np
from ivec import I
import planar_iv as P
from planar_box import evaluate_box

def float_solve(c0, curve):
    rg, yg, cg = curve[:, 0], curve[:, 1], curve[:, 2]
    r = float(np.interp(c0, cg, rg)); y = float(np.interp(c0, cg, yg))
    for it in range(40):
        E = P.evaluate(I(np.array([y])), I(np.array([r])), derivs=True)
        F = np.array([E['Gamma'].mid[0], E['c'].mid[0] - c0])
        Jm = np.array([[E['G_y'].mid[0], E['G_r'].mid[0]], [E['c_y'].mid[0], E['c_r'].mid[0]]])
        d = np.linalg.solve(Jm, -F); y += d[0]; r += d[1]
        if np.max(np.abs(d)) < 1e-16: break
    return y, r, np.linalg.inv(Jm)

def krawczyk2(c0, y, r, C, rad):
    X = [I(np.array([y - rad[0]]), np.array([y + rad[0]])), I(np.array([r - rad[1]]), np.array([r + rad[1]]))]
    Vm = P.evaluate(I(np.array([y])), I(np.array([r])))
    from planar_box import assemble
    assemble(Vm)
    Fm = [Vm['Gamma'], Vm['c'] - c0]
    V, G, ok, Cc = evaluate_box(X[0], X[1])
    J = [[G['Gamma'][0], G['Gamma'][1]], [G['c'][0], G['c'][1]]]
    dX = [X[0] - y, X[1] - r]
    K = []
    for i in range(2):
        acc = I(np.array([[y, r][i]])) - (Fm[0]*C[i, 0] + Fm[1]*C[i, 1])
        for j in range(2):
            Mij = (1.0 if i == j else 0.0) - (J[0][j]*C[i, 0] + J[1][j]*C[i, 1])
            acc = acc + Mij*dX[j]
        K.append(acc)
    inside = all(bool(K[i].lo[0] > X[i].lo[0] and K[i].hi[0] < X[i].hi[0]) for i in range(2)) \
        and bool(ok[0]) and bool(Vm['ok'][0])
    if not inside: return False, K, None
    VK, GK, okK, CK = evaluate_box(K[0], K[1])
    return bool(okK[0]), K, VK

if __name__ == '__main__':
    import glob
    curves = [np.load(f) for f in ('curve_float.npy', 'curve_float_ext.npy')]
    pieces = []
    for f in sorted(glob.glob('cert_*.json.gz')):                 # every certification run: its certified pieces
        d = json.load(gzip.open(f))
        pieces += [q for q in d['pieces'] if q['certified']]
    cs = [float(x) for x in sys.argv[1:]] or [1.0001, 1.001, 1.01, 1.1, 1.25, 1.5, 2, 2.2, 3, 4, 5, 10, 20, 50,
                                              100, 300, 1000, 2000, 3000]
    for c0 in cs:
        curve = next(cv for cv in curves if cv[0, 2] < c0 < cv[-1, 2])
        y, r, C = float_solve(c0, curve)
        for rel in (1e-9, 1e-8, 1e-7, 1e-6):
            ok, K, VK = krawczyk2(c0, y, r, C, (rel*y, rel*r))
            if ok: break
        used_rel = rel
        if not ok: print(f"{c0:7g}  FAILED"); continue
        rho = VK['rho']
        # outward rounding to 11 decimals, done exactly with Fraction
        from fractions import Fraction
        import math
        lo = Fraction(float(rho.lo[0])); hi = Fraction(float(rho.hi[0]))
        lo11 = math.floor(lo*10**11); hi11 = math.ceil(hi*10**11)
        # on C: rho_hat.hi * (1 + eta) < 1 for some piece [a, b] containing c0, in interval arithmetic
        onC = [(p['a'], p['b'], p['eta']) for p in pieces if p['a'] <= c0 <= p['b']
               and bool(((I(np.array([rho.hi[0]]))*(I(1.0) + I(float(p['eta'])))).hi < 1.0)[0])]
        tag = (f"on C: rho_hat <= rho_A of [{onC[0][0]:g}, {onC[0][1]:g}] "
               f"(margin {1/(1 + onC[0][2]) - rho.hi[0]:.1e})") if onC else "NOT SHOWN ON C"
        print(f"{c0:7g}  rho in [0.{lo11:011d}, 0.{hi11:011d}]   (width {rho.wid[0]:.1e})   "
              f"box rel. size {used_rel:.0e}   {tag}")
