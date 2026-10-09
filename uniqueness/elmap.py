"""Global map of the Euler-Lagrange critical set (exploratory).
Constrained system G(v;S0)=0, v=(eps,M,kappa,y1), as in the paper's App. C.6 (legc linearization):
body from the exact first integral, Gauss-Legendre quadrature in xi on [y1, yM].
Outer function Delta(S0) = psi(R1/R0) - y0/y1 (sign g' = sign Delta, Lemma 19)."""
import math, sys, numpy as np
from scipy import optimize
LOG, EXP = math.log, math.exp
XI, WXI = np.polynomial.legendre.leggauss(400)
XI = 0.5*(XI + 1.0); WXI = 0.5*WXI

def Phi(t): return t*np.log(t) - t
def psi(r): return r*(r - 1 - math.log(r))/(1 - r)**2

def pieces(v, S0, c):
    eps, M, kap, y1 = v
    yM = EXP(-eps); RM = eps*(c - M); W = yM - y1
    y = y1 + W*XI
    invR = 1.0/RM + (Phi(yM) - Phi(y))/kap
    R = 1.0/invR
    R1 = 1.0/(1.0/RM + (Phi(yM) - Phi(y1))/kap)
    R0 = S0 + R1; y0 = EXP(-S0)
    A1 = W*np.dot(WXI, R); A2 = W*np.dot(WXI, R**2); A3 = W*np.dot(WXI, y*R**2)
    Em = c - y0 - A3/kap - yM*(c - M)
    return yM, RM, R1, R0, y0, A1, A2, A3, Em, invR

def G(v, S0, c):
    eps, M, kap, y1 = v
    if not (eps > 0 and 1 < M < c and kap > 0 and 0 < y1 < EXP(-eps)):
        return np.array([1e3]*4)
    yM, RM, R1, R0, y0, A1, A2, A3, Em, invR = pieces(v, S0, c)
    if np.any(invR <= 0) or R1 <= 0 or Em <= 0:
        return np.array([1e3]*4)
    return np.array([kap*(M - 1) - A2,
                     (1.0/S0)*LOG(R0/R1) + A1/kap - 1.0,
                     kap - y1*R1*R0,
                     kap*M - RM*Em])

def csolve(S0, c, seed):
    s = optimize.fsolve(G, seed, args=(S0, c), full_output=True, xtol=1e-13)
    v = s[0]
    if s[2] != 1 or np.max(np.abs(G(v, S0, c))) > 1e-10: return None
    return v

def outer(v, S0, c):
    yM, RM, R1, R0, y0, A1, A2, A3, Em, invR = pieces(v, S0, c)
    eps, M, kap, y1 = v
    return psi(R1/R0) - y0/y1, M/Em, S0 - (-LOG(y1))   # Delta, g, atom-at-1 rate

def full_solve(c, seed_v, seed_S0):
    def F(z):
        v, S0 = z[:4], z[4]
        g = G(v, S0, c)
        return list(g) + [outer(v, S0, c)[0] if g[0] < 1e2 else 1e3]
    s = optimize.fsolve(F, list(seed_v) + [seed_S0], full_output=True, xtol=1e-13)
    z = s[0]
    if s[2] != 1 or max(abs(x) for x in F(z)) > 1e-10: return None
    return z[:4], z[4]

def trace(c, v0, S00, direction, step=0.02, S_max=30.0, max_steps=4000):
    """Continuation of the constrained root in S0 from (v0,S00); returns list of (S0, v)."""
    out = []; v = np.array(v0, float); S0 = S00; h = step
    for _ in range(max_steps):
        S1 = S0 + direction*h
        if S1 <= 1e-3 or S1 > S_max: break
        w = csolve(S1, c, v)
        if w is None:
            h /= 2
            if h < 1e-7: break
            continue
        out.append((S1, w)); S0, v = S1, w
        h = min(step, h*1.5)
    return out

if __name__ == '__main__':
    import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
    from el_shoot_solver import solve, ev
    c0 = 2.0
    rho, r = solve(c0)
    rr = ev(r['M'], r['eps'], r['kappa'], r['S0'], c0, 200000)
    v = np.array([r['eps'], r['M'], r['kappa'], rr['y1p']]); S0 = r['S0']
    print("seed at c=2:", v, S0, "rho", rho)
    fs = full_solve(c0, v, S0); print("full solve:", fs)
