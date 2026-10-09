"""Independent floating-point reference evaluator of the planar reduction (used only by the self-checks).
Integrals over [y1, y_M] are computed in the variable u = -ln t by composite 40-point Gauss-Legendre,
and y_M by Brent's method; no code is shared with the interval evaluator."""
import math, numpy as np
from scipy import optimize
g, gw = np.polynomial.legendre.leggauss(40)
def psi(r):
    return r*(r - 1 - math.log(r))/(1 - r)**2
def phi(t): return t*(1 - math.log(t))
def quad_u(f, a, b, npan=None):
    # composite GL on [a,b] in u
    if npan is None: npan = max(4, int(math.ceil((b - a)/0.5)))
    npan = min(npan, 4000)
    edges = np.linspace(a, b, npan + 1)
    tot = 0.0
    for k in range(npan):
        lo, hi = edges[k], edges[k+1]
        u = 0.5*(hi - lo)*g + 0.5*(hi + lo)
        tot += 0.5*(hi - lo)*np.dot(gw, f(u))
    return tot
def evalp(y1, r):
    L = -math.log(y1); ps = psi(r); S0 = L - math.log(ps)
    if S0 <= -math.log(r): return dict(feas=False, why='T*<=0')
    Ts = 1 + math.log(r)/S0
    q = S0/(1 - r); tau = q*y1; kap = r*q*q*y1
    dl = tau - phi(y1)          # D(t) = dl + phi(t)
    if tau*Ts >= 1 - y1: return dict(feas=False, why='tau*T*>=1-y1', tau=tau, Ts=Ts)
    Dfun = lambda u: dl + np.exp(-u)*(1 + u)
    f1 = lambda u: np.exp(-u)/Dfun(u)
    def K1(z):  # int_{y1}^{z} dt/D = int_{-ln z}^{L} ...
        return quad_u(f1, -math.log(z), L)
    if K1(1 - 1e-16) <= Ts: return dict(feas=False, why='K1(1)<=T*', tau=tau, Ts=Ts)
    # solve in uM = -ln yM
    uM = optimize.brentq(lambda uu: quad_u(f1, uu, L) - Ts, 1e-16, L*(1 - 1e-15), xtol=1e-15, rtol=1e-14)
    yM = math.exp(-uM)
    K2 = quad_u(lambda u: np.exp(-u)/Dfun(u)**2, uM, L)
    K3 = quad_u(lambda u: np.exp(-2*u)/Dfun(u)**2, uM, L)
    DM = dl + phi(yM); rho = 1/DM; eps = uM
    M = 1 + kap*K2; Qv = kap*rho/eps; y0 = y1*ps
    c = M + Qv; Gam = Qv*(1 - yM) + M*(1 - DM) - y0 - kap*K3
    return dict(feas=True, c=c, Gamma=Gam, rho=rho, yM=yM, tau=tau, dl=dl, Ts=Ts, kap=kap, M=M, S0=S0, eps=eps)
