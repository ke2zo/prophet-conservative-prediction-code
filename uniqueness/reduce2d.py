"""2D reduction of the full EL system: parameters (y1, r) with r = R1/R0.
(TS) gives S0 = -ln(y1 psi(r)); T1 gives R0 = q = S0/(1-r), tau = D(y1) = q y1, kappa = r q^2 y1.
The body is R(t) = kappa/D(t), D(t) = tau + Phi(y1) - Phi(t), t in [y1, yM].
(T0) K1(yM) = 1 + ln(r)/S0 fixes yM; the multiplier relation with c eliminated is Gamma = 0;
then c = M* + kappa*rho/eps, rho = 1/D(yM)."""
import math, numpy as np
from scipy import optimize
XI, WXI = np.polynomial.legendre.leggauss(200); XI = 0.5*(XI+1); WXI = 0.5*WXI
def Phi(t): return t*np.log(t) - t
def psi(r): return r*(r - 1 - math.log(r))/(1 - r)**2
def reduce(y1, r):
    ps = psi(r); S0 = -math.log(y1*ps)
    if S0 <= 0: return None
    q = S0/(1 - r); tau = q*y1; kap = r*q*q*y1
    target = 1 + math.log(r)/S0
    P1 = Phi(y1)
    def K1(yM):
        t = y1 + (yM - y1)*XI
        return (yM - y1)*np.dot(WXI, 1.0/(tau + P1 - Phi(t)))
    if target <= 0 or K1(1 - 1e-15) <= target: return None
    yM = optimize.brentq(lambda z: K1(z) - target, y1 + 1e-15, 1 - 1e-15, xtol=1e-15, rtol=1e-15)
    t = y1 + (yM - y1)*XI; D = tau + P1 - Phi(t); W = yM - y1
    K2 = W*np.dot(WXI, 1/D**2); K3 = W*np.dot(WXI, t/D**2)
    DM = tau + P1 - Phi(yM); rho = 1/DM; eps = -math.log(yM)
    M = 1 + kap*K2; y0 = y1*ps
    Gam = (kap*rho/eps)*(1 - yM) + M - y0 - kap*K3 - M/rho
    c = M + kap*rho/eps
    return dict(Gamma=Gam, c=c, rho=rho, M=M, eps=eps, kap=kap, S0=S0, yM=yM, y0=y0, tau=tau, q=q)
if __name__ == '__main__':
    # check against the solved extremal at c = 2: y1 = 0.23648002, S0* = 2.9313937
    y1 = 0.23648002; S0 = 2.931393692579283
    r = optimize.brentq(lambda rr: psi(rr) - math.exp(-S0)/y1, 1e-9, 1 - 1e-9)
    d = reduce(y1, r)
    print("r =", r, {k: round(float(v), 8) for k, v in d.items()})
