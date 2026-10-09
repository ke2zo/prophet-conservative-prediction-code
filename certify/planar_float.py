"""Planar reduction (manuscript, Proposition 5.1) in floating point, with analytic first derivatives in (y1, r).
Used to validate the derivative formulas that the interval certifier (planar_iv.py) implements.
All integrals over [y1, yM] of f_j(t) with D(t) = b - Phi(t), b = tau + Phi(y1):
   K1 = int 1/D, K2 = int 1/D^2, K3 = int t/D^2, K4 = int 1/D^3, K5 = int t/D^3."""
import math, numpy as np
from scipy import optimize
XI, WXI = np.polynomial.legendre.leggauss(300); XI = 0.5*(XI + 1); WXI = 0.5*WXI
def Phi(t): return t*np.log(t) - t
def psi(r): return r*(r - 1 - math.log(r))/(1 - r)**2
def dpsi(r): return (-2*(1 - r) - (1 + r)*math.log(r))/(1 - r)**3

def ints(y1, z, b):
    t = y1 + (z - y1)*XI; D = b - Phi(t); w = (z - y1)*WXI
    return (np.dot(w, 1/D), np.dot(w, 1/D**2), np.dot(w, t/D**2), np.dot(w, 1/D**3), np.dot(w, t/D**3))

def evaluate(y1, r, derivs=True):
    ps = psi(r); S0 = -math.log(y1*ps); Ts = 1 + math.log(r)/S0
    q = S0/(1 - r); tau = q*y1; kap = r*q*q*y1; P1 = y1*math.log(y1) - y1; b = tau + P1
    if Ts <= 0: return None
    K1max = ints(y1, 1 - 1e-15, b)[0]
    if K1max <= Ts: return None
    yM = optimize.brentq(lambda z: ints(y1, z, b)[0] - Ts, y1*(1 + 1e-15), 1 - 1e-15, xtol=1e-16, rtol=1e-15)
    K1, K2, K3, K4, K5 = ints(y1, yM, b)
    DM = b - (yM*math.log(yM) - yM); eps = -math.log(yM); y0 = y1*ps
    M = 1 + kap*K2; Qv = kap/(eps*DM)
    c = M + Qv; Gam = Qv*(1 - yM) + M*(1 - DM) - y0 - kap*K3
    out = dict(c=c, Gamma=Gam, rho=1/DM, yM=yM, DM=DM, eps=eps, M=M, S0=S0, tau=tau, kap=kap)
    if not derivs: return out
    dps = dpsi(r)
    S0y, S0r = -1/y1, -dps/ps
    Tsy, Tsr = -math.log(r)*S0y/S0**2, 1/(r*S0) - math.log(r)*S0r/S0**2
    qy, qr = S0y/(1 - r), S0r/(1 - r) + S0/(1 - r)**2
    tauy, taur = qy*y1 + q, qr*y1
    kapy, kapr = r*(2*q*qy*y1 + q*q), q*q*y1 + 2*r*q*qr*y1
    by, br = tauy + math.log(y1), taur
    yMy, yMr = DM*(Tsy + 1/tau + K2*by), DM*(Tsr + K2*br)
    DMy, DMr = by - math.log(yM)*yMy, br - math.log(yM)*yMr
    epsy, epsr = -yMy/yM, -yMr/yM
    K2y, K2r = yMy/DM**2 - 1/tau**2 - 2*K4*by, yMr/DM**2 - 2*K4*br
    K3y, K3r = yM*yMy/DM**2 - y1/tau**2 - 2*K5*by, yM*yMr/DM**2 - 2*K5*br
    My, Mr = kapy*K2 + kap*K2y, kapr*K2 + kap*K2r
    y0y, y0r = ps, y1*dps
    Qy, Qr = Qv*(kapy/kap - epsy/eps - DMy/DM), Qv*(kapr/kap - epsr/eps - DMr/DM)
    out.update(c_y=My + Qy, c_r=Mr + Qr,
               G_y=Qy*(1 - yM) - Qv*yMy + My*(1 - DM) - M*DMy - y0y - kapy*K3 - kap*K3y,
               G_r=Qr*(1 - yM) - Qv*yMr + Mr*(1 - DM) - M*DMr - y0r - kapr*K3 - kap*K3r,
               rho_y=-DMy/DM**2, rho_r=-DMr/DM**2)
    return out

if __name__ == '__main__':
    worst = 0.0
    for (y1, r) in [(0.2364800, 0.1605454), (0.13278, 0.4368), (0.05, 0.3), (0.3, 0.2), (0.01, 0.5), (0.2, 0.05)]:
        e = evaluate(y1, r)
        if e is None: print((y1, r), "infeasible"); continue
        h = 1e-6
        for var, k in (('y', 0), ('r', 1)):
            p = [y1, r]; m = [y1, r]; p[k] += h*[y1, r][k]; m[k] -= h*[y1, r][k]
            ep, em = evaluate(*p, derivs=False), evaluate(*m, derivs=False)
            for f in ('c', 'Gamma', 'rho'):
                fd = (ep[f] - em[f])/(2*h*[y1, r][k])
                an = e[{'c': 'c_', 'Gamma': 'G_', 'rho': 'rho_'}[f] + var]
                rel = abs(fd - an)/max(1e-8, abs(fd))
                worst = max(worst, rel)
        print(f"(y1,r)=({y1},{r}): c={e['c']:.6f} Gamma={e['Gamma']:+.3e} rho={e['rho']:.6f}  G_y={e['G_y']:+.4f} G_r={e['G_r']:+.4f} c_y={e['c_y']:+.4f} c_r={e['c_r']:+.4f}")
    print(f"max relative mismatch analytic vs finite-difference derivatives: {worst:.2e}")
