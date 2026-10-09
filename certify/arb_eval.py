"""Second, independent implementation of the planar reduction in ball arithmetic (Arb, through python-flint).

It shares no code with the certification (ivec.py, planar_iv.py, planar_box.py): the arithmetic is Arb's, and the
integrals K_1..K_5 are computed by Arb's rigorous integration (acb.integral, Petras' algorithm) instead of the
Taylor models on a fixed mesh.  Used by arb_check.py.

Formulas (Proposition 5.1 of the manuscript and the derivative formulas of planar_float.py):
   psi(r) = r (r - 1 - ln r)/(1 - r)^2,  S0 = -ln(y1 psi),  T* = 1 + ln r/S0,  q = S0/(1 - r),  tau = q y1,
   kappa = r q^2 y1,  Phi(t) = t ln t - t,  b = tau + Phi(y1),  D(t) = b - Phi(t),
   K1 = int 1/D, K2 = int 1/D^2, K3 = int t/D^2, K4 = int 1/D^3, K5 = int t/D^3   over [y1, y_M],  K1 = T*,
   D_M = D(y_M), rho_hat = 1/D_M, eps = -ln y_M, M = 1 + kappa K2, Q = kappa/(eps D_M), c = M + Q,
   Gamma = Q (1 - y_M) + M (1 - D_M) - y1 psi - kappa K3.

Inputs are balls (y1, r).  For thick inputs the integrals are enclosed through monotonicity: the integrands are
positive and decreasing in b, so with exact endpoints y1_c, z_c (midpoints) the integral lies between its values
at b_hi and b_lo; the two slivers [y1, y1_c] and [z_c, y_M] are added as (signed length) x (range of the
integrand).  y_M is enclosed from K1(y_M) = T*:  y_M - z_c = (T* - K1(z_c)) D(xi)."""
import math
from flint import arb, acb, ctx

PREC = 128
ctx.prec = PREC
ONE = arb(1)

class Infeasible(Exception):
    """The box contains no feasible point (the reason is the message)."""
class Undecided(Exception):
    """The enclosures are too wide to decide (the caller may subdivide)."""

def lo(x): return x.lower()          # exact lower endpoint, as a ball of radius 0
def hi(x): return x.upper()
def hull(a, b): return a.union(b)
def Phi(t): return t*t.log() - t
_HALF = arb(0.5)
def _hser(s, d):
    """h(s) = sum_{k>=0} s^k/(k+2) (d = 0) or h'(s) (d = 1) for a ball 0 <= s < 1/2: 220 terms and the tail."""
    N = 220; acc = arb(0)
    for k in range(N - 1, -1, -1):                       # Horner; all coefficients are positive
        acc = acc*s + (arb(1)/(k + 2) if d == 0 else arb(k + 1)/(k + 3))
    return acc + arb(0, 1)*arb(2)**(-200)
def hfun(r):
    """psi(r)/r = (r - 1 - ln r)/(1 - r)^2 = h(1 - r); the series avoids the cancellation near r = 1."""
    s = 1 - r
    return _hser(s, 0) if hi(s) < _HALF else (r - 1 - r.log())/s**2
def psi(r): return r*hfun(r)
def dpsi(r):
    s = 1 - r
    return _hser(s, 0) - r*_hser(s, 1) if hi(s) < _HALF else (-2*s - (1 + r)*r.log())/s**3

def _integral(k, j, bb, a, z, tol):
    """int_a^z t^j/D^k dt for exact a < z and the ball bb (the enclosure holds for every b in bb)."""
    bc = acb(bb)
    def f(t, analytic):
        D = bc - t*t.log(analytic=analytic) + t
        return (t if j else acb(1))/D**k
    v = acb.integral(f, acb(a), acb(z), rel_tol=arb(tol), abs_tol=arb(tol)*arb(10)**(-30), eval_limit=100000)
    if not v.is_finite(): raise Undecided('integral')
    return v.real

def integrals(bb, a, z, thin):
    """Enclosures of K1..K5 over the exact interval [a, z], valid for every b in bb."""
    if not lo(bb) - Phi(arb(a)) > 0: raise Undecided('D sign')      # D is increasing in t: D > 0 on [a, z]
    out = []
    for (k, j) in ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1)):
        if thin:
            out.append(_integral(k, j, bb, a, z, 2.0**(-(PREC - 20))))
        else:
            tol = 1e-13
            out.append(hull(_integral(k, j, hi(bb), a, z, tol), _integral(k, j, lo(bb), a, z, tol)))
    return out

def base(y1, r):
    """lambda = S0 - ln(1/r) = -ln(y1 psi(r)/r) and T* = lambda/S0 = 1/(1 + ell/lambda), ell = ln(1/r)."""
    if not (lo(y1) > 0 and hi(y1) < 1 and lo(r) > 0 and hi(r) < 1): raise Undecided('domain')
    h = hfun(r); ell = -r.log(); lam = -(y1*h).log()
    if not (h.is_finite() and lam.is_finite()): raise Undecided('psi')
    if hi(lam) <= 0: raise Infeasible('T*<=0')
    S0 = lam + ell; ps = r*h
    Tshi = 1/(1 + lo(ell)/hi(lam))
    Ts = hull(1/(1 + hi(ell)/lo(lam)), Tshi) if lo(lam) > 0 else hull(arb(0), Tshi)   # positive part
    q = S0/(1 - r); tau = q*y1; kap = r*q*q*y1
    return dict(ps=ps, S0=S0, Ts=Ts, q=q, tau=tau, kap=kap, b=tau + Phi(y1), lam=lam, ell=ell)

def primaries(y1, r, zc=None):
    """Enclosures of the primary quantities on the box y1 x r, by direct evaluation.
    Raises Infeasible if the box has no feasible point and Undecided if the enclosures are too wide."""
    y1 = arb(y1); r = arb(r)
    B = base(y1, r)
    ps, S0, Ts, q, tau, kap, b = (B[k] for k in ('ps', 'S0', 'Ts', 'q', 'tau', 'kap', 'b'))
    thin = bool(y1.rad() < arb(2)**(-60) and r.rad() < arb(2)**(-60))
    if not lo(tau) > 0: raise Undecided('tau')
    if not lo(B['lam']) > 0: raise Undecided('T* sign')
    # feasibility: K1(1) > T*.  On [y1, 1], D >= tau, so K1(1) <= (1 - y1)/tau
    if lo(tau*Ts) >= hi(1 - y1): raise Infeasible('tau T* >= 1 - y1')
    a = y1.mid()
    if zc is None: zc = _float_yM(float(y1.mid()), float(r.mid()), float(b.mid()), float(Ts.mid()))
    if zc is None:
        K1full = integrals(b, a, ONE, thin)[0] + (a - y1)/_Drange(b, hull(y1, a))
        if hi(K1full) < lo(Ts): raise Infeasible('K1(1) < T*')
        raise Undecided('feasibility')
    z = arb(zc)
    K = integrals(b, a, z, thin)
    Dbot = _Drange(b, hull(y1, a))                       # D on the bottom sliver
    e = Ts - K[0] - (a - y1)/Dbot                        # = int_{z_c}^{y_M} dt/D
    # y_M - z_c = e D(xi): verify a guess
    Dz = b - Phi(z)
    guess = e*Dz*arb(1, 0.25) + arb(0, 1e-30)*Dz
    T = z + guess
    if not (lo(T) > hi(y1) and hi(T) < 1): raise Undecided('y_M range')
    step = e*_Drange(b, T)
    if not (lo(step) > lo(guess) and hi(step) < hi(guess)): raise Undecided('y_M enclosure')
    yM = z + step
    bot = a - y1; tb = hull(y1, a); top = yM - z; tt = hull(z, yM); Dtop = _Drange(b, tt)
    def full(i, k, j):
        return K[i] + bot*(tb if j else ONE)/Dbot**k + top*(tt if j else ONE)/Dtop**k
    return dict(psi=ps, S0=S0, Ts=Ts, q=q, tau=tau, kap=kap, b=b, yM=yM, DM=b - Phi(yM), eps=-yM.log(), y0=y1*ps,
                K2=full(1, 2, 0), K3=full(2, 2, 1), K4=full(3, 3, 0), K5=full(4, 3, 1), zc=zc)

def E1(eps):
    """(1 - e^{-e})/e, decreasing in e > 0."""
    f = lambda e: (1 - (-e).exp())/e
    return hull(f(hi(eps)), f(lo(eps)))
def E2(eps):
    """(e^e - 1 - e)/e^2, increasing in e > 0."""
    f = lambda e: (e.exp() - 1 - e)/e**2
    return hull(f(lo(eps)), f(hi(eps)))

def assemble(V):
    """c, Gamma, rho_hat from the primaries; Gamma = P E1(eps) + M (1 - D_M) - y0 - kappa K3, P = kappa/D_M."""
    V['E1'] = E1(V['eps']); V['E2'] = E2(V['eps'])
    V['M'] = 1 + V['kap']*V['K2']; V['P'] = V['kap']/V['DM']
    V['c'] = V['M'] + V['P']/V['eps']
    V['Gamma'] = V['P']*V['E1'] + V['M']*(1 - V['DM']) - V['y0'] - V['kap']*V['K3']
    V['rho'] = 1/V['DM']
    return V

def grads(V, y1, r):
    """First derivatives in (y1, r), evaluated on the enclosures in V."""
    ps, S0, q, tau, kap, DM, yM, eps = (V[k] for k in ('psi', 'S0', 'q', 'tau', 'kap', 'DM', 'yM', 'eps'))
    K2, K3, K4, K5, M, Pv, e1, e2 = (V[k] for k in ('K2', 'K3', 'K4', 'K5', 'M', 'P', 'E1', 'E2'))
    lnr = r.log(); lny = y1.log(); om = 1 - r; dps = dpsi(r)
    S0y, S0r = -1/y1, -dps/ps
    Tsy, Tsr = -lnr*S0y/S0**2, 1/(r*S0) - lnr*S0r/S0**2
    qy, qr = S0y/om, S0r/om + S0/om**2
    tauy, taur = (S0 - 1)/om, qr*y1
    kapy, kapr = r*(2*q*qy*y1 + q*q), q*q*y1 + 2*r*q*qr*y1
    by, br = tauy + lny, taur
    yMy, yMr = DM*(Tsy + 1/tau + K2*by), DM*(Tsr + K2*br)
    DMy, DMr = by + eps*yMy, br + eps*yMr
    epsy, epsr = -yMy/yM, -yMr/yM
    K2y, K2r = yMy/DM**2 - 1/tau**2 - 2*K4*by, yMr/DM**2 - 2*K4*br
    K3y, K3r = yM*yMy/DM**2 - y1/tau**2 - 2*K5*by, yM*yMr/DM**2 - 2*K5*br
    My, Mr = kapy*K2 + kap*K2y, kapr*K2 + kap*K2r
    Py, Pr = kapy/DM - kap*DMy/DM**2, kapr/DM - kap*DMr/DM**2
    return dict(S0=(S0y, S0r), Ts=(Tsy, Tsr), q=(qy, qr), tau=(tauy, taur), kap=(kapy, kapr), b=(by, br),
                yM=(yMy, yMr), DM=(DMy, DMr), eps=(epsy, epsr), K2=(K2y, K2r), K3=(K3y, K3r), M=(My, Mr),
                P=(Py, Pr), y0=(ps, y1*dps), rho=(-DMy/DM**2, -DMr/DM**2),
                c=(My + Py/eps - Pv*epsy/eps**2, Mr + Pr/eps - Pv*epsr/eps**2),
                Gamma=(Py*e1 + Pv*e2*yMy + My*(1 - DM) - M*DMy - ps - kapy*K3 - kap*K3y,
                       Pr*e1 + Pv*e2*yMr + Mr*(1 - DM) - M*DMr - y1*dps - kapr*K3 - kap*K3r))

KEYS_MV = ('S0', 'Ts', 'q', 'tau', 'kap', 'b', 'yM', 'DM', 'eps', 'K2', 'K3', 'y0')

def evaluate(y1, r, passes=2):
    """Enclosures on the box y1 x r: returns (V, G) with V the values (c, Gamma, rho, ...) and G the gradients.
    For a thick box the primaries are tightened by mean-value forms around the center,
        X(box) in X(center) + X_y(box)(y1 - y_c) + X_r(box)(r - r_c),
    intersected with the direct enclosures."""
    y1 = arb(y1); r = arb(r)
    V = assemble(primaries(y1, r))
    if bool(y1.rad() < arb(2)**(-60) and r.rad() < arb(2)**(-60)):
        return V, grads(V, y1, r)
    yc, rc = y1.mid(), r.mid()
    C = assemble(primaries(yc, rc, zc=V['zc']))
    dy, dr = y1 - yc, r - rc
    for p in range(passes):
        G = grads(V, y1, r)
        for k in KEYS_MV:
            V[k] = _meet(V[k], C[k] + G[k][0]*dy + G[k][1]*dr)
        assemble(V)
    G = grads(V, y1, r)
    for k in ('c', 'Gamma', 'rho'):
        V[k] = _meet(V[k], C[k] + G[k][0]*dy + G[k][1]*dr)
    V['center'] = C
    return V, G

def _meet(a, b):
    if not a.overlaps(b): raise RuntimeError('empty intersection of two enclosures: one of them is invalid')
    return a.intersection(b)

def _Drange(b, T):
    """Enclosure of D(t) = b - Phi(t) for t in T (a ball inside (0,1)); D is increasing in t."""
    D = hull(lo(b) - Phi(lo(T)), hi(b) - Phi(hi(T)))
    if not lo(D) > 0: raise Undecided('D sign')
    return D

# ---- floating-point starting value for y_M (not part of any enclosure)
_GL = None
def _float_yM(y1, r, b, Ts):
    global _GL
    import numpy as np
    if _GL is None:
        x, w = np.polynomial.legendre.leggauss(60); _GL = (x, w)
    x, w = _GL
    L = -math.log(y1)
    def K1(u):                                           # int_{e^-u}^{y1} dt/D in the variable s = -ln t
        npan = max(2, int(math.ceil((L - u)/0.7))); ed = np.linspace(u, L, npan + 1); tot = 0.0
        for i in range(npan):
            s = 0.5*(ed[i + 1] - ed[i])*x + 0.5*(ed[i + 1] + ed[i]); t = np.exp(-s)
            tot += 0.5*(ed[i + 1] - ed[i])*np.dot(w, t/(b - t*np.log(t) + t))
        return tot
    if Ts <= 0 or K1(0.0) <= Ts: return None
    a_, b_ = 0.0, L                                       # K1 decreasing in u, K1(L) = 0
    for _ in range(200):
        m = 0.5*(a_ + b_)
        if K1(m) > Ts: a_ = m
        else: b_ = m
        if b_ - a_ < 1e-15*max(1.0, L): break
    return math.exp(-0.5*(a_ + b_))

if __name__ == '__main__':
    import time
    for (y, r, hy, hr) in [(0.2364800, 0.1605454, 1e-4, 1e-4), (0.03098395712773769, 0.7226459, 3.3e-5, 2.9e-6),
                           (0.0885839, 2.01e-05, 8.4e-4, 5e-3), (0.020750951308082296, 0.7814158, 4.7e-6, 2.5e-7)]:
        t0 = time.time(); V, G = evaluate(arb(y), arb(r)); t1 = time.time()
        print(f"thin ({y}, {r}): c={V['c'].str(25)} Gamma={V['Gamma'].str(8)} rho={V['rho'].str(25)}  {t1 - t0:.3f}s")
        t0 = time.time(); V, G = evaluate(arb(y, hy*y), arb(r, hr*r)); t1 = time.time()
        print(f"   thick: c={V['c'].str(8)} Gamma={V['Gamma'].str(6)} G_y={G['Gamma'][0].str(8)} "
              f"G_r={G['Gamma'][1].str(6)} c_y={G['c'][0].str(6)} rho={V['rho'].str(12)}  {t1 - t0:.3f}s")
