"""Independent deep-corner check of Theorem 6.1 (34 digits, Gamma from its ORIGINAL definition):
follows the Euler-Lagrange branch parametrized by y1 (arguments), reports eps/y1^2, c y1^3, gap/y1^2,
c^{2/3} gap, B/y1^2 and the size of the remainders.  Usage: python3 indep_corner.py 1e-2 1e-3 ..."""
import mpmath as mp, sys
mp.mp.dps = 34
phi = lambda t: t*(1 - mp.log(t)) if t > 0 else mp.mpf(0)
Phi = lambda t: t*mp.log(t) - t
def psi(r): return r*(r - 1 - mp.log(r))/(1 - r)**2
def gf(x): return x - 1 - mp.log(x)
def theta(e): return -mp.expm1(-e) - e*mp.exp(-e)
def pts(a, b):
    n = max(2, int(mp.ceil(mp.log(b/a)/mp.log(3))) + 1)
    return [a*(b/a)**(mp.mpf(k)/n) for k in range(n + 1)]
def quad(f, a, b): return mp.quad(f, pts(a, b))
K = mp.findroot(lambda K: mp.quad(lambda t: 1/(K - 1 + phi(t)), [0, mp.mpf('1e-9'), mp.mpf('1e-4'), 1]) - 1, mp.mpf('1.3415'))
F2 = mp.quad(lambda t: 1/(K - 1 + phi(t))**2, [0, mp.mpf('1e-9'), mp.mpf('1e-4'), 1])
gs = mp.log(2) - mp.mpf(1)/2
a = 2*gs*K/(K-1)**2; b = (K-1)**4/(K**2*2*gs); A = 3*gs/(2*(K-1)**2*F2); A0 = A*b**(mp.mpf(2)/3)

def base(y1, m):
    r = 1 - m; ps = psi(r); S0 = -mp.log(y1*ps); Ts = 1 + mp.log(r)/S0
    tau = S0*y1/m; delta = tau + Phi(y1)
    return r, ps, S0, Ts, tau, delta
def m_of(y1, eps, m0):
    yM = mp.exp(-eps)
    def f(lm):
        m = mp.exp(lm); r, ps, S0, Ts, tau, delta = base(y1, m)
        return quad(lambda t: 1/(delta + phi(t)), y1, yM) - Ts
    return mp.exp(mp.findroot(f, (mp.log(m0), mp.log(m0) + mp.mpf('0.01')), solver='secant', tol=mp.mpf(10)**-60))
def point(y1, eps, m0):
    m = m_of(y1, eps, m0); r, ps, S0, Ts, tau, delta = base(y1, m)
    yM = mp.exp(-eps); D = lambda t: delta + phi(t)
    kap = r*tau**2/y1
    K2 = quad(lambda t: 1/D(t)**2, y1, yM); K3 = quad(lambda t: t/D(t)**2, y1, yM)
    rho = 1/D(yM); M = 1 + kap*K2; y0 = y1*ps; c = M + kap*rho/eps
    Gam = kap*rho/eps*(1 - yM) + M - y0 - kap*K3 - M/rho          # ORIGINAL definition
    return dict(m=m, r=r, ps=ps, S0=S0, tau=tau, delta=delta, kap=kap, rho=rho, c=c, Gamma=Gam, yM=yM, D=D)
print(f"a={mp.nstr(a,10)} b={mp.nstr(b,10)} A={mp.nstr(A,10)} A0={mp.nstr(A0,10)} -g/(2(K-1)^2)={mp.nstr(-gs/(2*(K-1)**2),10)}")
hdr = f"{'y1':>7} {'c':>10} {'eps/y1^2':>9} {'c*y1^3':>9} {'Dl/y1^2':>9} {'c^2/3 Dl':>9} {'B/y1^2':>9} {'Brem/(y1^3L)':>12} {'Brem/(y1^3L^2)':>14} {'relerrA/(y1L)':>13} {'Gam_closed':>10}"
print(hdr)
for y1s in sys.argv[1:]:
    y1 = mp.mpf(y1s); L = -mp.log(y1)
    m0 = (L + mp.log(2))*y1/(K - 1)
    st = {'m0': m0}
    def G(le):
        P = point(y1, mp.exp(le), st['m0']); st['m0'] = P['m']; return P['Gamma']
    le = mp.findroot(G, (mp.log(a*y1**2), mp.log(a*y1**2) + mp.mpf('0.02')), solver='secant', tol=mp.mpf(10)**-40)
    eps = mp.exp(le); P = point(y1, eps, st['m0'])
    Dl = K - 1/P['rho']
    I0 = mp.quad(lambda t: 1/(K - 1 + phi(t)), [0, y1])
    ell = -mp.log(P['r']); B = I0 - ell/P['S0']
    u = P['delta'] - (K - 1)
    Bmain = -gs*y1**2/(2*(K-1)**2) + y1*u/(K-1)**2
    Brem = B - Bmain
    Gcl = P['c']*theta(eps) - y1*gf(P['ps'])
    print(f"{mp.nstr(y1,3):>7} {mp.nstr(P['c'],5):>10} {mp.nstr(eps/y1**2,7):>9} {mp.nstr(P['c']*y1**3,7):>9} {mp.nstr(Dl/y1**2,7):>9} "
          f"{mp.nstr(P['c']**(mp.mpf(2)/3)*Dl,7):>9} {mp.nstr(B/y1**2,7):>9} {mp.nstr(Brem/(y1**3*L),5):>12} {mp.nstr(Brem/(y1**3*L**2),5):>14} "
          f"{mp.nstr((Dl/y1**2/A - 1)/(y1*L),5):>13} {mp.nstr(Gcl,3):>10}", flush=True)
