"""High-precision (mpmath) Euler-Lagrange points deep in the corner c -> infinity, and the ratios whose
limits are predicted by the asymptotic analysis (manuscript, Section 6.1):
    eps/y1^2 -> a = 2 g K/(K-1)^2,   c y1^3 -> b = beta^2 (K-1)^4/(2 g),
    gap/y1^2 -> A = 3 g/(2 (K-1)^2 F2),   c^{2/3} gap -> A0 = A b^{2/3},
with g = ln 2 - 1/2, K = 1/beta, F2 = int_0^1 dt/(K-1+phi)^2, gap = 1/beta - 1/rho.
The zero curve is parametrized by y1: for given (y1, eps) the equation K1(e^{-eps}) = T* is solved for r by
bisection (on the ranges used K1 - T* changes sign exactly once in r, although it is not monotone), and eps
is then fixed by the closed form c theta(eps) = y1 g(psi(r))."""
import mpmath as mp
mp.mp.dps = 40
phi = lambda t: t*(1 - mp.log(t))
K = mp.findroot(lambda K: mp.quad(lambda t: 1/(K - 1 + phi(t)), [0, mp.mpf('1e-8'), 1]) - 1, mp.mpf('1.3415'))
beta = 1/K
F2 = mp.quad(lambda t: 1/(K - 1 + phi(t))**2, [0, mp.mpf('1e-8'), 1])
g2 = mp.log(2) - mp.mpf(1)/2
a = 2*g2*K/(K - 1)**2; b = beta**2*(K - 1)**4/(2*g2); A = 3*g2/(2*(K - 1)**2*F2); A0 = A*b**(mp.mpf(2)/3)
print(f"K = {mp.nstr(K, 16)}, beta = {mp.nstr(beta, 16)}, F2 = {mp.nstr(F2, 16)}")
print(f"predicted limits: a = {mp.nstr(a, 10)}, b = {mp.nstr(b, 10)}, A = {mp.nstr(A, 10)}, A0 = {mp.nstr(A0, 10)}")

def psi(r): return r*(r - 1 - mp.log(r))/(1 - r)**2
def base(y1, m):
    r = 1 - m; ps = psi(r); S0 = -mp.log(y1*ps); Ts = 1 + mp.log(r)/S0
    tau = S0*y1/m; delta = tau - phi(y1)
    return r, ps, S0, Ts, tau, delta
def integ(f, e, L, n=10):
    return mp.quad(f, mp.linspace(e, L, n))
def K1(y1, eps, delta):
    L = -mp.log(y1)
    return integ(lambda u: mp.exp(-u)/(delta + phi(mp.exp(-u))), eps, L)
def m_of(y1, eps):
    """1 - r solving K1(e^{-eps}) = T*  (bisection in log m; K1 - T* < 0 near r = 1, > 0 at small r)."""
    lo, hi = mp.mpf('1e-30'), mp.mpf(1) - mp.mpf('1e-20')      # K1 - T* < 0 at small m (r -> 1), > 0 at large m
    f = lambda m: (lambda r, ps, S0, Ts, tau, delta: K1(y1, eps, delta) - Ts)(*base(y1, m))
    flo, fhi = f(lo), f(hi)
    assert flo < 0 < fhi, (flo, fhi)
    for it in range(200):
        mid = mp.sqrt(lo*hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
        if hi/lo - 1 < mp.mpf('1e-30'): break
    return mp.sqrt(lo*hi)
def point(y1, eps):
    m = m_of(y1, eps); r, ps, S0, Ts, tau, delta = base(y1, m)
    kap = r*tau**2/y1; L = -mp.log(y1)
    K2 = integ(lambda u: mp.exp(-u)/(delta + phi(mp.exp(-u)))**2, eps, L)
    DM = delta + phi(mp.exp(-eps)); rho = 1/DM
    M = 1 + kap*K2; c = M + kap*rho/eps
    theta = -mp.expm1(-eps) - eps*mp.exp(-eps)
    Gam = c*theta - y1*(ps - 1 - mp.log(ps))
    return dict(m=m, c=c, rho=rho, Gamma=Gam, delta=delta, tau=tau)
def zero_at(y1, e_guess):
    """eps with Gamma(y1, eps) = 0 (Gamma < 0 for small eps, > 0 for larger eps): bisection in log eps."""
    lo, hi = e_guess/4, e_guess*4
    assert point(y1, lo)['Gamma'] < 0 < point(y1, hi)['Gamma']
    for it in range(120):
        mid = mp.sqrt(lo*hi)
        if point(y1, mid)['Gamma'] < 0: lo = mid
        else: hi = mid
        if hi/lo - 1 < mp.mpf('1e-25'): break
    e = mp.sqrt(lo*hi)
    return e, point(y1, e)

print(f"{'y1':>9} {'c':>11} {'1-r':>9} {'eps/y1^2':>9} {'c y1^3':>10} {'gap/y1^2':>9} {'c^(2/3) gap':>12} {'y1 ln(1/y1)':>11}")
for y1 in [mp.mpf('0.0312103189'), mp.mpf('1e-2'), mp.mpf('3e-3'), mp.mpf('1e-3'), mp.mpf('3e-4'), mp.mpf('1e-4')]:
    e, d = zero_at(y1, a*y1**2)
    gap = K - 1/d['rho']; c = d['c']
    print(f"{mp.nstr(y1, 4):>9} {mp.nstr(c, 5):>11} {mp.nstr(d['m'], 5):>9} {mp.nstr(e/y1**2, 6):>9} {mp.nstr(c*y1**3, 6):>10} "
          f"{mp.nstr(gap/y1**2, 6):>9} {mp.nstr(c**(mp.mpf(2)/3)*gap, 7):>12} {mp.nstr(-y1*mp.log(y1), 4):>11}", flush=True)
