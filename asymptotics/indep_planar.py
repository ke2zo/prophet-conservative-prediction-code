"""Independent mpmath implementation of the planar reduction (from the note's definitions only)."""
import mpmath as mp
mp.mp.dps = 30

phi = lambda t: t*(1 - mp.log(t)) if t > 0 else mp.mpf(0)
Phi = lambda t: t*mp.log(t) - t
def psi(r): return r*(r - 1 - mp.log(r))/(1 - r)**2
def g(x): return x - 1 - mp.log(x)
def theta(e): return 1 - (1 + e)*mp.exp(-e)

K = mp.findroot(lambda K: mp.quad(lambda t: 1/(K - 1 + phi(t)), [0, mp.mpf('1e-6'), mp.mpf('1e-3'), 1]) - 1, mp.mpf('1.3415'))
beta = 1/K

def pts(a, b):
    # geometric breakpoints between a and b (a<b) for quadrature
    n = max(2, int(mp.ceil(mp.log(b/a)/mp.log(4))) + 1)
    return [a*(b/a)**(mp.mpf(k)/n) for k in range(n + 1)]

def quad(f, a, b):
    return mp.quad(f, pts(a, b))

def reduced(y1, r):
    y1 = mp.mpf(y1); r = mp.mpf(r)
    ps = psi(r)
    S0 = -mp.log(y1*ps)
    q = S0/(1 - r); tau = q*y1; kap = r*q**2*y1
    D = lambda t: tau + Phi(y1) - Phi(t)
    Ts = 1 + mp.log(r)/S0
    K1 = lambda z: quad(lambda t: 1/D(t), y1, z)
    K1one = K1(mp.mpf(1))
    if not (0 < Ts < K1one):
        return None
    # solve K1(yM) = Ts by bisection+secant
    lo, hi = y1, mp.mpf(1)
    for _ in range(200):
        mid = (lo + hi)/2
        if K1(mid) < Ts: lo = mid
        else: hi = mid
        if hi - lo < mp.mpf(10)**(-mp.mp.dps + 3): break
    yM = mp.findroot(lambda z: K1(z) - Ts, (lo + hi)/2)
    eps = -mp.log(yM); rho = 1/D(yM)
    K2 = quad(lambda t: 1/D(t)**2, y1, yM)
    K3 = quad(lambda t: t/D(t)**2, y1, yM)
    M = 1 + kap*K2; y0 = y1*ps
    c = M + kap*rho/eps
    Gam = kap*rho/eps*(1 - yM) + M - y0 - kap*K3 - M/rho
    delta = tau + Phi(y1)
    return dict(y1=y1, r=r, psi=ps, S0=S0, q=q, tau=tau, kap=kap, Ts=Ts, yM=yM, eps=eps, rho=rho,
                K2=K2, K3=K3, M=M, y0=y0, c=c, Gamma=Gam, delta=delta, D=D)

def checks(V):
    y1, r = V['y1'], V['r']; K2, K3, delta, yM, rho, tau = V['K2'], V['K3'], V['delta'], V['yM'], V['rho'], V['tau']
    a_lhs = delta*K2 + K3; a_rhs = yM*rho - y1/tau
    th = theta(V['eps'])
    b_rhs = V['c']*th - y1*g(V['psi'])
    I0 = quad(lambda t: 1/(K - 1 + phi(t)), mp.mpf('1e-40'), y1) if y1 > 1e-40 else 0
    I1 = mp.quad(lambda t: 1/(K - 1 + phi(t)), [yM, 1])
    W = quad(lambda t: 1/(V['D'](t)*(K - 1 + phi(t))), y1, yM)
    c_lhs = K - 1/rho
    c_rhs = th + (I0 + I1 - mp.log(1/r)/V['S0'])/W
    return a_lhs - a_rhs, V['Gamma'] - b_rhs, c_lhs - c_rhs, V['Gamma']
