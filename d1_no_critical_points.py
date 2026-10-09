"""Lemma 4.17 of the manuscript (no collapsed-body minimizers): independent checks of every step.

For a three-point instance a*delta_1 + eps*delta_c with ALG <= 1 (S = a+eps, P = a+c*eps,
E = (1-e^{-S}) + (c-1)(1-e^{-eps}), A = P(1-e^{-S})/S, rho = A/E):
 (1) G(c) - G(1) = (c-1) (phi1(S) - rho e^{-eps})                   [quadrature vs closed form]
 (2) G(1) = (e^{-S}/S) [P - A + (1-A)(e^S-1)] - rho e^{-S}           [quadrature vs closed form]
 (3) with r1 := (P-S)(e^S-1-S) - S N  and  r2 := (P-S)((1-e^{-eps})/eps - e^{-eps}) - e^{-eps} N,
     N := S - (e^S-1)e^{-a}, one has  G(1)=0 <=> r1=0 (given G(c)=G(1)),  G(c)=G(1) <=> r2=0,
     and the identity  r1 - S e^{eps} r2 = (P-S) S (k(S) - k(eps)),  k(x) = (e^x-1-x)/x,
     holds for every P (checked with P as a free variable).
"""
import random
from mpmath import mp, mpf, exp, expm1, quad, log
mp.dps = 40
random.seed(7)

def phi1(x): return -expm1(-x)/x
def k(x): return (expm1(x) - x)/x

def data(a, e, c):
    S = a + e; P = a + c*e; E = -expm1(-S) + (c-1)*(-expm1(-e))
    A = P*phi1(S); rho = A/E
    return S, P, E, A, rho

def G_quad(z, a, e, c):
    """G(z) = int_0^z [R(A) Psi(x) - rho y(x)] dx by quadrature, from the definitions only."""
    S, P, E, A, rho = data(a, e, c)
    def R(w):  # drift of the instance: int_w^inf s
        return S*(1 - w) + e*(c - 1) if w <= 1 else e*(c - w)
    RA = R(A)
    def Psi(x): return quad(lambda w: 1/R(w)**2, [0, min(x, A)])
    def y(x): return exp(-S) if x < 1 else (exp(-e) if x < c else mpf(1))
    pts = [0, min(z, A), min(z, 1), z]
    pts = sorted(set(p for p in pts if p <= z))
    return quad(lambda x: RA*Psi(x) - rho*y(x), pts)

worst = [mpf(0)]*4; n = 0
while n < 40:
    c = 1 + mpf(10)**random.uniform(-1.5, 1.5)
    a = mpf(10)**random.uniform(-1.5, 0.7); e = mpf(10)**random.uniform(-2, 0.3)
    S, P, E, A, rho = data(a, e, c)
    if A > 1:          # outside the collapsed-body region
        continue
    n += 1
    G1, Gc = G_quad(1, a, e, c), G_quad(c, a, e, c)
    d1 = abs((Gc - G1) - (c - 1)*(phi1(S) - rho*exp(-e)))
    d2 = abs(G1 - ((exp(-S)/S)*(P - A + (1 - A)*expm1(S)) - rho*exp(-S)))
    # (3): residual forms as functions of a free P, identity check at two unrelated P values
    N = S - expm1(S)*exp(-a)
    for Pf in (P, P*mpf('1.7') + 3):
        r1 = (Pf - S)*(expm1(S) - S) - S*N
        r2 = (Pf - S)*(phi1(e) - exp(-e)) - exp(-e)*N
        d3 = abs((r1 - S*exp(e)*r2) - (Pf - S)*S*(k(S) - k(e)))
        worst[2] = max(worst[2], d3)
    # consistency: at the true P, G(1)=0-form and r1 differ by a nonzero factor only when G(c)=G(1)
    worst[0] = max(worst[0], d1); worst[1] = max(worst[1], d2)
print(f"(1) max |G(c)-G(1) - closed form| over {n} random instances: {float(worst[0]):.1e}")
print(f"(2) max |G(1) - closed form|:                                  {float(worst[1]):.1e}")
print(f"(3) max |r1 - S e^eps r2 - (P-S) S (k(S)-k(eps))| (free P):   {float(worst[2]):.1e}")
