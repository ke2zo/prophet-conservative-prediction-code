"""Collapsed-body alternative (D1) of Lemma 4.17 (Supplement (c) of the manuscript).

(D1) instances are three-point laws {0,1,c} (rates a at 1, eps at c, S0 = a + eps) whose
optimal rule never rejects a positive value (ALG <= 1).  There J is explicit:
   J = (1-e^{-S0}) (a + c eps) / ( S0 [ (1-e^{-S0}) + (c-1)(1-e^{-eps}) ] ),
and ALG <= 1  <=>  (1/S0) ln((S0 + eps(c-1)) / (eps(c-1))) >= 1.
We compute inf J over the WHOLE (D1) region (not only its critical points) and compare it
with rho(c) (certified values, Theorem 5.5 of the manuscript; in the preliminary version c in {50,100}
were numerical candidates).  A positive
margin means no (D1) instance can be a minimizer at that width.
"""
import os, sys, math
import numpy as np
from scipy import optimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
from rho_reduced2 import PWS

RHO = {1.25: 0.930255, 1.5: 0.896474, 2.0: 0.859258, 2.2: 0.849675, 3.0: 0.824717,
       5.0: 0.797303, 10.0: 0.775436, 20.0: 0.763096, 50.0: 0.754343, 100.0: 0.750790}

def J_d1(S, f, c):
    eps = f * S; a = S - eps
    num = (1 - math.exp(-S)) * (a + c * eps)
    den = S * ((1 - math.exp(-S)) + (c - 1) * (1 - math.exp(-eps)))
    return num / den

def T1(S, f, c):
    eps = f * S
    return math.log((S + eps * (c - 1)) / (eps * (c - 1))) / S

# sanity: explicit J equals the generic continuum evaluator whenever ALG <= 1
rng = np.random.default_rng(1)
for _ in range(200):
    c = float(rng.choice(list(RHO))); S = float(np.exp(rng.uniform(-3, 2))); f = float(rng.uniform(1e-4, 0.999))
    if T1(S, f, c) < 1:
        continue
    eps = f * S; p = PWS([1.0, c], [eps], S)
    assert abs(p.alg() / p.Emax() - J_d1(S, f, c)) < 1e-9, (c, S, f)
print("explicit (D1) formula agrees with the generic evaluator on random admissible points")

print(f"{'c':>6} {'rho(c)':>9} {'inf J over (D1)':>16} {'margin':>9} {'at S0':>8} {'eps/S0':>9} {'ALG<=1 active?':>15}")
for c, rho in RHO.items():
    best = (2.0, None)
    for S in np.exp(np.linspace(math.log(1e-3), math.log(60.0), 700)):
        for f in np.exp(np.linspace(math.log(1e-9), math.log(0.999999), 700)):
            if T1(S, f, c) >= 1.0:
                v = J_d1(S, f, c)
                if v < best[0]:
                    best = (v, (S, f))
    (S, f) = best[1]
    res = optimize.minimize(lambda z: J_d1(math.exp(z[0]), 1 / (1 + math.exp(-z[1])), c),
                            [math.log(S), math.log(f / (1 - f))], method='SLSQP',
                            constraints=[{'type': 'ineq', 'fun': lambda z: T1(math.exp(z[0]), 1 / (1 + math.exp(-z[1])), c) - 1.0}],
                            options=dict(ftol=1e-14, maxiter=500))
    S2, f2 = math.exp(res.x[0]), 1 / (1 + math.exp(-res.x[1]))
    v = min(best[0], J_d1(S2, f2, c)) if T1(S2, f2, c) >= 1 - 1e-12 else best[0]
    if v == best[0]:
        S2, f2 = S, f
    active = abs(T1(S2, f2, c) - 1.0) < 1e-6
    print(f"{c:6.2f} {rho:9.6f} {v:16.6f} {v - rho:+9.4f} {S2:8.4f} {f2:9.2e} {str(active):>15}")
