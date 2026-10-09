"""Sanity check for the journal-version argument (existence + first-order conditions):
at the EL extremal, adding a small atom of rate eta at z changes J = ALG/E[M] by
eta*G(z)/E[M] + O(eta^2), where G(z) = int_0^z [Rt(A) Psi(x) - rho y(x)] dx.
Predictions: G == 0 on the whole band [1,c]; G(c+h) = h*rho*(e^{-eps}-1) < 0."""
import sys, math
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
from el_shoot_solver import solve, integ
from rho_reduced2 import PWS

def pws_from_extremal(r, c, NB=4000):
    X, Y, R = integ(r['M'], r['eps'], r['kappa'], c, NB)
    s_mid = -np.log(0.5 * (Y[1:] + Y[:-1]))
    xb = np.concatenate([X, [c]])
    sval = np.concatenate([s_mid, [r['eps']]])
    return xb, sval, r['S0']

def add_atom(xb, sval, S0, z, eta):
    xb = list(xb); sval = list(sval)
    c = xb[-1]
    if z > c:
        xb.append(z); sval.append(0.0)
    elif z not in xb:
        i = int(np.searchsorted(xb, z)) - 1
        xb.insert(i + 1, z); sval.insert(i + 1, sval[i])
    xb = np.array(xb); sval = np.array(sval)
    sval = sval + eta * (xb[1:] <= z + 1e-15)      # s += eta on [.., z)
    return xb, sval, S0 + eta

def J(xb, sval, S0):
    p = PWS(xb, sval, S0)
    return p.alg() / p.Emax(), p.alg(), p.Emax()

for c in [2.0, 5.0, 10.0]:
    rho_el, r = solve(c)
    xb, sval, S0 = pws_from_extremal(r, c)
    J0, A0, E0 = J(xb, sval, S0)
    eps, M = r['eps'], r['M']
    print(f"\nc={c}: rho_EL={rho_el:.7f}  J(PWS)={J0:.7f}  M*={M:.4f}  eps={eps:.4f}  ALG={A0:.5f}")
    eta = 1e-6
    zs = [1.0 + 1e-9, 0.5 * (1 + M), M, 0.5 * (M + c), c - 1e-9, c + 0.05, c + 0.2, c + 1.0]
    for z in zs:
        Jp = J(*add_atom(xb, sval, S0, z, eta))[0]
        fd = (Jp - J0) / eta
        pred = (z - c) * rho_el * (math.exp(-eps) - 1.0) / E0 if z > c else 0.0
        print(f"   z={z:8.4f}  dJ/deta (finite diff) = {fd:+.3e}   first-order prediction = {pred:+.3e}")
