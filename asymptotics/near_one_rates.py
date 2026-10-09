"""Theorem thm:second (second order at alpha -> 1): the rates at the zero of Gamma on the fiber r = e^{-ell}
(the fiber is scanned over y1 in (1e-9 y_max, y_max), y_max = r/psi(r), on a grid that is logarithmic below
0.3 y_max and linear above, and the number of sign changes of Gamma is recorded), located in double precision (planar_float) and refined by secant steps in 30-digit arithmetic
(indep_planar).  Columns: (1 - y1*ell) ell^2/ln ell, (lambda*ell - 1) ell/ln ell,
((c-1) ln ell/(e^{-ell} ell) - 1) ell ln ell, A ell^2/ln ell, (chi(yM) - (1+ln ell)/ell) ell^2/ln ell and the
remainder of (ii) times ell^3/ln ell.  Rows with ell >= 60 exceed the working precision (1/rho - 1 ~ c - 1
~ 1e-25) and are not printed.  Output: near_one_rates.txt."""
import sys, math, numpy as np, warnings
warnings.filterwarnings('ignore')
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planar_float import evaluate, psi
from scipy.optimize import brentq
import mpmath as mp
import indep_planar as IP
mp.mp.dps = 30
def Gf(y, r):
    e = evaluate(y, r, derivs=False); return None if e is None else e['Gamma']
out = []
hdr = (f"{'ell':>5} {'#zeros':>6} {'(1-x)l^2/lnl':>12} {'(lam*l-1)l/lnl':>14} {'(c-1)/(rl/lnl)-1 *l*lnl':>24} {'A*l^2/lnl':>10} {'(chi-(1+lnl)/l)l^2/lnl':>22} {'(u-1+(1+lnl)/l)l^3/lnl':>24}")
print(hdr); out.append(hdr)
for ell in [8, 10, 15, 20, 30, 40]:
    r = math.exp(-ell); ymax = r/psi(r)*(1 - 1e-9)
    ys = np.concatenate([np.geomspace(1e-9*ymax, 0.3*ymax, 400, endpoint=False), np.linspace(0.3*ymax, ymax, 600)])
    vals = [Gf(float(y), r) for y in ys]
    zeros = [i for i in range(len(ys)-1) if vals[i] is not None and vals[i+1] is not None and (vals[i] < 0) != (vals[i+1] < 0)]
    assert all(v is not None for v in vals)
    i = zeros[0]
    z = brentq(lambda y: Gf(y, r), ys[i], ys[i+1], xtol=1e-17)
    rm = mp.exp(-ell); y = mp.mpf(z)
    for _ in range(4):   # secant refinement in 30 digits
        h = y*mp.mpf('1e-12')
        g0 = IP.reduced(y, rm)['Gamma']; g1 = IP.reduced(y + h, rm)['Gamma']
        y = y - g0*h/(g1 - g0)
    V = IP.reduced(y, rm)
    l = mp.mpf(ell); lnl = mp.log(l)
    x = y*l; lam = V['S0'] - l
    cm1 = V['kap']*V['K2'] + V['kap']*V['rho']/V['eps']
    A = (rm*V['tau'] - V['y0'])/cm1
    chi = V['yM']*(1 - mp.log(V['yM']))
    u = (1/V['rho'] - 1)/cm1
    line = (f"{ell:5d} {len(zeros):6d} {float((1-x)*l**2/lnl):12.4f} {float((lam*l-1)*l/lnl):14.4f} {float((cm1/(rm*l/lnl)-1)*l*lnl):24.4f} "
          f"{float(A*l**2/lnl):10.4f} {float((chi-(1+lnl)/l)*l**2/lnl):22.4f} {float((u-1+(1+lnl)/l)*l**3/lnl):24.4f}  Gamma={float(V['Gamma']):.1e}")
    print(line, flush=True); out.append(line)
open('near_one_rates.txt', 'w').write("\n".join(out) + "\n")
