"""Checks for Lemma D.1 and Theorems 6.3 and 6.4 of the manuscript (near-perfect prediction, c -> 1):
 (1) the identity 1/J - 1 = [delta kappa K2 + (r tau - y0) + theta(eps) Q]/M on F (Q = c - M), and
     r tau - y0 > 0;
 (2) the explicit path lambda = 1 (y1 = r/(e psi(r)), r -> 0) used in the proof: c -> 1 and
     (1/J - 1)/(c - 1) -> 1;
 (3) along the continuation branch: (1/rho - 1)/(c - 1) against 1 - (1 + ln l)/l, l = ln(1/r)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from ivec import I, iexp
import planar_iv as P
from planar_box import assemble

def psi(r): return r*(r - 1 - math.log(r))/(1 - r)**2
rng = np.random.default_rng(3)
y = np.exp(rng.uniform(math.log(1e-4), math.log(0.6), 3000)); r = 1/(1 + np.exp(-rng.uniform(-9, 3, 3000)))
V = P.evaluate(I(y), I(r)); assemble(V); ok = V['ok']
E = V['Gamma'] + V['M']*V['DM']; J = V['M']/E
th = 1.0 - (1.0 + V['eps'])*iexp(-V['eps']); Q = V['c'] - V['M']
rhs = (V['b']*V['kap']*V['K2'] + I(r)*V['tau'] - V['y0'] + th*Q)/V['M']
d = np.abs((J.recip() - 1.0).mid - rhs.mid)[ok]
print(f"(1) {int(ok.sum())} random points of F: max |(1/J-1) - formula| = {d.max():.1e}; "
      f"min (r tau - y0) = {(I(r)*V['tau'] - V['y0']).lo[ok].min():.2e} > 0")
print("(2) path lambda = 1 (30-digit evaluation with indep_planar.py; double precision loses digits for ell >= 20):")
import mpmath as mp
import indep_planar as IP
print(f"{'ell':>6} {'r':>10} {'c-1':>11} {'(1/J-1)/(c-1)':>14} {'theta(eps)':>11} {'M-1':>10} {'kap K2/Q':>10}"
      f" {'y1':>8} {'tau':>8} {'delta':>8} {'K1(1)':>8} {'T*':>8} {'L T*':>7}")
for ell in (6, 8, 10, 14, 20, 30):
    rr = mp.exp(-ell); y1 = rr/(mp.e*IP.psi(rr))
    V = IP.reduced(y1, rr)
    Jv = V['M']/(V['Gamma'] + V['M']/V['rho']); c = V['c']; M = V['M']
    t = IP.theta(V['eps']); kK2 = V['kap']*V['K2']
    K1one = IP.quad(lambda u: 1/V['D'](u), V['y1'], mp.mpf(1))
    print(f"{ell:6.1f} {float(rr):10.3e} {float(c-1):11.3e} {float((1/Jv - 1)/(c - 1)):14.6f} {float(t):11.6f} "
          f"{float(M-1):10.3e} {float(kK2/(c - M)):10.3e} {float(y1):8.5f} {float(V['tau']):8.5f} "
          f"{float(V['delta']):8.5f} {float(K1one):8.5f} {float(V['Ts']):8.5f} {float(-mp.log(y1)*V['Ts']):7.4f}")
print("(3) continuation branch:")
print(f"{'c-1':>10} {'ell':>7} {'(1/rho-1)/(c-1)':>16} {'1-(1+ln l)/l':>13}")
A = np.load('../certify/curve_float.npy')
for k in np.linspace(0, 1900, 8).astype(int):
    rr, y1 = A[k, 0], A[k, 1]
    V = P.evaluate(I(np.array([y1])), I(np.array([rr]))); assemble(V)
    c = V['c'].mid[0]; rho = V['rho'].mid[0]; ell = -math.log(rr)
    print(f"{c-1:10.3e} {ell:7.3f} {(1/rho - 1)/(c - 1):16.6f} {1-(1+math.log(ell))/ell:13.6f}")
