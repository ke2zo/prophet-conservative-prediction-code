"""Remark 6.5 and Supplement (l), Theorem S1 (uniqueness near the ends): the Jacobians used in its proof, at the Euler-Lagrange
points of the continuation branch (double precision, planar_float, analytic derivatives / central differences).
 (a) near c = 1, coordinates (ell, x) with y1 = x/ell, r = e^{-ell}, Phi = (Gamma, ln(c-1)); limit [[0,-1],[-1,0]].
     Also at x shifted by +-3 ln(ell)/ell^2 (the window of the proof).
 (b) large c, coordinates (sigma, e) = (ln y1, eps/y1^2) with delta implicit, Phi = (Gamma/y1, ln c);
     limit [[0, (K-1)^2 beta/2], [-3, -1/e]], determinant 3(K-1)^2 beta/2 = 0.1304.
For each Jacobian DY the script also prints ||A^-1 (DY - A)|| (spectral norm) with A the limit matrix of
the proof (in (b) at e = a_inf); the proof's injectivity criterion is that this is <= 1/2 on a window.
Output: uniqueness_jacobians.txt."""
import sys, math, os, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from planar_float import evaluate, psi
from scipy.optimize import brentq
from scipy.integrate import quad
K = 1.3414889923701555; beta = 1/K; gstar = math.log(2) - 0.5
chi = lambda t: t*(1 - math.log(t)) if t > 0 else 0.0
F2 = quad(lambda t: 1/(K - 1 + chi(t))**2, 0, 1, limit=200)[0]
ainf = 2*gstar*K/(K - 1)**2; dinf = 3*gstar/(2*(K - 1)**2*F2)
out = []
def say(t): print(t, flush=True); out.append(t)
def Gf(y, r):
    e = evaluate(y, r, derivs=False); return None if e is None else e['Gamma']

say("(a) near c = 1: rows ell, shift of x, dGamma/dell, dGamma/dx, dln(c-1)/dell, dln(c-1)/dx, det (limit -1)")
for ell in [8, 10, 12, 15, 20, 25]:
    r = math.exp(-ell); ymax = r/psi(r)*(1 - 1e-9)
    ys = np.linspace(0.3*ymax, ymax, 400); vals = [Gf(float(y), r) for y in ys]
    z = next(brentq(lambda y: Gf(y, r), ys[i], ys[i+1], xtol=1e-17) for i in range(len(ys) - 1)
             if vals[i] is not None and vals[i+1] is not None and (vals[i] < 0) != (vals[i+1] < 0))
    x0 = z*ell
    for dx in (-3*math.log(ell)/ell**2, 0.0, 3*math.log(ell)/ell**2):
        x = x0 + dx; e = evaluate(x/ell, r, derivs=True); cm1 = e['c'] - 1.0
        Gx = e['G_y']/ell; Gl = -(x/ell**2)*e['G_y'] - r*e['G_r']
        Lx = e['c_y']/(ell*cm1); Ll = (-(x/ell**2)*e['c_y'] - r*e['c_r'])/cm1
        Jm = np.array([[Gl, Gx], [Ll, Lx]]); A0 = np.array([[0.0, -1.0], [-1.0, 0.0]])
        crit = np.linalg.norm(np.linalg.solve(A0, Jm - A0), 2)
        say(f"  ell={ell:3d} dx={dx:+.5f}: {Gl:+.3e} {Gx:+.5f} {Ll:+.5f} {Lx:+.5f}  det {Gl*Lx - Gx*Ll:+.4f}  "
            f"||A^-1(DY-A)|| {crit:.3f}")

def r_from(s, delta):
    tau = delta + chi(s); L = -math.log(s)
    f = lambda rb: tau*rb - s*(L - math.log(psi(1 - rb)))
    grid = np.geomspace(1e-14, 0.999, 4000); prev = grid[0]; fp = f(prev)
    for g in grid[1:]:
        fg = f(g)
        if fp < 0 <= fg: return 1 - brentq(f, prev, g, xtol=1e-17, rtol=1e-15)
        prev, fp = g, fg
    return None
def Phi_v(ls, v):
    s = math.exp(ls); r = r_from(s, K - 1 + s*s*v)
    if r is None: return None
    e = evaluate(s, r, derivs=False)
    if e is None: return None
    return np.array([e['Gamma']/s, math.log(e['c']), e['eps']/s**2]), e
say(f"(b) large c: rows s = y1, c, e = eps/s^2 at the zero, Jacobian in (ln s, e), det (limit {3*(K-1)**2*beta/2:.4f})")
for s in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    ls = math.log(s)
    g = lambda v: (Phi_v(ls, v) or [np.array([np.nan]*3)])[0][0]
    vs = np.linspace(-dinf - 4, -dinf + 4, 161); gv = [g(v) for v in vs]
    z = next(brentq(g, vs[k], vs[k+1], xtol=1e-13) for k in range(len(vs) - 1)
             if np.isfinite(gv[k]) and np.isfinite(gv[k+1]) and (gv[k] < 0) != (gv[k+1] < 0))
    h = 1e-4; Jv = np.zeros((3, 2))
    for k, (dls, dv) in enumerate(((h, 0.0), (0.0, h))):
        Jv[:, k] = (Phi_v(ls + dls, z + dv)[0] - Phi_v(ls - dls, z - dv)[0])/(2*h)
    T = np.array([[1.0, 0.0], [-Jv[2, 0]/Jv[2, 1], 1/Jv[2, 1]]])      # (ls, e) -> (ls, v)
    Je = Jv[:2] @ T
    P0, e0 = Phi_v(ls, z)
    A0 = np.array([[0.0, (K - 1)**2*beta/2], [-3.0, -1/ainf]])
    crit = np.linalg.norm(np.linalg.solve(A0, Je - A0), 2)
    say(f"  s={s:g}: c={e0['c']:.4g}, e={P0[2]:.4f}: J = [[{Je[0,0]:+.4f}, {Je[0,1]:+.4f}], [{Je[1,0]:+.4f}, {Je[1,1]:+.4f}]]  "
        f"det {np.linalg.det(Je):+.4f}  ||A^-1(DY-A)|| {crit:.3f}")
open('uniqueness_jacobians.txt', 'w').write("\n".join(out) + "\n")
