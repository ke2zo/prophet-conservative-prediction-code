"""Structure check without the Euler-Lagrange ansatz (Supplement (b) of the manuscript).

Minimize J = ALG/E[M] over ALL continuum band instances whose positive values sit on a
uniform grid 1 = x_0 < ... < x_K = c (rates a_i >= 0 at the grid points, no structural
assumption), from several starts, and compare the minimizer with the predictions of the
manuscript: (i) J_min is just above rho_EL(c); (ii) positive atom at c; (iii) (almost) no mass in
(ALG, c); (iv) G >= 0 on [1,c] (first-order condition), G ~ 0 on the support.
"""
import os, sys, math
import numpy as np
from scipy import optimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
from el_shoot_solver import solve
from rho_reduced2 import PWS

K = int(os.environ.get('FREEMIN_K', '80'))

def pws_from_atoms(a, c):
    x = 1.0 + (c - 1.0) * np.arange(K + 1) / K
    suffix = np.cumsum(a[::-1])[::-1]          # suffix[i] = sum_{j>=i} a_j
    sval = suffix[1:]                          # s on [x_i, x_{i+1}) = sum_{j>=i+1} a_j
    return x, PWS(x, sval, suffix[0])

def J_of(a, c):
    if a.sum() <= 1e-12:
        return 1.0
    x, p = pws_from_atoms(a, c)
    E = p.Emax()
    return p.alg() / E if E > 1e-12 else 1.0

def G_at_nodes(a, c, sub=400):
    """G(x_i) = int_0^{x_i} [R(A) Psi(x) - J y(x)] dx at the grid nodes, with Psi integrated
    exactly on each linear piece of R and the phi-integral done by the midpoint rule on cells
    aligned with the atoms (no quadrature error from the jumps of y).  Also returns the
    finite-difference values E * dJ/da_i, which must agree (Lemma 4.7 of the manuscript)."""
    x, p = pws_from_atoms(a, c)
    A, E = p.alg(), p.Emax(); J = A / E; RA = p.R(A)
    suffix = np.cumsum(a[::-1])[::-1]
    # pieces [lo, hi) with constant s: [0,1) -> S0, [x_i, x_{i+1}) -> suffix[i+1]
    los = np.concatenate([[0.0], x[:-1]]); his = np.concatenate([[1.0], x[1:]])
    ss = np.concatenate([[suffix[0]], suffix[1:]])
    def Psi(xv):
        tot = 0.0; top = min(xv, A)
        for lo, hi, sv in zip(los, his, ss):
            if lo >= top: break
            h2 = min(hi, top)
            Rlo, Rhi = p.R(lo), p.R(h2)
            tot += (h2 - lo) / Rlo**2 if sv < 1e-14 else (1.0 / Rhi - 1.0 / Rlo) / sv
        return tot
    G = [0.0]; acc = 0.0
    for lo, hi, sv in zip(los, his, ss):
        mids = lo + (np.arange(sub) + 0.5) * (hi - lo) / sub
        phi = np.array([RA * Psi(m) for m in mids]) - J * math.exp(-sv)
        acc += phi.sum() * (hi - lo) / sub
        G.append(acc)
    G = np.array(G[1:])                      # G at 1 = x_0, x_1, ..., x_K
    fd = []
    for i in range(K + 1):
        d = 1e-7; b = a.copy(); b[i] += d
        fd.append(E * (J_of(b, c) - J) / d)
    return G, np.array(fd), A, J

def run(c, rng):
    rho_el, r = solve(c)
    print(f"\n=== c = {c}:  rho_EL = {rho_el:.7f}   (EL: M* = {r['M']:.4f}, eps = {r['eps']:.4f}, "
          f"atom@1 = {r['atom1']:.4f}, S0 = {r['S0']:.4f})")
    starts = [np.concatenate([[1.0], np.full(K - 1, 2.0 / K), [0.3]])]
    for _ in range(4):
        starts.append(np.concatenate([[rng.uniform(0.3, 2.5)], rng.uniform(0.0, 0.08, K - 1),
                                      [rng.uniform(0.05, 1.0)]]))
    best = None
    for k, a0 in enumerate(starts):
        res = optimize.minimize(lambda a: J_of(a, c), a0, method='L-BFGS-B',
                                bounds=[(0.0, 40.0)] * (K + 1),
                                options=dict(maxiter=20000, maxfun=2_000_000, ftol=1e-15, gtol=1e-11))
        a = res.x; x, p = pws_from_atoms(a, c); A = p.alg(); J = res.fun
        h = (c - 1.0) / K
        gap = a[(x > A + h) & (x < c - 1e-12)].sum()
        body = a[(x > 1.0 + 1e-12) & (x <= A + h)].sum()
        sig = (a > 1e-3) & (x < c - 1e-12)
        top = x[sig].max() if np.any(sig) else float('nan')
        print(f"  start {k}: J = {J:.7f} (J - rho_EL = {J - rho_el:+.2e})  ALG = {A:.4f}  "
              f"atom@1 = {a[0]:.4f}  body mass = {body:.4f}  top of support below c (rate>1e-3) = {top:.4f}  "
              f"gap mass (ALG,c) = {gap:.2e}  atom@c = {a[-1]:.4f}")
        if best is None or J < best[0]:
            best = (J, a)
    J, a = best
    G, fd, A, _ = G_at_nodes(a, c)
    on = a > 1e-3
    print(f"  best: G at grid nodes (exact formula): min_i G(x_i) = {G.min():+.2e};  "
          f"max over support |G(x_i)| = {np.abs(G[on]).max():.2e};  G(c) = {G[-1]:+.2e};  "
          f"max_i |G(x_i) - E dJ/da_i| = {np.abs(G - fd).max():.1e}")

if __name__ == '__main__':
    rng = np.random.default_rng(20260923)
    print(f"grid K = {K} (spacing (c-1)/K)")
    for c in [float(v) for v in os.environ.get('FREEMIN_C', '2,5').split(',')]:
        run(c, rng)
