"""Rigorous check (mpmath interval arithmetic) of the numerical inequalities used in the four tail
lemmas that complete the cover of (0,1)^2 outside the main rectangle
      u = ln y1 in [-40, ln y_top],   x = logit r in [-30, 30].
Cases (Z1)-(Z4) of Lemma 5.7 of the manuscript. Notation: L = ln(1/y1), ell = ln(1/r), psi = psi(r), S0 = L + ln(1/psi), lambda = S0 - ell,
T* = lambda/S0, tau = S0 y1/(1-r), kappa = r tau^2/y1, delta = tau - y1(1+L).  beta0 = 0.745 < beta."""
import sys
from mpmath import iv, mpf
iv.prec = 80
def lg(x): return iv.log(x)
def ex(x): return iv.exp(x)

def check(eta, ca, cb, ytop, beta0='0.745', X=30, U=40, quiet=False):
    import builtins
    print = (lambda *a, **k: None) if quiet else builtins.print
    assert float(eta) > 0, 'eta must be positive'
    eta = iv.mpf(eta); beta0 = iv.mpf(beta0); ca = iv.mpf(ca); cb = iv.mpf(cb); ytop = iv.mpf(ytop)
    ok = True
    # ---------- (Z1)  r <= e^{-X}:  delta > eta  =>  lambda < lam_max  =>  c < c_a  (on rho >= beta0)
    eX = ex(-X)
    etap = eta*(1 - eX) - ex(-(X + 1))
    lam_max = lg(1/etap)
    t1 = lam_max/X                                   # T* < lambda/ell
    z1 = 1/iv.mpf(X - 1) + t1/beta0                  # y1 + T*/beta0
    eps1 = lg(1/z1)
    bnd = eX/(1 - eX)*(lam_max + (X + lam_max)/eps1)
    c1 = bool(etap.a > 0) and bool(z1.b < 1) and bool(bnd.b < ca - 1)
    print(f"(Z1) r<=e^-{X}: lam_max={float(lam_max.b):.4f}, T*<{float(t1.b):.4f}, y1+T*/b0<{float(z1.b):.4f}, "
          f"c-1 <= {float(bnd.b):.3e}  (< c_a-1: {c1})")
    ok &= c1
    # ---------- (Z2)  1-r <= e^{-X}: feasible & delta > eta  =>  c > c_b
    m = eX
    ln2 = lg(2)
    Tmin = 1 - (m/(1 - m))/ln2                       # T* = 1 - ell/S0, ell <= m/(1-m), S0 > ln 2
    y1max = m/(Tmin*ln2)                             # from tau T* < 1 and S0 y1 > y1 ln 2
    kap_min = (1 - m)*eta**2/y1max
    cl = 1 + kap_min*Tmin**2
    c2 = bool(Tmin.a > 0) and bool(cl.a > cb)
    print(f"(Z2) 1-r<=e^-{X}: 1-T*<{float((1 - Tmin).b):.3e}, y1<{float(y1max.b):.3e}, kappa>{float(kap_min.a):.3e}, "
          f"c > {float(cl.a):.3e}  (> c_b: {c2})")
    ok &= c2
    # ---------- (Z3)  y1 <= e^{-U}, e^{-(X+1)} <= r <= 1-e^{-(X+1)}:  delta <= eta
    L = iv.mpf(U)
    lnipsi = (X + 1) - lg(iv.mpf(X))                 # ln(1/psi(r)) <= ln(1/(r(ell-1))) at r=e^{-(X+1)}
    dmax = ex(-L)*(L + lnipsi)*ex(X + 1)
    c3 = bool(dmax.b <= eta.a)
    print(f"(Z3) y1<=e^-{U}: delta <= {float(dmax.b):.4e}  (<= eta: {c3})")
    ok &= c3
    # ---------- (Z4)  y1 >= y_top: feasible => r > r_*, and c > 1 + beta0 r_* (y_top ln2)^2 / ln(1/y_top) > c_b
    rs = iv.mpf('0.31')
    g = (rs - 1 - lg(rs))/(1 - rs)**2                # g = psi(r)/r, strictly decreasing
    c4a = bool(g.a > (1/ytop).b)                     # g(0.31) > 1/y_top  =>  feasible r > 0.31
    cl4 = 1 + beta0*rs*(ytop*ln2)**2/lg(1/ytop)
    c4 = c4a and bool(cl4.a > cb)
    print(f"(Z4) y1>={float(ytop.a)}: g(0.31)={float(g.a):.6f} > 1/y_top={float((1/ytop).b):.6f}: {c4a}; "
          f"c > {float(cl4.a):.3f}  (> c_b: {c4})")
    ok &= c4
    # ---------- the pieces cover (0,1)^2
    s30 = 1/(1 + ex(iv.mpf(X)))                      # sigma(-X)
    c5 = bool(s30.b < eX.a) and bool(s30.a > ex(-(X + 1)).b)
    print(f"cover: sigma(-{X}) in (e^-{X+1}, e^-{X}): {c5}")
    ok &= c5
    print("ALL TAIL CHECKS PASS" if ok else "TAIL CHECK FAILED")
    return ok

if __name__ == '__main__':
    check(*sys.argv[1:5], U=int(sys.argv[5]) if len(sys.argv) > 5 else 40)
