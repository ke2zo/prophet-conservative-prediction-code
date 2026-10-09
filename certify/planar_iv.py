"""Rigorous interval evaluation of the planar reduction (manuscript, Proposition 5.1) on boxes of (y1, r).

For a batch of boxes Y1 x R (numpy interval arrays) it returns enclosures of
    S0, T*, tau, kappa, delta (= b), y_M, rho_hat = 1/D(y_M), eps, M, c, Gamma,
and, on request, of the first derivatives of c, Gamma, rho_hat in y1 and r.

Quadrature.  D(t) = delta + phi(t) = b - Phi(t),  Phi(t) = t ln t - t.  The integrals
K_j = int_{y1}^{yM} f_j,  f = 1/D, 1/D^2, t/D^2, 1/D^3, t/D^3,
are computed on a fixed geometric mesh t_0 = TMIN < t_1 < ... < t_J = 1 by Taylor models:
on each piece [t_j, t_{j+1}] with center m_j, f(t) = sum_{k<=N} f_k(m_j)(t-m_j)^k + f_{N+1}(xi)(t-m_j)^{N+1}
with f_{N+1}(xi) enclosed by the Taylor coefficient evaluated on the whole piece.  The Taylor
coefficients of D are explicit (D^{(k)} = -Phi^{(k)}, Phi^{(k)}(t) = (-1)^k (k-2)!/t^{k-1}, k>=2),
and those of 1/D, 1/D^2, 1/D^3 follow from the usual recurrences, all in interval arithmetic.
Endpoint uncertainty (y1 and y_M are intervals) is handled by "slivers" bounded with
D >= D(y1) = tau on [y1, .] and D increasing."""
import numpy as np
from ivec import I, ilog, const, where

NTAY = 10
THETA = 1.12
TMIN = 1e-7

# ------------------------------------------------------------------ global mesh
def _make_mesh():
    t = [TMIN]
    while t[-1]*THETA < 1.0:
        t.append(float(t[-1]*THETA))
    t.append(1.0)
    return np.array(t)
TM = _make_mesh(); J = len(TM) - 1
TL, TH = TM[:-1], TM[1:]
MC = 0.5*(TL + TH)                      # centers (floats, used exactly)
_MCi = I(MC); _TLi = I(TL); _THi = I(TH); _Ti = I(TL, TH)
_lnM = ilog(_MCi); _lnT = ilog(_Ti)
_PhiM = _MCi*_lnM - _MCi
# Phi is decreasing on (0,1): Phi(T) = [Phi(t_hi), Phi(t_lo)]
_PhiTL = _TLi*ilog(_TLi) - _TLi; _PhiTH = _THi*ilog(_THi) - _THi
_PhiT = I(_PhiTH.lo, _PhiTL.hi)
def _dcoef(center, lnc):
    """Taylor coefficients d_k, k=1..N+1, of D at `center` (interval), without d_0."""
    out = [-lnc]
    inv = center.recip(); p = I(np.ones_like(center.lo))    # p = 1/center^{k-1}
    for k in range(2, NTAY + 2):
        p = p*inv
        s = -1.0 if k % 2 == 0 else 1.0                      # -(-1)^k
        out.append((p*s)/float(k*(k - 1)))                   # k(k-1) is an exact small integer
    return out
_DM = _dcoef(_MCi, _lnM)                                      # at centers
_DT = _dcoef(_Ti, _lnT)                                       # on whole pieces
# powers (t_{j+1}-m_j)^k and (t_j-m_j)^k for full-piece integration
_HP = _THi - _MCi; _HM = _TLi - _MCi

def _series_recip(d0, d):
    """g = 1/D as Taylor series given d0 (interval array) and d[1..] (list)."""
    g0 = d0.recip(); g = [g0]
    for k in range(1, NTAY + 2):
        acc = d[0]*g[k - 1]
        for i in range(2, k + 1):
            acc = acc + d[i - 1]*g[k - i]
        g.append(-(g0*acc))
    return g
def _cauchy(a, b, sym=False):
    out = []
    for k in range(NTAY + 2):
        if sym:
            acc = None
            for i in range((k + 1)//2):
                t = a[i]*b[k - i]
                acc = t if acc is None else acc + t
            acc = (acc*2.0) if acc is not None else None
            if k % 2 == 0:
                sq = a[k//2].sqr()
                acc = sq if acc is None else acc + sq
            out.append(acc)
        else:
            acc = a[0]*b[k]
            for i in range(1, k + 1):
                acc = acc + a[i]*b[k - i]
            out.append(acc)
    return out
def _times_t(center, a):
    return [center*a[0]] + [center*a[k] + a[k - 1] for k in range(1, NTAY + 2)]

def taylor_all(b):
    """b: interval array shape (n,1). Returns dict name -> (coef list at centers [0..N], remainder coef on pieces)."""
    d0c = b - _PhiM                     # (n,J)
    d0t = b - _PhiT
    res = {}
    for where_, d0, d, cen in (('c', d0c, _DM, _MCi), ('t', d0t, _DT, _Ti)):
        g = _series_recip(d0, d)
        g2 = _cauchy(g, g, sym=True)
        g3 = _cauchy(g2, g)
        tg2 = _times_t(cen, g2); tg3 = _times_t(cen, g3)
        res[where_] = dict(f1=g, f2=g2, f3=tg2, f4=g3, f5=tg3, d0=d0)
    out = {}
    for nm in ('f1', 'f2', 'f3', 'f4', 'f5'):
        out[nm] = (res['c'][nm][:NTAY + 1], res['t'][nm][NTAY + 1])
    out['d0c'] = res['c']['d0']; out['d0t'] = res['t']['d0']
    return out

def _poly_int(coefs, rem, A, B, Mc):
    """int_A^B sum_k coefs[k](t-Mc)^k dt + remainder; A,B,Mc interval arrays broadcastable.
    Remainder: |rem| * int_A^B |t-Mc|^{N+1} dt (A<=B)."""
    ua = A - Mc; ub = B - Mc
    pa = ua; pb = ub
    acc = coefs[0]*(ub - ua)
    for k in range(1, NTAY + 1):
        pa = pa*ua; pb = pb*ub
        acc = acc + coefs[k]*((pb - pa)/float(k + 1))
    pa = pa*ua; pb = pb*ub                                  # (.)^{N+2}
    # int_A^B |t-m|^{N+1} dt  <=  (|ub|^{N+2} + |ua|^{N+2})/(N+2)   (valid in all cases, A<=B)
    bound = (I(np.abs(pb.lo)).hull(I(np.abs(pb.hi))) + I(np.abs(pa.lo)).hull(I(np.abs(pa.hi))))/float(NTAY + 2)
    rb = I(rem.abs_hi())*I(bound.hi)
    return acc + I(-rb.hi, rb.hi)

def base(Y1, R):
    """Base quantities on boxes (monotone enclosures where available)."""
    one = I(1.0)
    lny = ilog(Y1); lnr = ilog(R)
    def psi_thin(r, lr): return r*(r - one - lr)/((one - r).sqr())
    rlo = I(R.lo); rhi = I(R.hi); lrlo = I(lnr.lo); lrhi = I(lnr.hi)
    ps_lo = psi_thin(rlo, ilog(rlo)); ps_hi = psi_thin(rhi, ilog(rhi))
    psi = I(ps_lo.lo, ps_hi.hi)                                   # psi increasing
    lpsi = ilog(psi)
    # ln(r/psi) = ln((1-r)^2/(r-1-ln r)) is increasing in r
    def lrp(r, lr): return ilog((one - r).sqr()/(r - one - lr))
    lrp_lo = lrp(rlo, ilog(rlo)); lrp_hi = lrp(rhi, ilog(rhi))
    LRP = I(lrp_lo.lo, lrp_hi.hi)
    S0 = -lny - lpsi
    lam = -lny + LRP                                                # = S0 - ell > 0 iff feasible
    ell = -lnr
    # T* = lam/(lam+ell): decreasing in y1, increasing in r -> corners
    def Ts_at(y, r):
        l = -ilog(y) + lrp(r, ilog(r)); e = -ilog(r)
        return l/(l + e)
    Ts = I(Ts_at(I(Y1.hi), rhi).lo, Ts_at(I(Y1.lo), rlo).hi) if False else None
    Ts_a = Ts_at(I(Y1.hi), rlo); Ts_b = Ts_at(I(Y1.lo), rhi)
    Ts = I(Ts_a.lo, Ts_b.hi)
    q = S0/(one - R)
    tau = q*Y1
    kap = R*q.sqr()*Y1
    PhiY = I((I(Y1.hi)*ilog(I(Y1.hi)) - I(Y1.hi)).lo, (I(Y1.lo)*ilog(I(Y1.lo)) - I(Y1.lo)).hi)  # Phi decreasing
    b = tau + PhiY                                                  # = delta
    return dict(Y1=Y1, R=R, lny=lny, lnr=lnr, psi=psi, S0=S0, lam=lam, ell=ell, Ts=Ts, q=q, tau=tau,
                kap=kap, PhiY=PhiY, b=b, y0=Y1*psi)

def _piece_index(z):
    j = np.searchsorted(TM, z, side='right') - 1
    return np.clip(j, 0, J - 1)

class Quad:
    """Taylor data for a batch of boxes with parameter b and thin start s (= y1_hi)."""
    def __init__(self, b, s):
        self.n = len(s); self.s = s
        self.tay = taylor_all(I(b.lo[:, None], b.hi[:, None]))
        js = _piece_index(s); self.js = js
        A = np.maximum(TL[None, :], s[:, None])                     # start of each piece
        A = np.minimum(A, TH[None, :])
        self.active = TH[None, :] > s[:, None]
        Ai = I(A); Bi = I(np.broadcast_to(TH[None, :], A.shape)); Mc = I(np.broadcast_to(MC[None, :], A.shape))
        self.cum = {}
        for nm in ('f1', 'f2', 'f3', 'f4', 'f5'):
            coefs, rem = self.tay[nm]
            P = _poly_int(coefs, rem, Ai, Bi, Mc)
            P = where(self.active, P, I(0.0))
            # cumulative sums with outward rounding: cum[:, j] = sum_{i<j} P[:, i]
            lo = np.zeros((self.n, J + 1)); hi = np.zeros((self.n, J + 1))
            for j in range(J):
                x = I(lo[:, j], hi[:, j]) + P[:, j]
                lo[:, j + 1] = x.lo; hi[:, j + 1] = x.hi
            self.cum[nm] = (lo, hi)
    def integral(self, nm, z):
        """int_s^z f (thin z >= s), vectorized."""
        jz = _piece_index(z); idx = np.arange(self.n)
        coefs, rem = self.tay[nm]
        lo, hi = self.cum[nm]
        full = I(lo[idx, jz], hi[idx, jz])
        A = np.maximum(TL[jz], self.s)
        cz = [I(ck.lo[idx, jz], ck.hi[idx, jz]) for ck in coefs]
        rz = I(rem.lo[idx, jz], rem.hi[idx, jz])
        part = _poly_int(cz, rz, I(A), I(z), I(MC[jz]))
        return full + part

def fmax_bound(nm, Dlo, tmax):
    """sup of f over a t-range where D >= Dlo > 0 and t <= tmax."""
    Dl = I(Dlo)
    if nm == 'f1': return Dl.recip().hi
    if nm == 'f2': return (Dl.sqr()).recip().hi
    if nm == 'f3': return (I(tmax)/Dl.sqr()).hi
    if nm == 'f4': return ((Dl**3)).recip().hi
    if nm == 'f5': return (I(tmax)/(Dl**3)).hi

def evaluate(Y1, R, derivs=False, want_gamma=True):
    """Main evaluator. Returns dict of intervals; 'ok' mask marks boxes where all steps succeeded."""
    n = len(Y1.lo)
    B = base(Y1, R)
    ok = (B['lam'].lo > 0) & (B['b'].lo > 0) & (Y1.lo >= TMIN) & (B['Ts'].lo > 0)
    s = Y1.hi.copy()
    b = B['b']
    Q = Quad(I(np.where(ok, b.lo, 1.0), np.where(ok, b.hi, 1.0)), np.where(ok, s, 0.5))
    tau = B['tau']; Ts = B['Ts']
    sl1_f1 = (Y1.hi - Y1.lo)*(np.where(tau.lo > 0, 1.0/np.where(tau.lo > 0, tau.lo, 1.0), np.inf))
    sl1_f1 = np.nextafter(np.nextafter(sl1_f1, np.inf), np.inf)
    # ---------------- float root of mid K1(z) = mid T*  (bisection on (s, 1))
    tgt = Ts.mid
    zl = s.copy(); zh = np.full(n, 1.0)
    for it in range(52):
        zm = 0.5*(zl + zh)
        v = Q.integral('f1', zm).mid + 0.5*sl1_f1
        up = v > tgt
        zh = np.where(up, zm, zh); zl = np.where(up, zl, zm)
    z0 = 0.5*(zl + zh)
    # ---------------- rigorous bracket  sup K1(z_lo) < inf T*,  inf K1(z_hi) > sup T*
    Dz = b.hi - (z0*np.log(z0) - z0)                              # rough D(z0) for step size
    w = (Ts.wid + 2*sl1_f1 + 1e-14)*np.abs(Dz) + 4e-16*z0
    zlo = np.full(n, np.nan); zhi = np.full(n, np.nan)
    todo_lo = ok.copy(); todo_hi = ok.copy()
    for k in range(12):
        cand_lo = np.maximum(z0 - w, s); cand_hi = np.minimum(z0 + w, 1.0)
        Klo = Q.integral('f1', np.where(todo_lo, cand_lo, s))
        Khi = Q.integral('f1', np.where(todo_hi, cand_hi, s))
        good_lo = todo_lo & (Klo.hi + sl1_f1 < Ts.lo) & (cand_lo >= s)
        good_hi = todo_hi & (Khi.lo > Ts.hi) & (cand_hi < 1.0)
        zlo = np.where(good_lo, cand_lo, zlo); zhi = np.where(good_hi, cand_hi, zhi)
        todo_lo &= ~good_lo; todo_hi &= ~good_hi
        if not (todo_lo.any() or todo_hi.any()): break
        w = w*4.0
    # partial information for boxes where the bracket fails
    has_lo = ok & np.isfinite(zlo)
    K1one = Q.integral('f1', np.where(ok, 1.0, 0.5))
    infeas = ok & (K1one.hi + sl1_f1 < Ts.lo)                 # K1(1) < T* on the whole box
    K2part = Q.integral('f2', np.where(has_lo, zlo, s))       # int_{y1_hi}^{z_lo} 1/D^2 <= K2 (feasible pts)
    ok &= np.isfinite(zlo) & np.isfinite(zhi)
    zlo_keep = np.where(has_lo, zlo, s)
    zlo = np.where(ok, zlo, s); zhi = np.where(ok, zhi, np.minimum(s*1.0000001 + 1e-12, 1.0))
    yM = I(zlo, zhi)
    phi_lo = I(zlo)*(I(1.0) - ilog(I(zlo))); phi_hi = I(zhi)*(I(1.0) - ilog(I(zhi)))
    DM = b + I(phi_lo.lo, phi_hi.hi)                                # phi increasing on (0,1)
    Dz_lo = b + phi_lo                                              # D(z_lo) (lower bound on [zlo,zhi])
    eps = -ilog(yM)
    out = dict(B, ok=ok, yM=yM, DM=DM, eps=eps, rho=DM.recip(), K1=Ts, infeas=infeas, has_lo=has_lo,
               zlo_part=zlo_keep, K2part=K2part)
    # ---------------- integrals K2..K5 over [y1, yM]
    names = ('f2', 'f3', 'f4', 'f5') if derivs else ('f2', 'f3')
    for nm in names:
        main = Q.integral(nm, zlo)
        s1 = (Y1.hi - Y1.lo)*fmax_bound(nm, np.where(tau.lo > 0, tau.lo, 1e-300), Y1.hi)
        s2 = (zhi - zlo)*fmax_bound(nm, np.where(Dz_lo.lo > 0, Dz_lo.lo, 1e-300), zhi)
        out['K' + nm[1]] = main + I(0.0, np.nextafter(np.nextafter(s1 + s2, np.inf), np.inf))
    kap = B['kap']; K2 = out['K2']; K3 = out['K3']
    M = 1.0 + kap*K2
    Qv = kap/(eps*DM)
    c = M + Qv
    out.update(M=M, Qv=Qv, c=c)
    if want_gamma:
        out['Gamma'] = Qv*(1.0 - yM) + M*(1.0 - DM) - B['y0'] - kap*K3
    if derivs:
        one = I(1.0); r = R; y1 = Y1
        lnr = B['lnr']; lny = B['lny']; S0 = B['S0']; q = B['q']; ps = B['psi']
        dps = (-2.0*(one - r) - (one + r)*lnr)/((one - r)**3)
        S0y = -(y1.recip()); S0r = -(dps/ps)
        Tsy = -(lnr*S0y)/S0.sqr(); Tsr = (r*S0).recip() - lnr*S0r/S0.sqr()
        qy = S0y/(one - r); qr = S0r/(one - r) + S0/((one - r).sqr())
        tauy = qy*y1 + q; taur = qr*y1
        kapy = r*(2.0*q*qy*y1 + q.sqr()); kapr = q.sqr()*y1 + 2.0*r*q*qr*y1
        by = tauy + lny; br = taur
        K4 = out['K4']; K5 = out['K5']
        lnyM = -eps
        yMy = DM*(Tsy + tau.recip() + K2*by); yMr = DM*(Tsr + K2*br)
        DMy = by - lnyM*yMy; DMr = br - lnyM*yMr
        epsy = -(yMy/yM); epsr = -(yMr/yM)
        iD2 = DM.sqr().recip(); it2 = tau.sqr().recip()
        K2y = yMy*iD2 - it2 - 2.0*K4*by; K2r = yMr*iD2 - 2.0*K4*br
        K3y = yM*yMy*iD2 - y1*it2 - 2.0*K5*by; K3r = yM*yMr*iD2 - 2.0*K5*br
        My = kapy*K2 + kap*K2y; Mr = kapr*K2 + kap*K2r
        y0y = ps; y0r = y1*dps
        Qy = Qv*(kapy/kap - epsy/eps - DMy/DM); Qr = Qv*(kapr/kap - epsr/eps - DMr/DM)
        out.update(c_y=My + Qy, c_r=Mr + Qr,
                   G_y=Qy*(one - yM) - Qv*yMy + My*(one - DM) - M*DMy - y0y - kapy*K3 - kap*K3y,
                   G_r=Qr*(one - yM) - Qv*yMr + Mr*(one - DM) - M*DMr - y0r - kapr*K3 - kap*K3r,
                   rho_y=-(DMy*iD2), rho_r=-(DMr*iD2))
    return out

if __name__ == '__main__':
    import time, math, sys
    from planar_float import evaluate as fev
    pts = [(0.2364800, 0.1605454), (0.13278, 0.4368), (0.05, 0.3), (0.3, 0.2), (0.01, 0.5), (0.2, 0.05), (0.002, 0.8)]
    y = np.array([p[0] for p in pts]); r = np.array([p[1] for p in pts])
    t0 = time.time(); E = evaluate(I(y), I(r), derivs=True); t1 = time.time()
    print(f"thin evaluation of {len(pts)} points with derivatives: {t1 - t0:.3f}s")
    for k, (a, bb) in enumerate(pts):
        f = fev(a, bb)
        if f is None: print((a, bb), 'float: infeasible', E['ok'][k]); continue
        line = []
        for key, fk in (('c', 'c'), ('Gamma', 'Gamma'), ('rho', 'rho'), ('G_y', 'G_y'), ('G_r', 'G_r'), ('c_y', 'c_y'), ('c_r', 'c_r')):
            X = E[key][k]; inside = (X.lo <= f[fk] <= X.hi)
            line.append(f"{key}:{'in' if inside else 'OUT'} w={float(X.wid):.1e}")
        print((a, bb), 'ok' if E['ok'][k] else 'NOT OK', ' '.join(line))
