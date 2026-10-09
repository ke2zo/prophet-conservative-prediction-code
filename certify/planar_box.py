"""Tight enclosures of the planar quantities over boxes Y1 x R, via mean-value forms.

  * thin evaluation at the box center (Taylor-model quadrature, planar_iv),
  * naive interval evaluation over the box (valid but loose),
  * gradient formulas evaluated on enclosures over the box, then mean-value forms
        X(box) in X(center) + X_y(box) (Y1 - y_c) + X_r(box) (R - r_c),
    intersected with the naive enclosure; two passes.
Gamma is written as  Gamma = P E1(eps) + M (1 - D_M) - y0 - kappa K3  with P = kappa/D_M and
E1(eps) = (1 - e^{-eps})/eps  (=(1-y_M)/eps), whose y_M-derivative is E2(eps) = (e^eps - 1 - eps)/eps^2;
this avoids the cancellation between Q(1-y_M) and Q y_M' of the naive formula."""
import numpy as np
from ivec import I, ilog, iexp, where
import planar_iv as P

def E1(eps):
    """(1-e^{-e})/e, decreasing in e>0: enclose from endpoints."""
    def f(e):
        return (1.0 - iexp(-e))/e
    lo = f(I(eps.hi)); hi = f(I(eps.lo))
    return I(lo.lo, hi.hi)
def E2(eps):
    """(e^e-1-e)/e^2, increasing in e>0."""
    def f(e):
        return (iexp(e) - 1.0 - e)/e.sqr()
    lo = f(I(eps.lo)); hi = f(I(eps.hi))
    return I(lo.lo, hi.hi)

def grads(V, Y1, R):
    """Gradient formulas (y1 and r partials) evaluated on the enclosures in V (dict of intervals)."""
    one = I(1.0); r = R; y1 = Y1
    lnr = V['lnr']; lny = V['lny']; S0 = V['S0']; q = V['q']; ps = V['psi']
    tau = V['tau']; kap = V['kap']; DM = V['DM']; yM = V['yM']; eps = V['eps']
    K2, K3, K4, K5 = V['K2'], V['K3'], V['K4'], V['K5']
    M = V['M']; Pv = V['P']; e1 = V['E1']; e2 = V['E2']
    om = one - r
    dps = (-2.0*om - (one + r)*lnr)/(om**3)
    S0y = -(y1.recip()); S0r = -(dps/ps)
    Tsy = -(lnr*S0y)/S0.sqr(); Tsr = (r*S0).recip() - lnr*S0r/S0.sqr()
    qy = S0y/om; qr = S0r/om + S0/om.sqr()
    tauy = (S0 - 1.0)/om; taur = qr*y1
    kapy = r*(2.0*q*qy*y1 + q.sqr()); kapr = q.sqr()*y1 + 2.0*r*q*qr*y1
    by = tauy + lny; br = taur
    yMy = DM*(Tsy + tau.recip() + K2*by); yMr = DM*(Tsr + K2*br)
    DMy = by + eps*yMy; DMr = br + eps*yMr
    epsy = -(yMy/yM); epsr = -(yMr/yM)
    iD2 = DM.sqr().recip(); it2 = tau.sqr().recip()
    K2y = yMy*iD2 - it2 - 2.0*K4*by; K2r = yMr*iD2 - 2.0*K4*br
    K3y = yM*yMy*iD2 - y1*it2 - 2.0*K5*by; K3r = yM*yMr*iD2 - 2.0*K5*br
    My = kapy*K2 + kap*K2y; Mr = kapr*K2 + kap*K2r
    Py = kapy/DM - kap*DMy*iD2; Pr = kapr/DM - kap*DMr*iD2
    ie = eps.recip()
    cy = My + Py*ie - Pv*epsy*ie.sqr(); cr = Mr + Pr*ie - Pv*epsr*ie.sqr()
    y0y = ps; y0r = y1*dps
    Gy = Py*e1 + Pv*e2*yMy + My*(one - DM) - M*DMy - y0y - kapy*K3 - kap*K3y
    Gr = Pr*e1 + Pv*e2*yMr + Mr*(one - DM) - M*DMr - y0r - kapr*K3 - kap*K3r
    return dict(S0=(S0y, S0r), Ts=(Tsy, Tsr), q=(qy, qr), tau=(tauy, taur), kap=(kapy, kapr), b=(by, br),
                yM=(yMy, yMr), DM=(DMy, DMr), eps=(epsy, epsr), K2=(K2y, K2r), K3=(K3y, K3r), M=(My, Mr),
                P=(Py, Pr), c=(cy, cr), Gamma=(Gy, Gr), rho=(-(DMy*iD2), -(DMr*iD2)), y0=(y0y, y0r))

def assemble(V):
    """c, Gamma, rho from the enclosures in V (in place)."""
    V['E1'] = E1(V['eps']); V['E2'] = E2(V['eps'])
    V['M'] = 1.0 + V['kap']*V['K2']
    V['P'] = V['kap']/V['DM']
    V['c'] = V['M'] + V['P']/V['eps']
    V['Gamma'] = V['P']*V['E1'] + V['M']*(1.0 - V['DM']) - V['y0'] - V['kap']*V['K3']
    V['rho'] = V['DM'].recip()
    return V

KEYS_MV = ('S0', 'Ts', 'q', 'tau', 'kap', 'b', 'yM', 'DM', 'eps', 'K2', 'K3', 'M', 'P', 'c', 'Gamma', 'rho', 'y0')

def evaluate_box(Y1, R, passes=2):
    """Returns (V, G, ok): enclosures over the boxes, gradient enclosures over the boxes, ok-mask."""
    yc = Y1.mid; rc = R.mid
    C = P.evaluate(I(yc), I(rc), derivs=True)             # thin center
    C['E1'] = E1(C['eps']); C['E2'] = E2(C['eps']); C['P'] = C['kap']/C['DM']
    C['M'] = 1.0 + C['kap']*C['K2']; C['c'] = C['M'] + C['P']/C['eps']
    C['Gamma'] = C['P']*C['E1'] + C['M']*(1.0 - C['DM']) - C['y0'] - C['kap']*C['K3']
    C['rho'] = C['DM'].recip()
    Vb = P.evaluate(Y1, R, derivs=True)                    # naive box
    ok = C['ok'] & Vb['ok']
    V = dict(Vb); assemble(V)
    dY = Y1 - I(yc); dR = R - I(rc)
    for p in range(passes):
        G = grads(V, Y1, R)
        for k in KEYS_MV:
            gy, gr = G[k]
            mv = C[k] + gy*dY + gr*dR
            V[k] = V[k].meet(mv)
        # re-derive dependent quantities consistently from the tightened primaries
        V['E1'] = E1(V['eps']); V['E2'] = E2(V['eps'])
        for k, f in (('M', lambda: 1.0 + V['kap']*V['K2']), ('P', lambda: V['kap']/V['DM']),
                     ('c', lambda: V['M'] + V['P']/V['eps']),
                     ('Gamma', lambda: V['P']*V['E1'] + V['M']*(1.0 - V['DM']) - V['y0'] - V['kap']*V['K3']),
                     ('rho', lambda: V['DM'].recip())):
            V[k] = V[k].meet(f())
    G = grads(V, Y1, R)
    for k in ('c', 'Gamma', 'rho'):
        gy, gr = G[k]
        V[k] = V[k].meet(C[k] + gy*dY + gr*dR)
    # an empty intersection can only come from an invalid enclosure: flag it (and count it)
    bad = np.zeros(len(yc), bool)
    for k in KEYS_MV + ('E1', 'E2'):
        bad |= V[k].lo > V[k].hi
    V['empty'] = ok & bad
    for k in ('infeas', 'has_lo', 'zlo_part', 'K2part'): V[k] = Vb[k]
    return V, G, ok & ~bad, C

if __name__ == '__main__':
    import time, sys
    from float_ref import evalp
    rng = np.random.default_rng(3)
    centers = [(0.2364800, 0.1605454), (0.13278, 0.4368), (0.05, 0.3), (0.3, 0.2), (0.01, 0.5), (0.2, 0.05), (0.002, 0.8), (0.1, 0.6)]
    yc = np.array([p[0] for p in centers]); rc = np.array([p[1] for p in centers])
    for h in (1e-4, 1e-3, 1e-2):
        Y = I(yc*(1 - h), yc*(1 + h)); R = I(rc*(1 - h), rc*(1 + h))
        t0 = time.time(); V, G, ok, C = evaluate_box(Y, R); dt = time.time() - t0
        bad = 0
        for k in range(len(centers)):
            if not ok[k]: continue
            for t in range(20):
                y = rng.uniform(Y.lo[k], Y.hi[k]); r = rng.uniform(R.lo[k], R.hi[k]); f = evalp(y, r)
                if not f['feas']: continue
                for key in ('c', 'Gamma', 'rho'):
                    X = V[key][k]
                    if not (X.lo <= f[key] <= X.hi): bad += 1
        print(f"h={h}: {dt:.2f}s ok={ok.astype(int)} violations={bad}")
        print("   Gamma width:", np.array2string(V['Gamma'].wid, precision=2))
        print("   c width    :", np.array2string(V['c'].wid, precision=2))
        print("   G_y width  :", np.array2string(G['Gamma'][0].wid, precision=2), " G_y center:", np.array2string(C['Gamma'].mid*0 + P.evaluate(I(yc), I(rc), derivs=True)['G_y'].mid, precision=3))
