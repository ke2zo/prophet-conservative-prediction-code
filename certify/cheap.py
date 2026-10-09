"""Cheap (quadrature-free) rigorous exclusion criteria on boxes, and the box maps from (u, x) cells.
   u = ln y1,  x = logit r = ln(r/(1-r)).
Criteria (valid for every (y1,r) in the box; see the note):
 C0a  lambda <= 0                       -> T* <= 0, infeasible
 C0b  tau*T* >= 1 - y1                  -> K1(1) < (1-y1)/tau <= T*, infeasible
 C1   delta = b <= eta                  -> rho_hat > 1/(1+eta) = rho_A
 C5   tau >= 1/beta0                    -> rho_hat <= 1/tau <= beta0
 C24  1 + kap T*^2/(1-y1) + beta0 kap/L > c_b     -> c > c_b   (on rho_hat >= beta0)
 C3   1 + r/(1-r) (lambda + S0/eps_lo) < c_a,  eps_lo = -ln(y1 + T*/beta0)  -> c < c_a (on rho_hat >= beta0)
"""
import numpy as np
from ivec import I, ilog, iexp
import planar_iv as P

def boxes_from_cells(ulo, uhi, xlo, xhi):
    Y1 = I(iexp(I(ulo)).lo, iexp(I(uhi)).hi)
    def sig(x): return (1.0 + iexp(-I(x))).recip()
    R = I(sig(xlo).lo, sig(xhi).hi)
    return Y1, R

def cheap(Y1, R, prm):
    B = P.base(Y1, R)
    one = I(1.0)
    lam, Ts, tau, kap, b, S0 = B['lam'], B['Ts'], B['tau'], B['kap'], B['b'], B['S0']
    L = -B['lny']
    code = np.zeros(len(Y1.lo), dtype=np.int8)       # 0 = unresolved
    c0a = lam.hi <= 0
    code[c0a & (code == 0)] = 1
    # points with lambda <= 0 are infeasible, so on the feasible part lambda, T* may be replaced by
    # their positive parts (this keeps every bound below valid for boxes straddling lambda = 0)
    lamp = I(np.maximum(lam.lo, 0.0), lam.hi)
    Tsp = I(np.maximum(Ts.lo, 0.0), Ts.hi)
    c0b = (tau*Tsp).lo >= (one - Y1).hi
    code[c0b & (code == 0)] = 2
    c1 = b.hi <= prm['eta']
    code[c1 & (code == 0)] = 3
    c5 = (I(tau.lo)*prm['beta0']).lo > 1.0
    code[c5 & (code == 0)] = 4
    lowc = one + kap*Tsp.sqr()/(one - Y1) + prm['beta0']*kap/L
    c24 = lowc.lo > prm['cb']
    code[c24 & (code == 0)] = 5
    z = Y1 + Tsp/prm['beta0']
    zok = z.hi < 1.0
    zz = I(np.where(zok, z.lo, 0.5), np.where(zok, z.hi, 0.5))
    epslo = -ilog(zz)
    upc = one + (R/(one - R))*(lamp + S0/epslo)
    c3 = zok & (upc.hi < prm['ca'])
    code[c3 & (code == 0)] = 6
    # C7: small-T* enclosures (no quadrature), valid at every feasible point of the box when L T* < 1:
    #   y_M - y1 <= T* D_M,  tau <= D_M <= tau/(1 - L T*),  tau T*/D_M^2 <= K2 <= T*/tau,
    #   y1 K2 <= K3 <= y_M K2,  eps in [-ln(y1 + T* D_M), L],  and then c, rho_hat, Gamma from their formulas.
    LT = (L*Tsp).hi
    sm = (code == 0) & (LT < 1.0) & (tau.lo > 0)
    if sm.any():
        k = np.nonzero(sm)[0]
        y1 = Y1[k]; t = tau[k]; Tk = Tsp[k]; kk = kap[k]; Lk = L[k]
        DMhi = t.hi/(1.0 - I(LT[k]))
        DM = I(t.lo, DMhi.hi)
        yMhi = y1 + Tk*DM
        ok7 = yMhi.hi < 1.0
        yMhi = I(np.where(ok7, yMhi.lo, 0.5), np.where(ok7, yMhi.hi, 0.5))
        yM = I(y1.lo, yMhi.hi)
        eps = I((-ilog(I(yMhi.hi))).lo, Lk.hi)
        K2 = I(((t*Tk)/DM.sqr()).lo, (Tk/t).hi)
        K2 = I(np.maximum(K2.lo, 0.0), K2.hi)
        K3 = I((y1*K2).lo, (yM*K2).hi)
        M = one + kk*K2
        Pv = kk/DM
        from planar_box import E1
        e1 = E1(eps)
        G7 = Pv*e1 + M*(one - DM) - B['y0'][k] - kk*K3
        c7 = M + Pv/eps
        rho7 = DM.recip()
        onepeta = I(1.0) + I(prm['eta'])
        excl = ok7 & ((G7.lo > 0) | (G7.hi < 0) | (c7.lo > prm['cb']) | (c7.hi < prm['ca'])
                      | ((I(rho7.lo)*onepeta).lo > 1.0) | (rho7.hi < prm['beta0']))
        code[k[excl]] = 7
    return code, B

CODES = {1: 'T*<=0', 2: 'tau T*>=1-y1', 3: 'delta<=eta', 4: 'tau>=1/beta0', 5: 'c>c_b (lower bd)', 6: 'c<c_a (upper bd)',
         7: 'small-T* bounds'}
