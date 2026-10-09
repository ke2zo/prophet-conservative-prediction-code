"""
================================================================================
CLEAN HIGH-RESOLUTION EULER-LAGRANGE SHOOTING SOLVER  for  rho(c)
================================================================================
Proof obligation 3 -- numerical companion to the analytic derivation in
EL_DERIVATION.md.  Replaces the noisy np.gradient / free-min seeding of
methodC_odeverify.py and /tmp/el_forward2.py (which corroborated the EL relation
only to ~2e-3, seeded spurious atoms, and -- crucially -- OMITTED the atom at 1).

Continuum (n->infty) worst-case i.i.d. body on {0} U [1,c]:
   tail-rate s(x)=lim n(1-F(x)),  y=e^{-s}=lim P(M<=x),  R(w)=INT_w^c s,  R'=ln y.

EXTREMAL STRUCTURE (validated):
   * atom at 0:        mass y0 = e^{-S0},  S0 = total positive rate = s on [0,1).
   * atom at 1:        rate S0 - s(1+) >= 0   (s jumps DOWN at x=1).
   * EL body [1,M]:    R(x)^2 y'(x) = kappa  (const);  reparam dy/dx=kappa/R^2, dR/dx=ln y.
                       integrated DOWN from the cap end to x=1, giving y(1+) and R1=R(1).
   * flat gap (M,c):   s == eps  (no continuous mass), R(M)=eps*(c-M).
   * cap atom at c:    rate eps = -ln y(M).

The four FREE parameters are (M, eps, kappa, S0); they are pinned by the four
stationarity / transversality conditions derived in EL_DERIVATION.md:
   (T0  top)   ALG = M                                  [free-edge: threshold reaches the body edge]
   (Tk  bulk)  kappa = R(M)/rho                          [EL constant value]
   (T1  left-body)  rho*y(1+) = R(M)/(R1*R0),  R0=S0+R1  [EL* at x=1, body bottom]
                    <=> kappa = y(1+)*R1*R0
   (TS  left-atoms) rho*y0 = R(M)*I_S0,                   [atom-0 / atom-1 split]
                    I_S0 = INT_0^1 (1-w)/Rtilde(w)^2 dw = (1/S0^2)[ln(R0/R1)+R1/R0-1].

We ROOT-FIND these four (Newton/fsolve), seeded by a robust 4-parameter
minimisation of rho.  rho_EL is the continuum (infinite-resolution) value; the
finite free-PWS minimiser and the exact discrete DP converge DOWN onto it.
================================================================================
"""
import numpy as np
import math
from scipy import optimize

LOG = math.log
EXP = math.exp


# ----------------------------------------------------------------------------
# Fast inlined downward EL integrator (RK4 in x, from the cap end M to x=1)
# ----------------------------------------------------------------------------
def integ(M, eps, kappa, c, NB):
    """Integrate dy/dx=kappa/R^2, dR/dx=ln y downward from (x=M, y=e^{-eps},
    R=eps*(c-M)) to x=1. Returns (X,Y,R) ascending in x, or None on failure."""
    if not (1.0 < M < c) or eps <= 0.0 or kappa <= 0.0:
        return None
    h = (M - 1.0) / NB
    y = EXP(-eps)
    R = eps * (c - M)
    X = np.empty(NB + 1); Y = np.empty(NB + 1); RR = np.empty(NB + 1)
    X[0] = M; Y[0] = y; RR[0] = R
    for i in range(1, NB + 1):
        if R <= 0.0 or y <= 0.0 or y >= 1.0:
            return None
        k1y = kappa / (R * R); k1R = LOG(y)
        ya = y - 0.5 * h * k1y; Ra = R - 0.5 * h * k1R
        if Ra <= 0.0 or ya <= 0.0:
            return None
        k2y = kappa / (Ra * Ra); k2R = LOG(ya)
        yb = y - 0.5 * h * k2y; Rb = R - 0.5 * h * k2R
        if Rb <= 0.0 or yb <= 0.0:
            return None
        k3y = kappa / (Rb * Rb); k3R = LOG(yb)
        yc = y - h * k3y; Rc = R - h * k3R
        if Rc <= 0.0 or yc <= 0.0:
            return None
        k4y = kappa / (Rc * Rc); k4R = LOG(yc)
        y = y - (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        R = R - (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)
        if not (0.0 < y < 1.0) or R <= 0.0:
            return None
        X[i] = M - i * h; Y[i] = y; RR[i] = R
    return X[::-1], Y[::-1], RR[::-1]


def ev(M, eps, kappa, S0, c, NB=20000):
    """Honest rho=ALG/Emax and all transversality quantities for parameters
    (M, eps, kappa, S0).  S0 (total rate / atom-0) is FREE; the body bottom
    y(1+) and R1 come from the ODE; the atom at 1 has rate S0 - (-ln y(1+))."""
    out = integ(M, eps, kappa, c, NB)
    if out is None:
        return None
    X, Y, R = out
    y1p = Y[0]; R1 = R[0]; s1p = -LOG(y1p)
    if S0 < s1p - 1e-9:            # atom-1 rate must be nonnegative
        return None
    y0 = EXP(-S0); yM = EXP(-eps); R0 = S0 + R1
    # Emax = c - INT_0^c y  =  c - [ y0*1 + INT_1^M y dx + yM*(c-M) ]
    Emax = c - (y0 + np.trapezoid(Y, X) + yM * (c - M))
    if Emax <= 1e-12:
        return None
    # ALG: INT_0^A dw/Rtilde = 1.  [0,1] part is analytic; body part trapezoid.
    G0 = (1.0 / S0) * LOG(R0 / R1)
    invR = 1.0 / R
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (invR[1:] + invR[:-1]) * np.diff(X))])
    Gt = G0 + cum
    if Gt[-1] >= 1.0:
        idx = int(np.searchsorted(Gt, 1.0)); idx = min(max(idx, 1), len(X) - 1)
        a, b = Gt[idx - 1], Gt[idx]
        fr = (1.0 - a) / max(b - a, 1e-300)
        ALG = X[idx - 1] + fr * (X[idx] - X[idx - 1])
    else:                          # threshold runs into the flat cap region
        need = 1.0 - Gt[-1]
        ALG = min(c - (c - M) * EXP(-eps * need), c)
    I_S0 = (1.0 / S0 ** 2) * (LOG(R0 / R1) + R1 / R0 - 1.0)
    return dict(rho=ALG / Emax, ALG=ALG, Emax=Emax, M=M, eps=eps, kappa=kappa,
                S0=S0, y0=y0, y1p=y1p, R1=R1, R0=R0, RM=eps * (c - M),
                I_S0=I_S0, atom1=S0 - s1p)


# ----------------------------------------------------------------------------
# The four transversality residuals  (each -> 0 at the EL optimum)
# ----------------------------------------------------------------------------
def residuals(p, c, NB):
    M, eps, kappa, S0 = p
    r = ev(M, eps, kappa, S0, c, NB)
    if r is None:
        return [1.0, 1.0, 1.0, 1.0]
    rho, RM, R1, R0 = r['rho'], r['RM'], r['R1'], r['R0']
    return [r['ALG'] / M - 1.0,                       # (T0) top / ALG=M
            kappa * rho / RM - 1.0,                   # (Tk) EL constant
            rho * r['y1p'] * R1 * R0 / RM - 1.0,      # (T1) left-body (uses y(1+))
            rho * r['y0'] / (RM * r['I_S0']) - 1.0]   # (TS) atom-0/atom-1 split


# ----------------------------------------------------------------------------
# Robust seed: 4-parameter minimisation of rho  (family min == extremal once the
# atom at 1 is included, so this no longer hits the spurious-kappa basin).
# ----------------------------------------------------------------------------
def seed_min(c, NB=8000, seed=0):
    rng = np.random.default_rng(seed)
    best = (2.0, None)

    def obj(th, NBB):
        M = 1.0 + (c - 1.0) / (1.0 + EXP(-th[0]))
        eps = EXP(th[1]); kap = EXP(th[2]); S0 = EXP(th[3])
        r = ev(M, eps, kap, S0, c, NBB)
        return r['rho'] if r is not None else 2.0

    # multistart: scale guesses with c (M ~ 1 + O(sqrt small), eps ~ R/c, S0 grows with c)
    Mg = [1.0 + (c - 1.0) * f for f in (0.04, 0.1, 0.25)]
    epsg = [g for g in (0.05, 0.15, 0.35) if g * c < 8]
    Sg = [1.5, 2.5, 3.5] if c <= 20 else [3.0, 4.5, 6.0]
    for M0 in Mg:
        for e0 in epsg:
            for S00 in Sg:
                k0 = e0 * (c - M0) / 0.8
                th = [math.log((M0 - 1) / (c - M0)), math.log(e0), math.log(k0), math.log(S00)]
                for meth in ('Nelder-Mead', 'Powell'):
                    res = optimize.minimize(lambda t: obj(t, NB), th, method=meth,
                                            options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-13}
                                            if meth == 'Nelder-Mead' else {'maxiter': 4000})
                    th = res.x
                if res.fun < best[0]:
                    best = (res.fun, th)
    th = best[1]
    return (1.0 + (c - 1.0) / (1.0 + EXP(-th[0])), EXP(th[1]), EXP(th[2]), EXP(th[3]))


def solve(c, NB_seed=8000, NB_root=30000, NB_final=200000, seed=0):
    """Returns (rho_EL, params dict). Seed by minimisation, polish by 4x4 root-find."""
    p0 = seed_min(c, NB=NB_seed, seed=seed)
    sol = optimize.fsolve(lambda p: residuals(p, c, NB_root), p0,
                          full_output=True, xtol=1e-13)
    p = sol[0]
    r = ev(*p, c, NB_final)
    if r is None:                  # fall back to the seed if the root-find wandered
        r = ev(*p0, c, NB_final); p = p0
    res = residuals(p, c, NB_final)
    r['resid'] = res
    r['ok'] = sol[2] == 1
    return r['rho'], r


if __name__ == "__main__":
    import pickle, os
    ref = pickle.load(open('/tmp/ref_targets.pkl', 'rb')) if os.path.exists('/tmp/ref_targets.pkl') else {}
    print(f"{'c':>6} {'rho_EL':>10} {'ref(PWS/DP)':>12} {'M*':>8} {'eps':>8} "
          f"{'kappa':>8} {'S0':>7} {'y0':>7} {'atom@1':>7} {'ALG/M':>8} {'max|resid|':>10}")
    out = {}
    for c in [1.25, 2.0, 5.0, 10.0, 100.0]:
        rho, r = solve(c)
        refv = ref.get(c, {}).get('rf', float('nan'))
        out[c] = r
        print(f"{c:6.2f} {rho:10.6f} {refv:12.6f} {r['M']:8.4f} {r['eps']:8.5f} "
              f"{r['kappa']:8.5f} {r['S0']:7.4f} {r['y0']:7.5f} {r['atom1']:7.4f} "
              f"{r['ALG']/r['M']:8.5f} {max(abs(v) for v in r['resid']):10.1e}", flush=True)
    pickle.dump(out, open('/tmp/el_solver_out.pkl', 'wb'))
