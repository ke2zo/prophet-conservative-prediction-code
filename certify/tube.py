"""Validated tube around the zero curve {Gamma = 0}, r in [r1, r2].

For each r-cell R_k = [r_k, r_{k+1}] and y1-interval Y_k = [m_k - w_k, m_k + w_k]:
  Krawczyk:  K_k = m_k - C Gamma(m_k, R_k) + (1 - C Gamma_y(Y_k, R_k)) (Y_k - m_k)  subset of int Y_k
      => for every r in R_k, Gamma(., r) has exactly one zero in Y_k, and it lies in K_k;
  continuity: K_k subset of Y_{k+1} at the shared endpoint (so the zeros glue into one curve);
  monotonicity: dc/dr = c_r - c_y Gamma_r/Gamma_y > 0 on Y_k x R_k  (c increases along the curve);
  enclosures of c and rho_hat on Y_k x R_k are recorded."""
import numpy as np, math, json, sys, time
from ivec import I
import planar_iv as P
from planar_box import evaluate_box

def krawczyk_batch(rlo, rhi, m, w):
    n = len(rlo)
    R = I(rlo, rhi); rc = R.mid
    Ym = I(m); Y = I(m - w, m + w)
    Vm, Gm, okm, Cm = evaluate_box(Ym, R)               # Gamma(m, R) via mean-value form in r
    VY, GY, okY, CY = evaluate_box(Y, R)                # Gamma_y, c_y, c_r, Gamma_r over the box
    Gy_c = Cm['G_y'].mid if 'G_y' in Cm else None
    Cth = P.evaluate(I(m), I(rc), derivs=True)
    Cinv = 1.0/Cth['G_y'].mid
    Gamma_mR = Vm['Gamma']
    Gy = GY['Gamma'][0]; Gr = GY['Gamma'][1]; cy = GY['c'][0]; cr = GY['c'][1]
    K = Ym - Gamma_mR*Cinv + (1.0 - Gy*Cinv)*(Y - Ym)
    inside = (K.lo > Y.lo) & (K.hi < Y.hi)
    dcdr = cr - cy*Gr/Gy
    ok = okm & okY & Cth['ok'] & inside & (dcdr.lo > 0) & (Gy.hi < 0)
    # enclosures of c and rho_hat over K x R (the zeros lie there) ...
    Kc = I(np.where(inside, K.lo, Y.lo), np.where(inside, K.hi, Y.hi))
    VK, GK, okK, CK = evaluate_box(Kc, R)
    ok &= okK
    # ... and sharper ones along the curve: f(Y(r), r) in f(Y(r_c), r_c) + (df/dr)(Y x R) (R - r_c), where
    # df/dr = f_r - f_y Gamma_r/Gamma_y is the derivative along the curve and Y(r_c) lies in the thin
    # Krawczyk image K_c = m - C Gamma(m, r_c) + (1 - C Gamma_y(Y,R))(Y - m).
    Emc = P.evaluate(I(m), I(rc))
    Gmc = Emc['Gamma']; ok &= Emc['ok']
    Kcen = Ym - Gmc*Cinv + (1.0 - Gy*Cinv)*(Y - Ym)
    Kcen = I(np.maximum(Kcen.lo, Y.lo), np.minimum(Kcen.hi, Y.hi))
    Vc, Gc, okc, Cc = evaluate_box(Kcen, I(rc))
    ok &= okc
    dR = R - I(rc)
    curve = {}; dfd = {}
    for f in ('c', 'rho'):
        fy, fr = GY[f]
        dfdr = fr - fy*Gr/Gy
        dfd[f] = dfdr
        curve[f] = (Vc[f] + dfdr*dR).meet(VK[f])
    return dict(ok=ok, K=K, Y=Y, R=R, dcdr=dcdr, c=curve['c'], rho=curve['rho'], Gy=Gy, inside=inside,
                okm=okm, okY=okY, Cinv=Cinv, drhodr=dfd['rho'])

def _kb(args):
    return krawczyk_batch(*args)

def _batched(lo, hi, m, w, pool, chunk=300):
    parts = [(lo[i:i + chunk], hi[i:i + chunk], m[i:i + chunk], w[i:i + chunk]) for i in range(0, len(lo), chunk)]
    outs = pool.map(_kb, parts) if pool is not None else [krawczyk_batch(*p) for p in parts]
    res = {}
    for key in outs[0]:
        v = [o[key] for o in outs]
        if isinstance(v[0], I): res[key] = I(np.concatenate([x.lo for x in v]), np.concatenate([x.hi for x in v]))
        else: res[key] = np.concatenate(v)
    return res

def _cell(lo, hi, m, w, res, k):
    return dict(rlo=float(lo[k]), rhi=float(hi[k]), m=float(m[k]), w=float(w[k]),
                Klo=float(res['K'].lo[k]), Khi=float(res['K'].hi[k]),
                clo=float(res['c'].lo[k]), chi=float(res['c'].hi[k]),
                rholo=float(res['rho'].lo[k]), rhohi=float(res['rho'].hi[k]),
                dcdr_lo=float(res['dcdr'].lo[k]), Cinv=float(res['Cinv'][k]),
                Gylo=float(res['Gy'].lo[k]), Gyhi=float(res['Gy'].hi[k]),
                drhodr_hi=float(res['drhodr'].hi[k]))

def revalidate(T, pool=None):
    """Re-run every test of krawczyk_batch on the cells of a stored tube (e.g. a cached json) and rebuild the
    cell records from the fresh enclosures.  Returns (cells, number of cells that fail)."""
    lo = np.array([d['rlo'] for d in T]); hi = np.array([d['rhi'] for d in T])
    m = np.array([d['m'] for d in T]); w = np.array([d['w'] for d in T])
    res = _batched(lo, hi, m, w, pool)
    return [_cell(lo, hi, m, w, res, k) for k in range(len(T))], int((~res['ok']).sum())

def build(r1, r2, curve, fac=2.0, dc_cell=0.2, dr_max=0.002, pool=None):
    rg, yg, cg, rhog, dYg, dcg = curve[:, 0], curve[:, 1], curve[:, 2], curve[:, 3], curve[:, 4], curve[:, 5]
    # initial partition
    edges = [r1]
    while edges[-1] < r2:
        r = edges[-1]; dcdr = np.interp(r, rg, dcg)
        cc = np.interp(r, rg, cg)
        edges.append(min(r2, r + min(dr_max, dc_cell*max(1.0, cc/100.0)/dcdr, (0.01 if r < 0.02 else 0.05)*r)))
    cells = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    accepted = []
    rounds = 0
    while cells and rounds < 12:
        rounds += 1
        lo = np.array([c[0] for c in cells]); hi = np.array([c[1] for c in cells])
        # float center of the curve at the midpoint: Newton from interpolation
        rc = 0.5*(lo + hi)
        y0 = np.interp(rc, rg, yg)
        for it in range(6):
            E = P.evaluate(I(y0), I(rc), derivs=True)
            y0 = y0 - E['Gamma'].mid/E['G_y'].mid
        slope = np.abs(np.interp(rc, rg, dYg))
        w = fac*np.maximum(slope, 0.25)*(hi - lo)/2 + 1e-7*y0
        res = _batched(lo, hi, y0, w, pool)
        new_cells = []
        for k in range(len(cells)):
            if res['ok'][k]:
                accepted.append(_cell(lo, hi, y0, w, res, k))
            else:
                mid = 0.5*(lo[k] + hi[k])
                new_cells += [(lo[k], mid), (mid, hi[k])]
        print(f"round {rounds}: tested {len(cells)}, accepted {int(res['ok'].sum())}, split {len(cells) - int(res['ok'].sum())}", flush=True)
        cells = new_cells
    accepted.sort(key=lambda d: d['rlo'])
    return accepted, cells

def _kstar(cells, rs):
    """Krawczyk image at single r values; also returns the ok flags of the thin evaluation of Gamma."""
    m = np.array([d['m'] for d in cells]); w = np.array([d['w'] for d in cells])
    Cinv = np.array([d['Cinv'] for d in cells])
    Gy = I(np.array([d['Gylo'] for d in cells]), np.array([d['Gyhi'] for d in cells]))
    E = P.evaluate(I(m), I(rs))
    return I(m) - E['Gamma']*Cinv + (1.0 - Gy*Cinv)*(I(m - w, m + w) - I(m)), E['ok']

def check_glue(T):
    """At a shared endpoint r* of cells k, k+1: the zero of Gamma(., r*) in Y_k lies in
    K*_k = m_k - C_k Gamma(m_k, r*) + (1 - C_k Gy_k)(Y_k - m_k)  (Gy_k encloses Gamma_y on Y_k x R_k, r* in R_k).
    If K*_k lies inside Y_{k+1}, that zero is also the unique zero in Y_{k+1} (or symmetrically with k, k+1
    exchanged); hence the zero sets of consecutive cells glue into one continuous curve."""
    bad = []
    a = T[:-1]; b = T[1:]
    rs = np.array([d['rhi'] for d in a])
    Ka, oka = _kstar(a, rs); Kb, okb = _kstar(b, rs)
    for k in range(len(a)):
        if a[k]['rhi'] != b[k]['rlo']: bad.append(('gap', a[k]['rhi'])); continue
        if not (oka[k] and okb[k]): bad.append(('evaluation', a[k]['rhi'])); continue
        ya = (a[k]['m'] - a[k]['w'], a[k]['m'] + a[k]['w']); yb = (b[k]['m'] - b[k]['w'], b[k]['m'] + b[k]['w'])
        fwd = Ka.lo[k] > max(ya[0], yb[0]) and Ka.hi[k] < min(ya[1], yb[1])
        bwd = Kb.lo[k] > max(ya[0], yb[0]) and Kb.hi[k] < min(ya[1], yb[1])
        if not (fwd or bwd): bad.append(('glue', a[k]['rhi']))
    return bad

if __name__ == '__main__':
    curve = np.load('curve_float.npy')
    r1, r2 = float(sys.argv[1]), float(sys.argv[2])
    t0 = time.time()
    T, left = build(r1, r2, curve)
    print(f"tube cells: {len(T)}, unresolved: {len(left)}, time {time.time() - t0:.1f}s")
    bad = check_glue(T)
    print("glue/gap problems:", bad[:5], len(bad))
    print("c range on first cell: [%.6f, %.6f]; last cell: [%.6f, %.6f]" % (T[0]['clo'], T[0]['chi'], T[-1]['clo'], T[-1]['chi']))
    print("min dc/dr lower bound over tube: %.4f" % min(d['dcdr_lo'] for d in T))
    print("max rho_hat upper bound over tube: %.7f" % max(d['rhohi'] for d in T))
    json.dump(dict(r1=r1, r2=r2, cells=T), open(sys.argv[3], 'w'))
