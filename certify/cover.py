"""Exclusion cover of the main rectangle  u = ln y1 in [U0, U1],  x = logit r in [-X, X]  (tails: tails.py).

Every cell (mapped to a box Y1 x R) must be resolved by one of:
  cheap criteria C0a..C3 (cheap.py);
  full evaluation (planar_box):
    F0  K1(1) < T* on the box                          -> infeasible
    F0' lower bracket only: c >= 1 + kap K2part + beta0 kap/(-ln z_lo) > c_b   (on rho >= beta0)
    F1  rho_hat > rho_A  or  rho_hat < beta0
    F2  c > c_b or c < c_a
    F3  Gamma > 0 or Gamma < 0
  containment in the validated tube (tube.py).
Unresolved cells are split into four."""
import numpy as np, math, json, sys, time
from multiprocessing import Pool
from ivec import I, ilog
import planar_iv as P
from planar_box import evaluate_box
from cheap import cheap, boxes_from_cells, CODES

PRM = None; TUBE = None
def _init(prm, tube):
    global PRM, TUBE
    PRM = prm; TUBE = tube

def in_tube(Y1, R, tube):
    """Box inside the union of tube boxes Y_k x R_k (r-cells contiguous)."""
    rlo = tube['rlo']; rhi = tube['rhi']; ylo = tube['ylo']; yhi = tube['yhi']
    n = len(Y1.lo); res = np.zeros(n, bool)
    inside_r = (R.lo >= rlo[0]) & (R.hi <= rhi[-1])
    k0 = np.searchsorted(rhi, R.lo, side='left')          # first cell with rhi >= R.lo
    k1 = np.searchsorted(rlo, R.hi, side='right') - 1     # last cell with rlo <= R.hi
    for i in np.nonzero(inside_r)[0]:
        a, b = k0[i], k1[i]
        if a > b: continue
        if np.all(ylo[a:b + 1] <= Y1.lo[i]) and np.all(Y1.hi[i] <= yhi[a:b + 1]): res[i] = True
    return res

FULL_CODES = {10: 'F0 infeasible (K1(1)<T*)', 11: "F0' c>c_b (partial bracket)", 12: 'F1 rho>rho_A', 13: 'F1 rho<beta0',
              14: 'F2 c>c_b', 15: 'F2 c<c_a', 16: 'F3 Gamma>0', 17: 'F3 Gamma<0', 20: 'in tube'}

def process(cells):
    """cells: array (n,4) of (ulo,uhi,xlo,xhi). Returns code per cell (0 = unresolved)."""
    prm, tube = PRM, TUBE
    Y1, R = boxes_from_cells(cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3])
    code, B = cheap(Y1, R, prm)
    code = code.astype(np.int16)
    tub = in_tube(Y1, R, tube)
    code[(code == 0) & tub] = 20
    idx = np.nonzero((code == 0) & (Y1.lo >= P.TMIN) & (B['b'].lo > 0) & (B['lam'].lo > 0))[0]
    if len(idx):
        Ys = I(Y1.lo[idx], Y1.hi[idx]); Rs = I(R.lo[idx], R.hi[idx])
        V, G, ok, C = evaluate_box(Ys, Rs)
        cc = np.zeros(len(idx), np.int16)
        if V['empty'].any():
            cc[V['empty']] = -1                               # never happens for valid enclosures
        one = I(1.0)
        # partial-bracket criteria (valid whatever 'ok' is)
        cc[V['infeas']] = 10
        hl = V['has_lo'] & (cc == 0)
        if hl.any():
            z = I(V['zlo_part'])
            lowc = one + V['kap']*I(np.maximum(V['K2part'].lo, 0.0), np.maximum(V['K2part'].hi, 0.0)) \
                   + prm['beta0']*V['kap']/(-ilog(I(np.where(hl, z.lo, 0.5))))
            cc[hl & (lowc.lo > prm['cb'])] = 11
        g = ok & (cc == 0)
        rho = V['rho']; c = V['c']; Gm = V['Gamma']
        onepeta = I(1.0) + I(prm['eta'])                     # rho_A = 1/(1+eta) exactly
        cc[g & ((I(rho.lo)*onepeta).lo > 1.0)] = 12
        cc[g & (cc == 0) & (rho.hi < prm['beta0'])] = 13
        cc[g & (cc == 0) & (c.lo > prm['cb'])] = 14
        cc[g & (cc == 0) & (c.hi < prm['ca'])] = 15
        cc[g & (cc == 0) & (Gm.lo > 0)] = 16
        cc[g & (cc == 0) & (Gm.hi < 0)] = 17
        code[idx] = cc
    return code

def split(cells):
    a, b, c, d = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
    um = 0.5*(a + b); xm = 0.5*(c + d)
    return np.concatenate([np.column_stack([a, um, c, xm]), np.column_stack([um, b, c, xm]),
                           np.column_stack([a, um, xm, d]), np.column_stack([um, b, xm, d])])

def run(prm, tube, U0, U1, X, n_u=64, n_x=96, max_level=16, chunk=400, procs=10, log=print):
    ue = np.linspace(U0, U1, n_u + 1); xe = np.linspace(-X, X, n_x + 1)
    cells = np.array([(ue[i], ue[i + 1], xe[j], xe[j + 1]) for i in range(n_u) for j in range(n_x)])
    stats = {}; unresolved = []
    t0 = time.time()
    with Pool(procs, initializer=_init, initargs=(prm, tube)) as pool:
        for level in range(max_level + 1):
            if len(cells) == 0: break
            parts = [cells[i:i + chunk] for i in range(0, len(cells), chunk)]
            codes = np.concatenate(pool.map(process, parts))
            if (codes == -1).any(): log(f"!!! EMPTY INTERSECTIONS: {int((codes == -1).sum())} cells"); stats['EMPTY'] = stats.get('EMPTY', 0) + int((codes == -1).sum())
            codes = np.where(codes == -1, 0, codes)
            for k in np.unique(codes):
                if k == 0: continue
                name = CODES.get(int(k), FULL_CODES.get(int(k)))
                stats[name] = stats.get(name, 0) + int((codes == k).sum())
            left = cells[codes == 0]
            extra = ""
            if len(left):
                Yl, Rl = boxes_from_cells(left[:, 0], left[:, 1], left[:, 2], left[:, 3])
                extra = f"  [left: y1 in ({Yl.lo.min():.3g},{Yl.hi.max():.3g}), r in ({Rl.lo.min():.3g},{Rl.hi.max():.3g})]"
            log(f"level {level}: cells {len(cells)}, resolved {int((codes != 0).sum())}, left {len(left)}, "
                f"elapsed {time.time() - t0:.0f}s" + extra)
            if level == max_level: unresolved = left; break
            cells = split(left) if len(left) else left
    return stats, unresolved

if __name__ == '__main__':
    T = json.load(open(sys.argv[1]))['cells']
    tube = dict(rlo=np.array([d['rlo'] for d in T]), rhi=np.array([d['rhi'] for d in T]),
                ylo=np.array([d['m'] - d['w'] for d in T]), yhi=np.array([d['m'] + d['w'] for d in T]))
    eta = float(sys.argv[2]); ca = float(sys.argv[3]); cb = float(sys.argv[4]); ytop = float(sys.argv[5])
    prm = dict(ca=ca, cb=cb, beta0=0.745, eta=eta)
    stats, unres = run(prm, tube, -40.0, math.log(ytop), 30.0, max_level=int(sys.argv[6]) if len(sys.argv) > 6 else 16)
    print("resolved by criterion:", json.dumps(stats, indent=1))
    print("UNRESOLVED:", len(unres))
    if len(unres):
        np.save('unresolved.npy', unres)
        Y1, R = boxes_from_cells(unres[:, 0], unres[:, 1], unres[:, 2], unres[:, 3])
        print(" y1 in [%.4g, %.4g], r in [%.4g, %.4g]" % (Y1.lo.min(), Y1.hi.max(), R.lo.min(), R.hi.max()))
