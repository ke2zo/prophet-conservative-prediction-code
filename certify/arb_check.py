"""Cross-check of the certification with a second implementation in ball arithmetic (Arb, through python-flint).

The certification of Theorem 5.5 uses our own interval library (ivec.py) and Taylor-model quadrature
(planar_iv.py, planar_box.py).  This script repeats its tests with arb_eval.py, which shares no code with them:

  tails    the inequalities of Lemma 5.7, (Z1)-(Z4), for every certified piece (all of them);
  points   the values of Table 2: a 2D Krawczyk test for {Gamma = 0, c = c0} at 128 bits; the enclosure of rho(c0)
           must lie inside the interval printed in point_values.txt, and below rho_A of the piece (all 19 widths);
  tube     every cell of every validated tube: the Krawczyk inclusion, Gamma_y < 0, dc/dr > 0, the gluing of
           consecutive cells, c(r_1) < c_a, c(r_2) > c_b, and rho_hat <= rho_A on every cell that meets a piece;
  cover    a random sample of the leaves of the exclusion cover of every piece: each sampled leaf is excluded
           again in Arb (by the criteria of Lemma 5.6 and Section 5.2.2), after subdivision where needed.

The first three parts treat every piece, value and cell; the last one is a sample.  Inputs taken from the
certification: the tube cells, eta, U, y_top, the float curves (starting points) and, for the sample, the leaves
enumerated by cover.py.  Not repeated: the agreement of the tubes on their overlaps (glue_tubes.py) and the fact
that the leaves cover the rectangle.
Usage:  python3 arb_check.py [tails] [points] [tube] [cover]      (default: all; PROCS = number of processes)
python-flint is needed only here; the certification itself does not use it."""
import sys, os, gzip, json, glob, math, time, random
import numpy as np
from flint import arb
import arb_eval as A
from arb_eval import lo, hi, hull, Infeasible, Undecided

BETA0 = arb(0.745)
CERTS = ['cert_1.0001_1.001.json.gz', 'cert_1.001_1000.json.gz', 'cert_1000_3000.json.gz', 'cert_2200_3000.json.gz']

def flo(x):
    """A float <= the lower endpoint of the ball x (-inf if x is not finite)."""
    if not x.is_finite(): return -math.inf
    f = float(lo(x))
    for _ in range(4):
        if arb(f) <= lo(x): return f
        f = math.nextafter(f, -math.inf)
    return -math.inf
def fhi(x):
    if not x.is_finite(): return math.inf
    f = float(hi(x))
    for _ in range(4):
        if arb(f) >= hi(x): return f
        f = math.nextafter(f, math.inf)
    return math.inf
def box(a, b): return hull(arb(a), arb(b))

# ------------------------------------------------------------------------------------------------ tails
def tails(eta, ca, cb, ytop, U, X=30):
    """Lemma 5.7 (Z1)-(Z4) and the covering condition; the same inequalities as tails.py, in Arb."""
    eta = arb(eta); ca = arb(ca); cb = arb(cb); ytop = arb(ytop); ln2 = arb(2).log()
    eX = arb(-X).exp()
    etap = eta*(1 - eX) - arb(-(X + 1)).exp()
    if not etap > 0: return False
    lam_max = (1/etap).log(); z1 = 1/arb(X - 1) + lam_max/X/BETA0
    if not z1 < 1: return False
    ok = bool(eX/(1 - eX)*(lam_max + (X + lam_max)/(1/z1).log()) < ca - 1)                    # (Z1)
    Tmin = 1 - (eX/(1 - eX))/ln2; y1max = eX/(Tmin*ln2)
    ok &= bool(Tmin > 0) and bool(1 + (1 - eX)*eta**2/y1max*Tmin**2 > cb)                     # (Z2)
    L = arb(U); lnipsi = (X + 1) - arb(X).log()
    ok &= bool((-L).exp()*(L + lnipsi)*arb(X + 1).exp() <= eta)                               # (Z3)
    rs = arb('0.31'); g = (rs - 1 - rs.log())/(1 - rs)**2
    ok &= bool(g > 1/ytop) and bool(1 + BETA0*rs*(ytop*ln2)**2/(1/ytop).log() > cb)           # (Z4)
    s = 1/(1 + arb(X).exp())
    ok &= bool(s < eX) and bool(s > arb(-(X + 1)).exp())                                      # the pieces cover
    return ok

# ------------------------------------------------------------------------------------------------ points
def point_value(c0s, y, r, rel=arb(10)**(-22)):
    c0 = arb(c0s); y = arb(y); r = arb(r)
    for it in range(12):                                   # Newton on midpoints (not part of the proof)
        V, G = A.evaluate(y, r)
        F = (V['Gamma'].mid(), V['c'].mid() - c0.mid())
        J = [[G['Gamma'][0].mid(), G['Gamma'][1].mid()], [G['c'][0].mid(), G['c'][1].mid()]]
        det = J[0][0]*J[1][1] - J[0][1]*J[1][0]
        C = [[J[1][1]/det, -J[0][1]/det], [-J[1][0]/det, J[0][0]/det]]
        y = (y - (C[0][0]*F[0] + C[0][1]*F[1])).mid(); r = (r - (C[1][0]*F[0] + C[1][1]*F[1])).mid()
    C = [[x.mid() for x in row] for row in C]
    x = [y, r]; X = [y + arb(0, 1)*rel*y, r + arb(0, 1)*rel*r]
    Vm, _ = A.evaluate(y, r); Fm = [Vm['Gamma'], Vm['c'] - c0]
    V, G = A.evaluate(X[0], X[1]); J = [G['Gamma'], G['c']]
    K = []
    for i in range(2):
        acc = x[i] - (C[i][0]*Fm[0] + C[i][1]*Fm[1])
        for j in range(2):
            acc += ((1 if i == j else 0) - (C[i][0]*J[0][j] + C[i][1]*J[1][j]))*(X[j] - x[j])
        K.append(acc)
    inside = all(lo(K[i]) > lo(X[i]) and hi(K[i]) < hi(X[i]) for i in range(2))
    if not inside: return None
    return A.evaluate(K[0], K[1])[0]['rho'], K

# ------------------------------------------------------------------------------------------------ tube
def _cell(rlo, rhi, m, w, depth=0):
    """All tests on one r-cell [rlo, rhi] with Y = [m - w, m + w].  Returns None (fails) or a dict."""
    ylo, yhi = m - w, m + w
    Y = box(ylo, yhi); R = box(rlo, rhi); am = arb(m)
    try:
        VY, GY = A.evaluate(Y, R)
        Gy, Gr = GY['Gamma']; cy, cr = GY['c']
        if not hi(Gy) < 0: raise Undecided('Gamma_y')
        C = 1/Gy.mid()
        S = (1 - C*Gy)*(Y - am)
        K = am - C*A.evaluate(am, R)[0]['Gamma'] + S
        if not (lo(K) > ylo and hi(K) < yhi): raise Undecided('Krawczyk')
        dcdr = cr - cy*Gr/Gy
        if not lo(dcdr) > 0: raise Undecided('dc/dr')
        rc = R.mid()
        Kc = am - C*A.evaluate(am, rc)[0]['Gamma'] + S
        if not Kc.overlaps(Y): raise RuntimeError('invalid enclosure')
        Vc = A.evaluate(Kc.intersection(Y), rc)[0]
        dR = R - rc
        c = Vc['c'] + dcdr*dR
        rho = Vc['rho'] + (GY['rho'][1] - GY['rho'][0]*Gr/Gy)*dR
        ks = []
        for rr in (rlo, rhi):                              # Krawczyk images at the two ends (for the gluing)
            Ks = am - C*A.evaluate(am, arb(rr))[0]['Gamma'] + S
            ce = A.evaluate(Ks.intersection(Y), arb(rr))[0]['c']
            ks.append((flo(Ks), fhi(Ks), flo(ce), fhi(ce)))
        return dict(clo=flo(c), chi=fhi(c), rholo=flo(rho), rhohi=fhi(rho), dcdr=flo(dcdr), klo=ks[0], khi=ks[1],
                    sub=1)
    except (Undecided, Infeasible):
        if depth >= 5: return None
        mid = 0.5*(rlo + rhi)
        a = _cell(rlo, mid, m, w, depth + 1)
        if a is None: return None
        b = _cell(mid, rhi, m, w, depth + 1)
        if b is None: return None
        # the two halves have the same Y: the unique zero in Y at r = mid is common, nothing to glue
        return dict(clo=min(a['clo'], b['clo']), chi=max(a['chi'], b['chi']), rholo=min(a['rholo'], b['rholo']),
                    rhohi=max(a['rhohi'], b['rhohi']), dcdr=min(a['dcdr'], b['dcdr']), klo=a['klo'], khi=b['khi'],
                    sub=a['sub'] + b['sub'])

def _cells(chunk):
    return [_cell(*c) for c in chunk]

def _refined(rlo, rhi, m, w, f):
    """The cell cut into 2^f parts in r (same Y): the list of (c_lo, c_hi, rho_hi) along the curve, or None."""
    ed = [rlo + (rhi - rlo)*i/2**f for i in range(2**f)] + [rhi]
    out = [_cell(ed[i], ed[i + 1], m, w) for i in range(2**f)]
    return None if any(x is None for x in out) else [(x['clo'], x['chi'], x['rhohi']) for x in out]

_DONE = {}
def check_tube(d, pool, name, log):
    T = d['tube']; n = len(T)
    cells = [(t['rlo'], t['rhi'], t['m'], t['w']) for t in T]
    key = (cells[0], cells[-1], n)
    if key not in _DONE:                                   # two certificate files share one tube
        chunks = [cells[i:i + 200] for i in range(0, n, 200)]
        _DONE[key] = [x for part in pool.map(_cells, chunks) for x in part]
    res = _DONE[key]
    bad = [k for k in range(n) if res[k] is None]
    ok = not bad
    log(f"  {name}: {n} cells, failing: {len(bad)}; cells subdivided in r: {sum(1 for x in res if x and x['sub'] > 1)}")
    if bad: return False
    glue = 0
    for k in range(n - 1):
        a, b = T[k], T[k + 1]
        if a['rhi'] != b['rlo']: glue += 1; continue
        l = max(a['m'] - a['w'], b['m'] - b['w']); h = min(a['m'] + a['w'], b['m'] + b['w'])
        fwd = res[k]['khi'][0] > l and res[k]['khi'][1] < h
        bwd = res[k + 1]['klo'][0] > l and res[k + 1]['klo'][1] < h
        if not (fwd or bwd): glue += 1
    ends = res[0]['klo'][3] < d['ca'] and res[-1]['khi'][2] > d['cb']
    log(f"  {name}: gluing problems: {glue}; min dc/dr >= {min(x['dcdr'] for x in res):.4f}; "
        f"c(r_1) <= {res[0]['klo'][3]:.8f} (< c_a = {d['ca']}), c(r_2) >= {res[-1]['khi'][2]:.6f} (> c_b = {d['cb']}): {ends}")
    ok &= glue == 0 and ends
    # the recorded enclosures of c and rho_hat along the curve must overlap ours; how much wider is which
    incons = sum(1 for k in range(n) if res[k]['chi'] < T[k]['clo'] or res[k]['clo'] > T[k]['chi']
                 or res[k]['rhohi'] < T[k]['rholo'] or res[k]['rholo'] > T[k]['rhohi'])
    log(f"  {name}: cells whose enclosures of c or rho_hat along the curve are disjoint from the recorded ones: {incons}")
    ok &= incons == 0
    for p in d['pieces']:
        if not p['certified']: continue
        onep = 1 + arb(p['eta'])
        def below(x): return not (x[1] >= p['a'] and x[0] <= p['b']) or bool(arb(x[2])*onep < 1)
        meet = [k for k in range(n) if res[k]['chi'] >= p['a'] and res[k]['clo'] <= p['b']]
        rmax = 0.0; nref = 0; good = True
        for k in meet:
            leaves = [(res[k]['clo'], res[k]['chi'], res[k]['rhohi'])]; f = 0
            while not all(below(x) for x in leaves) and f < 6:    # sharper enclosures: cut the cell in r
                f += 1; leaves = _refined(*cells[k], f) or leaves
            nref += f > 0
            good &= all(below(x) for x in leaves)
            rmax = max([rmax] + [x[2] for x in leaves if x[1] >= p['a'] and x[0] <= p['b']])
        log(f"  {name}: piece [{p['a']:g}, {p['b']:g}]: {len(meet)} cells meet it ({nref} cut in r for this test), "
            f"max rho_hat <= {rmax:.10f} < rho_A = {1/(1 + p['eta']):.10f}: {good}")
        ok &= good
    return ok

# ------------------------------------------------------------------------------------------------ cover
def _in_tube(Y, R, tube):
    rlo, rhi, ylo, yhi = tube
    a, b, y0, y1 = flo(R), fhi(R), flo(Y), fhi(Y)
    if not (a >= rlo[0] and b <= rhi[-1]): return False
    k0 = int(np.searchsorted(rhi, a, side='left')); k1 = int(np.searchsorted(rlo, b, side='right')) - 1
    return k0 <= k1 and bool(np.all(ylo[k0:k1 + 1] <= y0) and np.all(y1 <= yhi[k0:k1 + 1]))

def _excluded(Y, R, prm, tube):
    """A reason why the box contains no Euler-Lagrange point with c in [c_a, c_b] and beta0 <= rho_hat <= rho_A
    off the tube, or None."""
    eta, ca, cb = arb(prm['eta']), arb(prm['ca']), arb(prm['cb'])
    try:
        B = A.base(Y, R)
    except Infeasible:
        return 'infeasible'
    except Undecided:
        return None
    S0, Tsp, tau, kap, b = (B[k] for k in ('S0', 'Ts', 'tau', 'kap', 'b'))   # T* >= 0: lambda <= 0 is infeasible
    lamp = hull(arb(max(flo(B['lam']), 0.0)), hi(B['lam']))
    if lo(tau*Tsp) >= hi(1 - Y): return 'infeasible'                 # Lemma 5.6(a)
    if hi(b) <= eta: return 'rho>rho_A'                              # (b): rho_hat > 1/(1 + delta)
    if lo(tau)*BETA0 > 1: return 'rho<beta0'                         # (b): rho_hat <= 1/tau
    if lo(1 + kap*Tsp**2/(1 - Y) + BETA0*kap/(-Y.log())) > cb: return 'c>c_b'            # (c), on rho_hat >= beta0
    z = Y + Tsp/BETA0
    if hi(z) < 1 and hi(1 + R/(1 - R)*(lamp + S0/(-z.log()))) < ca: return 'c<c_a'       # (d), on rho_hat >= beta0
    if _in_tube(Y, R, tube): return 'in tube'
    L = -Y.log(); one = arb(1)
    if hi(L*Tsp) < 1 and lo(tau) > 0:                                # (e): bounds for small T*, no quadrature
        DM = hull(lo(tau), hi(tau)/(1 - hi(L*Tsp))); yMhi = Y + Tsp*DM
        if hi(yMhi) < 1:
            yM = hull(lo(Y), hi(yMhi)); eps = hull(-hi(yMhi).log(), hi(L))
            K2 = hull(arb(max(flo(tau*Tsp/DM**2), 0.0)), hi(Tsp/tau)); K3 = hull(lo(Y*K2), hi(yM*K2))
            M = 1 + kap*K2; P = kap/DM; c = M + P/eps
            G = P*A.E1(eps) + M*(1 - DM) - Y*B['ps'] - kap*K3
            if lo(G) > 0 or hi(G) < 0: return 'Gamma!=0'
            if lo(c) > cb: return 'c>c_b'
            if hi(c) < ca: return 'c<c_a'
            if lo(1/DM)*(1 + eta) > 1: return 'rho>rho_A'
            if hi(1/DM) < BETA0: return 'rho<beta0'
    try:
        V = A.evaluate(Y, R)[0]
    except Infeasible:
        return 'infeasible'
    except Undecided:
        # y_M may be too close to 1 for an enclosure.  A lower bracket is enough: if K1(z) < T* on the box, then
        # y_M > z, so eps < ln(1/z) and, as c = M + kappa rho_hat/eps with M >= 1 and rho_hat >= beta0,
        # c > 1 + beta0 kappa/ln(1/z).
        try:
            a = Y.mid()
            if not lo(B['lam']) > 0: return None
            Dbot = A._Drange(b, hull(Y, a))
            if not lo(b) - A.Phi(a) > 0: return None
            def below(k):                                        # K1(1 - 2^-k) < T* on the whole box
                z = one - arb(2)**(-k)
                return bool(hi(Y) < z) and bool(hi(hull(A._integral(1, 0, hi(b), a, z, 1e-13),
                                                         A._integral(1, 0, lo(b), a, z, 1e-13)) + (a - Y)/Dbot) < lo(Tsp))
            klo, khi = 1, 60                                     # K1 is increasing in z: bisection on k
            if not below(klo): return None
            while khi - klo > 1 and not below(khi):
                km = (klo + khi)//2
                if below(km): klo = km
                else: khi = km
            k = khi if below(khi) else klo
            return 'c>c_b' if lo(1 + BETA0*kap/(-(one - arb(2)**(-k)).log())) > cb else None
        except Undecided:
            pass
        return None
    if lo(V['rho'])*(1 + eta) > 1: return 'rho>rho_A'
    if hi(V['rho']) < BETA0: return 'rho<beta0'
    if lo(V['c']) > cb: return 'c>c_b'
    if hi(V['c']) < ca: return 'c<c_a'
    if lo(V['Gamma']) > 0 or hi(V['Gamma']) < 0: return 'Gamma!=0'
    return None

def _leaf(args):
    """One leaf (u, x cell) of the cover: exclude it, subdividing into four where needed.
    Returns (resolved, number of boxes used)."""
    cell, prm, tube, maxdepth = args
    stack = [(tuple(cell), 0)]; used = 0
    while stack:
        (ul, uh, xl, xh), dep = stack.pop(); used += 1
        Y = hull(arb(ul).exp(), arb(uh).exp())
        R = hull(1/(1 + (-arb(xl)).exp()), 1/(1 + (-arb(xh)).exp()))
        if _excluded(Y, R, prm, tube) is None:
            if dep >= maxdepth or used > 3000: return False, used
            um, xm = 0.5*(ul + uh), 0.5*(xl + xh)
            stack += [((ul, um, xl, xm), dep + 1), ((um, uh, xl, xm), dep + 1), ((ul, um, xm, xh), dep + 1),
                      ((um, uh, xm, xh), dep + 1)]
    return True, used

def sample_leaves(prm, tubeT, U, ytop, rng, cap=1500, per_code=40, max_level=20):
    """Leaves of the cover of one piece, found by the code of the certification (cover.process) on a randomly
    pruned quadtree; up to per_code leaves for each criterion."""
    import cover
    tube = dict(rlo=np.array([d['rlo'] for d in tubeT]), rhi=np.array([d['rhi'] for d in tubeT]),
                ylo=np.array([d['m'] - d['w'] for d in tubeT]), yhi=np.array([d['m'] + d['w'] for d in tubeT]))
    cover._init(prm, tube)
    ue = np.linspace(-float(U), math.log(float(ytop)) + 1e-12, 65); xe = np.linspace(-30.0, 30.0, 97)
    cells = np.array([(ue[i], ue[i + 1], xe[j], xe[j + 1]) for i in range(64) for j in range(96)])
    leaves = {}
    for level in range(max_level + 1):
        if len(cells) == 0: break
        codes = np.concatenate([cover.process(cells[i:i + 400]) for i in range(0, len(cells), 400)])
        for k in np.unique(codes):
            if k > 0: leaves.setdefault(int(k), []).extend(map(tuple, cells[codes == k]))
        left = cells[codes == 0]
        if len(left) > cap: left = left[rng.choice(len(left), cap, replace=False)]
        cells = cover.split(left) if len(left) else left
    out = []
    for k, v in sorted(leaves.items()):
        idx = rng.choice(len(v), min(per_code, len(v)), replace=False)
        out += [(k, v[i]) for i in idx]
    return out, (tube['rlo'], tube['rhi'], tube['ylo'], tube['yhi'])

# ------------------------------------------------------------------------------------------------ main
if __name__ == '__main__':
    from multiprocessing import Pool
    import flint
    parts = sys.argv[1:] or ['tails', 'points', 'tube', 'cover']
    procs = int(os.environ.get('PROCS', 4))
    t00 = time.time()
    def log(s): print(s, flush=True)
    log(f"python-flint {flint.__version__}, precision {A.PREC} bits")
    D = {f: json.load(gzip.open(f)) for f in CERTS}
    pieces = [(f, p) for f in CERTS for p in D[f]['pieces'] if p['certified']]
    allok = True

    if 'tails' in parts:
        log("\nTAILS (Lemma 5.7) for every certified piece")
        for f, p in pieces:
            ok = tails(str(p['eta']), str(p['a']), str(p['b']), p['ytop'], p['U'])
            log(f"  [{p['a']:g}, {p['b']:g}]: eta = {p['eta']}, U = {p['U']}, y_top = {p['ytop']}: {'pass' if ok else 'FAIL'}")
            allok &= ok

    if 'points' in parts:
        log("\nPOINT VALUES (Table 2): Krawczyk for {Gamma = 0, c = c0}, box of relative size 1e-22")
        curves = [np.load(g) for g in ('curve_float.npy', 'curve_float_ext.npy')]
        for line in open('point_values.txt'):
            w = line.split()
            c0s = w[0]; c0 = float(c0s)
            a, b = w[3].strip('[,'), w[4].strip('],')
            cv = next(c for c in curves if c[0, 2] < c0 < c[-1, 2])
            res = point_value(c0s, float(np.interp(c0, cv[:, 2], cv[:, 1])), float(np.interp(c0, cv[:, 2], cv[:, 0])))
            if res is None: log(f"  {c0s:>7}: FAIL (no inclusion)"); allok = False; continue
            rho, K = res
            inside = bool(rho > arb(a) and rho < arb(b))
            onC = any(p['a'] <= c0 <= p['b'] and bool(hi(rho)*(1 + arb(p['eta'])) < 1) for f, p in pieces)
            log(f"  {c0s:>7}: rho = {rho.str(22)}  inside [{a}, {b}]: {inside};  below rho_A of its piece: {onC}")
            allok &= inside and onC

    if 'tube' in parts or 'cover' in parts:
        pool = Pool(procs)

    if 'tube' in parts:
        log("\nTUBE: every cell of every validated tube")
        done = {}
        for f in CERTS:
            key = (D[f]['tube'][0]['rlo'], D[f]['tube'][-1]['rhi'], len(D[f]['tube']))
            t0 = time.time()
            if key in done and all(D[f]['tube'][k] == D[done[key]]['tube'][k] for k in range(len(D[f]['tube']))):
                # same tube as in an earlier file: only the pieces differ
                d = dict(D[done[key]]); d['pieces'] = D[f]['pieces']; d['ca'] = D[done[key]]['ca']; d['cb'] = D[f]['cb']
                log(f"  {f}: its tube is the tube of {done[key]}")
            else:
                d = D[f]; done[key] = f
            ok = check_tube(d, pool, f, log)
            log(f"  {f}: {'pass' if ok else 'FAIL'}  ({time.time() - t0:.0f}s)")
            allok &= ok

    if 'cover' in parts:
        log("\nCOVER: a random sample of the leaves of the exclusion cover of every piece (seed 2026)")
        rng = np.random.default_rng(2026)
        tot = dict(leaves=0, resolved=0, boxes=0)
        for f, p in pieces:
            t0 = time.time()
            prm = dict(ca=p['a'], cb=p['b'], beta0=0.745, eta=p['eta'])
            leaves, tube = sample_leaves(prm, D[f]['tube'], p['U'], p['ytop'], rng)
            res = pool.map(_leaf, [(c, prm, tube, 8) for k, c in leaves], chunksize=10)
            nres = sum(1 for r in res if r[0]); nb = sum(r[1] for r in res)
            unres = {}
            for (k, c), r in zip(leaves, res):
                if not r[0]: unres[k] = unres.get(k, 0) + 1
            log(f"  [{p['a']:g}, {p['b']:g}]: {len(leaves)} leaves sampled, excluded again: {nres}, not resolved: "
                f"{len(leaves) - nres} {unres if unres else ''}; Arb boxes used: {nb}  ({time.time() - t0:.0f}s)")
            tot['leaves'] += len(leaves); tot['resolved'] += nres; tot['boxes'] += nb
            allok &= nres == len(leaves)
        log(f"  total: {tot['leaves']} leaves, excluded again: {tot['resolved']}, Arb boxes: {tot['boxes']}")

    log(f"\n{'ALL CHECKS PASS' if allok else 'SOME CHECKS FAILED'}   (parts: {' '.join(parts)}; time {time.time() - t00:.0f}s)")
