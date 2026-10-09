"""Driver: interval certification that rho(c) is the continuation value for every c in [c_a, c_b].

Steps (see the manuscript, Section 5.2):
  1. validated tube around the zero curve {Gamma=0} for r in [r_1, r_2]  (tube.py): Krawczyk in y1 with r
     an interval, gluing of consecutive cells, dc/dr > 0, and c(r_1) < c_a, c(r_2) > c_b;
  2. the range [c_a, c_b] is split into pieces [a, b]; on each piece rho_A := 1/(1+eta) with eta chosen so
     that rho_hat <= rho_A at every curve point of the tube with c in [a, b];
  3. tail lemmas (tails.py) outside the main rectangle, for each piece;
  4. exclusion cover of the main rectangle (cover.py), for each piece: no zero of Gamma off the tube with
     c in [a, b] and beta0 <= rho_hat <= rho_A.
Usage: python3 run_certify.py c_a c_b [split points ...]
Environment: CURVE (float curve file), PROCS (pool size), TUBE (cached tube file to reuse), MAXLEVEL (maximal
quadtree level of the cover, default 18)."""
import sys, json, math, time, numpy as np
from ivec import I, iexp
import planar_iv as P
from planar_box import evaluate_box
import tube as TB
import tails
import cover

def certify_piece(a, b, T, tube):
    meet = [d for d in T if d['chi'] >= a and d['clo'] <= b]
    rmax = max(d['rhohi'] for d in meet)
    eta = math.floor((1.0/(rmax*(1 + 2e-6)) - 1.0)*1e6)/1e6
    chk = (I(rmax)*(I(1.0) + I(eta))).hi < 1.0
    print(f"\n==== piece c in [{a}, {b}]: max rho_hat on the curve = {rmax:.7f};  eta = {eta}  "
          f"(rho_A = 1/(1+eta) = {1/(1+eta):.7f}; rho_hat <= rho_A verified: {bool(chk)})", flush=True)
    assert chk and eta > 0
    done = False
    for U in (40, 45, 50, 60):
        for ytop in ('0.995', '0.999', '0.9999', '0.99999', '0.999999'):
            if tails.check(str(eta), str(a), str(b), ytop, U=U, quiet=True): done = True; break
        if done: break
    if not done: raise SystemExit("tails failed")
    print(f"tails pass with U = {U}, y_top = {ytop}", flush=True)
    tails.check(str(eta), str(a), str(b), ytop, U=U)
    U1 = math.log(float(ytop)) + 1e-12
    assert iexp(I(U1)).lo >= float(ytop)
    prm = dict(ca=a, cb=b, beta0=0.745, eta=eta)
    import os
    stats, unres = cover.run(prm, tube, -float(U), U1, 30.0, max_level=int(os.environ.get('MAXLEVEL', 18)),
                             procs=int(os.environ.get('PROCS', 10)),
                             log=lambda s: print(s, flush=True))
    print("cover: resolved by criterion:", json.dumps(stats), flush=True)
    print(f"cover: UNRESOLVED CELLS: {len(unres)}", flush=True)
    ok = len(unres) == 0 and 'EMPTY' not in stats
    return dict(a=a, b=b, eta=eta, rho_A=1/(1 + eta), U=U, ytop=ytop, stats=stats, certified=ok,
                cells=int(sum(stats.values())))

def main(ca, cb, splits):
    t0 = time.time()
    import os
    curve = np.load(os.environ.get('CURVE', 'curve_float.npy'))   # CURVE=curve_float_ext.npy for c outside [1.001, 1000]
    rg, cg = curve[:, 0], curve[:, 2]
    assert cg[0] < ca and cg[-1] > cb, "float curve does not cover the requested range"
    ca_m = 1.0 + 0.97*(ca - 1.0) if ca < 1.1 else ca - 0.002*ca
    r1 = float(np.interp(ca_m, cg, rg)); r2 = float(np.interp(cb*1.02, cg, rg))
    r1 = float(f"{r1:.6f}"); r2 = float(f"{r2:.6f}")
    print(f"== c in [{ca}, {cb}]:  tube r in [{r1}, {r2}]", flush=True)
    import os
    cache = os.environ.get('TUBE', f'tube_{r1}_{r2}_{P.TMIN}.json')   # TUBE: reuse a cached tube covering [c_a, c_b]
    if os.path.exists(cache):
        T = json.load(open(cache)); left = []
        T = T if isinstance(T, list) else T['cells']
        r1, r2 = T[0]['rlo'], T[-1]['rhi']
        from multiprocessing import Pool
        with Pool(int(os.environ.get('PROCS', 10))) as pool:
            T, nbad = TB.revalidate(T, pool=pool)
        print(f"tube loaded from {cache} and re-validated: r in [{r1}, {r2}], {len(T)} cells, {nbad} failing", flush=True)
        assert nbad == 0, "cached tube fails re-validation"
    else:
        from multiprocessing import Pool
        with Pool(int(os.environ.get('PROCS', 10))) as pool:
            T, left = TB.build(r1, r2, curve, pool=pool)
        json.dump(T, open(cache, 'w'))
    assert not left, "tube: unresolved r-cells"
    glue = TB.check_glue(T)
    print(f"tube: {len(T)} cells, glue problems: {len(glue)}, min dc/dr >= {min(d['dcdr_lo'] for d in T):.4f}, "
          f"max drho/dr <= {max(d['drhodr_hi'] for d in T):.4f}", flush=True)
    assert not glue
    ends = []
    for d, rr in ((T[0], T[0]['rlo']), (T[-1], T[-1]['rhi'])):
        Ks, oks = TB._kstar([d], np.array([rr]))
        V, G, ok, C = evaluate_box(Ks, I(np.array([rr])))
        assert oks[0] and ok[0]
        ends.append((float(V['c'].lo[0]), float(V['c'].hi[0])))
    print(f"c(r_1) in [{ends[0][0]:.6f}, {ends[0][1]:.6f}]  (< c_a: {ends[0][1] < ca});  "
          f"c(r_2) in [{ends[1][0]:.6f}, {ends[1][1]:.6f}]  (> c_b: {ends[1][0] > cb})", flush=True)
    assert ends[0][1] < ca and ends[1][0] > cb
    tube = dict(rlo=np.array([d['rlo'] for d in T]), rhi=np.array([d['rhi'] for d in T]),
                ylo=np.array([d['m'] - d['w'] for d in T]), yhi=np.array([d['m'] + d['w'] for d in T]))
    pts = [ca] + [s for s in splits if ca < s < cb] + [cb]
    pieces = [certify_piece(a, b, T, tube) for a, b in zip(pts[:-1], pts[1:])]
    ok = all(p['certified'] for p in pieces)
    band = [(d['clo'], d['chi'], d['rholo'], d['rhohi']) for d in T]
    import gzip
    with gzip.open(f'cert_{ca:g}_{cb:g}.json.gz', 'wt') as f:
        json.dump(dict(ca=ca, cb=cb, r1=r1, r2=r2, c_r1=ends[0], c_r2=ends[1], tube=T, band=band,
                       pieces=pieces, certified=ok), f)
    print("\nSUMMARY")
    for p in pieces:
        print(f"  c in [{p['a']}, {p['b']}]: rho_A = {p['rho_A']:.6f}, U = {p['U']}, y_top = {p['ytop']}, "
              f"cells = {p['cells']}, certified = {p['certified']}")
    print(("CERTIFIED" if ok else "NOT CERTIFIED") + f": c in [{ca}, {cb}]   (total time {time.time() - t0:.0f}s)")

if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
