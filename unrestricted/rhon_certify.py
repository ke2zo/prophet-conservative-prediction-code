"""Rigorous lower bounds on the horizon-n band constants rho_n(3), n = 3, ..., 6 (manuscript,
Proposition prop:finite), and exact upper bounds from explicit laws.

The reduction to finitely many atoms and the case split j = 0, ..., n-1 (number of thresholds below 1)
are described in rhon_model.py.  In each case we prove f = OPT_n - rho_lo E[M_n] >= 0 on the feasible
part of the coordinate box by branch and bound in interval arithmetic (../certify/ivec.py, outward
rounding).  A box is
  - discarded if some constraint g of the case satisfies g < 0 on all of it (no feasible point);
  - accepted if one of the following enclosures is >= 0 on all of it:
      (1) the natural interval extension of f, or of OPT_n - rho_lo c (then f >= 0 since E[M_n] <= c);
      (2) the mean-value form f(m) + grad f(B).(B - m), with grad f(B) enclosed by forward-mode automatic
          differentiation over intervals;
      (3) the mean-value form of the Lagrangian relaxation L = f - sum_k mu_k g_k, for multipliers
          mu_k >= 0 chosen numerically (any mu >= 0 is valid: at a feasible point g_k >= 0, so f >= L);
          this handles minimizers on the boundary of a case, where f < 0 just outside the feasible set;
  - otherwise split in two, along the coordinate with the largest width times |partial derivative|.
A feasible box midpoint with f < 0 refutes the bound; the program stops there (control runs).
Upper bounds: explicit laws (atoms at 0, 1, and thresholds), OPT_n/E[M_n] in exact rational arithmetic.

Usage: python3 rhon_certify.py                     (n = 3..6, writes rhon_certify.txt)
       python3 rhon_certify.py 78 4                (n = 7, 8 with 4 processes, writes rhon_certify_78.txt)
       python3 rhon_certify.py 78control           (controls for n = 7, 8, appended to rhon_certify_78.txt)
       python3 rhon_certify.py 4 0.858415           (one n, one bound; no output file)
       python3 rhon_certify.py 4 0.858415 --nolag   (the same without the Lagrangian test)"""
import os, sys, time
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from ivec import I, const
from rhon_model import model, domain, dims

C = 3.0


class AD:
    """Forward-mode automatic differentiation over interval arrays."""
    __slots__ = ('v', 'g')
    def __init__(self, v, g): self.v = v; self.g = g
    def __add__(a, b):
        if isinstance(b, AD): return AD(a.v + b.v, [p + q for p, q in zip(a.g, b.g)])
        return AD(a.v + b, a.g)
    __radd__ = __add__
    def __neg__(a): return AD(-a.v, [-p for p in a.g])
    def __sub__(a, b):
        if isinstance(b, AD): return AD(a.v - b.v, [p - q for p, q in zip(a.g, b.g)])
        return AD(a.v - b, a.g)
    def __rsub__(a, b): return (-a) + b
    def __mul__(a, b):
        if isinstance(b, AD): return AD(a.v*b.v, [p*b.v + a.v*q for p, q in zip(a.g, b.g)])
        return AD(a.v*b, [p*b for p in a.g])
    __rmul__ = __mul__
    def __truediv__(a, b):
        if isinstance(b, AD):
            inv = b.v.recip(); val = a.v*inv
            return AD(val, [(p - val*q)*inv for p, q in zip(a.g, b.g)])
        return AD(a.v/b, [p/b for p in a.g])
    def __rtruediv__(a, b):
        inv = a.v.recip(); val = b*inv
        return AD(val, [-(val*inv)*p for p in a.g])
    def __pow__(a, n):
        assert isinstance(n, int) and n >= 1
        if n == 1: return a
        pm1 = a.v**(n - 1)
        return AD(pm1*a.v, [(n*pm1)*p for p in a.g])


def _nnls_batch(A, b, iters=60):
    """min |A mu - b|^2 over mu >= 0 for a batch: A (N, d, K), b (N, d).  Projected gradient; any mu >= 0
    is valid for the certificate, so an approximate solution only costs efficiency."""
    N, d, K = A.shape
    AtA = np.einsum('ndk,ndl->nkl', A, A); Atb = np.einsum('ndk,nd->nk', A, b)
    L = np.einsum('nkk->n', AtA)[:, None] + 1e-300
    mu = np.zeros((N, K))
    for _ in range(iters):
        grad = np.einsum('nkl,nl->nk', AtA, mu) - Atb
        mu = np.maximum(mu - grad/L, 0.0)
    return mu


def certify(n, j, rho, min_width=1e-10, batch=100000, verbose=False, max_boxes=10**9, lagrange=True, box=None):
    lo0, hi0 = domain(n, j, C) if box is None else box
    d = len(lo0)
    stack_lo = [np.array([lo0])]; stack_hi = [np.array([hi0])]
    st = dict(evaluated=0, natural=0, meanvalue=0, lagrange=0, infeasible=0)
    fails = []; t0 = time.time(); rnd = 0
    while stack_lo:
        lo = np.concatenate(stack_lo); hi = np.concatenate(stack_hi)
        stack_lo = []; stack_hi = []
        for k0 in range(0, len(lo), batch):
            Lb = lo[k0:k0 + batch]; Hb = hi[k0:k0 + batch]; N = len(Lb)
            st['evaluated'] += N
            if st['evaluated'] > max_boxes:
                return st, [('too many boxes', None)]
            X = [I(Lb[:, i], Hb[:, i]) for i in range(d)]
            f, cons, extra, _ = model(n, j, X, rho, C)
            infeas = np.zeros(N, bool)
            for g in cons:
                infeas |= g.hi < 0
            ok = f.lo >= 0
            for e in extra:
                ok |= e.lo >= 0
            ok &= ~infeas
            st['infeasible'] += int(infeas.sum()); st['natural'] += int(ok.sum())
            und = np.where(~(ok | infeas))[0]
            split_dim = np.zeros(N, int)
            if len(und):
                Lu = Lb[und]; Hu = Hb[und]; M = len(und)
                m = 0.5*(Lu + Hu)
                r = np.nextafter(np.maximum(m - Lu, Hu - m), np.inf)     # B inside m + [-r, r]
                fm, cm, _, _ = model(n, j, [I(m[:, i]) for i in range(d)], rho, C)
                feas_m = np.ones(M, bool)
                for g in cm:
                    feas_m &= g.lo >= 0
                bad = feas_m & (fm.hi < 0)
                if np.any(bad):
                    k = np.where(bad)[0][0]
                    return st, [('counterexample', m[k].tolist(), float(fm.mid[k]))]
                zero = np.zeros(M); one = np.ones(M)
                Xad = [AD(I(Lu[:, i], Hu[:, i]), [I(one if q == i else zero) for q in range(d)]) for i in range(d)]
                fad, cad, _, _ = model(n, j, Xad, rho, C)
                R = [I(-r[:, i], r[:, i]) for i in range(d)]
                lb = fm
                for i in range(d):
                    lb = lb + fad.g[i]*R[i]
                okm = lb.lo >= 0
                st['meanvalue'] += int(okm.sum())
                # Lagrangian relaxation with the constraints that are not >= 0 on the whole box
                rest = np.where(~okm)[0] if lagrange else np.zeros(0, int)
                okl = np.zeros(M, bool)
                if not lagrange:                                     # split direction from grad f
                    sm = np.stack([np.maximum(np.abs(fad.g[i].lo), np.abs(fad.g[i].hi))*(Hu[:, i] - Lu[:, i])
                                   for i in range(d)], axis=1)
                    fin = np.isfinite(sm)
                    sd = np.argmax(np.where(fin, sm, -1.0), axis=1)
                    split_dim[und] = np.where(fin.all(axis=1), sd, np.argmax(Hu - Lu, axis=1))
                if len(rest):
                    cons_box = [g for g in cons]
                    act = np.stack([cons_box[q].lo[und][rest] < 0 for q in range(len(cons))], axis=1)   # (Mr, K)
                    gf = np.stack([fad.g[i].mid[rest] for i in range(d)], axis=1)                        # (Mr, d)
                    Ag = np.stack([np.stack([cad[q].g[i].mid[rest] for i in range(d)], axis=1)
                                   for q in range(len(cons))], axis=2)                                   # (Mr, d, K)
                    Ag = np.where(act[:, None, :] & np.isfinite(Ag), Ag, 0.0)
                    gf = np.where(np.isfinite(gf), gf, 0.0)
                    mu = _nnls_batch(Ag, gf)
                    mu = np.where(act, mu, 0.0)
                    Lm = fm[rest]; Lg = [fad.g[i][rest] for i in range(d)]
                    for q in range(len(cons)):
                        if not np.any(mu[:, q] > 0): continue
                        mq = mu[:, q]
                        Lm = Lm - cm[q][rest]*mq
                        Lg = [Lg[i] - cad[q].g[i][rest]*mq for i in range(d)]
                    lbl = Lm
                    for i in range(d):
                        lbl = lbl + Lg[i]*R[i][rest]
                    okl[rest] = lbl.lo >= 0
                    st['lagrange'] += int(okl.sum())
                    # split direction from the Lagrangian's gradient
                    sm = np.stack([np.maximum(np.abs(Lg[i].lo), np.abs(Lg[i].hi))*(Hu[rest, i] - Lu[rest, i])
                                   for i in range(d)], axis=1)
                    sm = np.where(np.isfinite(sm), sm, np.inf)
                    sd = np.argmax(sm, axis=1)
                    wid = Hu[rest] - Lu[rest]
                    allinf = ~np.isfinite(sm).any(axis=1) | np.isinf(sm).all(axis=1)
                    sd = np.where(allinf, np.argmax(wid, axis=1), sd)
                    split_dim[und[rest]] = sd
                okc = okm | okl
                ok[und[okc]] = True
            split = np.where(~(ok | infeas))[0]
            if len(split):
                Ls = Lb[split]; Hs = Hb[split]; jd = split_dim[split]
                wall = Hs - Ls
                narrow = wall[np.arange(len(split)), jd] < min_width      # heuristic picked a narrow side:
                jd = np.where(narrow, np.argmax(wall, axis=1), jd)          # split the widest one instead
                tiny = wall.max(axis=1) < min_width
                if np.any(tiny):
                    for k in np.where(tiny)[0][:20]:
                        fails.append((Ls[k].tolist(), Hs[k].tolist()))
                    return st, fails
                mid = 0.5*(Ls[np.arange(len(split)), jd] + Hs[np.arange(len(split)), jd])
                L1 = Ls.copy(); H1 = Hs.copy(); H1[np.arange(len(split)), jd] = mid
                L2 = Ls.copy(); H2 = Hs.copy(); L2[np.arange(len(split)), jd] = mid
                stack_lo += [L1, L2]; stack_hi += [H1, H2]
        rnd += 1
        if verbose:
            pend = sum(len(s) for s in stack_lo)
            print(f"    n={n} j={j} round {rnd}: {st}, pending {pend}  [{time.time() - t0:.1f}s]", flush=True)
    return st, fails


def _certify_part(args):
    n, j, rho_s, lo, hi = args
    return certify(n, j, const(rho_s), box=(lo, hi))


def certify_parallel(n, j, rho_s, nproc=10, parts=64):
    """The same certificate, with the case's box cut into `parts` pieces (by repeated bisection of the widest
    side) that are certified in parallel.  The pieces cover the box, so the conclusion is the same."""
    from multiprocessing import Pool
    lo0, hi0 = domain(n, j, C)
    boxes = [(list(lo0), list(hi0))]
    while len(boxes) < parts:
        lo, hi = boxes.pop(0)
        k = int(np.argmax(np.array(hi) - np.array(lo))); mid = 0.5*(lo[k] + hi[k])
        h1 = list(hi); h1[k] = mid; l2 = list(lo); l2[k] = mid
        boxes += [(lo, h1), (l2, hi)]
    tot = dict(evaluated=0, natural=0, meanvalue=0, lagrange=0, infeasible=0); fails = []
    with Pool(nproc) as pool:
        for st, fl in pool.imap_unordered(_certify_part, [(n, j, rho_s, lo, hi) for lo, hi in boxes]):
            for k, v in st.items(): tot[k] += v
            fails += fl
    return tot, fails


def exact_ratio(atoms, probs, n):
    W = Fr(0)
    for _ in range(n):
        W = sum((p*max(a, W) for a, p in zip(atoms, probs)), Fr(0))
    order = sorted(range(len(atoms)), key=lambda i: atoms[i])
    EM = Fr(0); Fc = Fr(0)
    for i in order:
        EM += atoms[i]*((Fc + probs[i])**n - Fc**n); Fc += probs[i]
    return W/EM


# certified lower bounds, controls (must be refuted), and explicit laws for the upper bounds
LOWER = {3: '0.873303', 4: '0.858415', 5: '0.853851', 6: '0.847779'}
CONTROL = {3: '0.8733036', 4: '0.8584161', 5: '0.8538516', 6: '0.8477798'}
UPPER_LAWS = {
    3: (['0', '1', '1.32065', '3'], ['0.32065', '0.306398', '0.253226', '0.119726']),
    4: (['0', '1', '1.176613', '3'], ['0.517737', '0.273811', '0.131783', '0.076669']),
    5: (['0', '1', '1.122377', '3'], ['0.627802', '0.236763', '0.079086', '0.056349']),
    6: (['0', '1', '1.126235', '1.230558', '3'], ['0.636347', '0.190073', '0.066163', '0.054792', '0.052625']),
}


def self_tests(rng):
    """(1) the reduced model against the dynamic program; (2) AD gradients against central differences;
    (3) interval enclosures on random boxes contain the floating-point values at random points."""
    from rhon_model import law, direct_ratio
    rho = const('0.85'); dev_dp = dev_ad = 0.0; n_dp = n_ad = n_box = 0; bad_box = 0
    for n in (3, 4, 5, 6):
        for j in range(n):
            lo, hi = map(np.array, domain(n, j, C)); d = len(lo)
            for _ in range(1000):
                x = rng.uniform(lo, hi)
                f, cons, _, info = model(n, j, list(x), 0.0, C)
                if all(g >= 0 for g in cons) and x[0] > 0:
                    at, pr = law(n, j, x, C)
                    if j < n - 1:
                        r_model = info['OPT']/info['EM']
                    else:       # f/a at rho = 0 is OPT/a; E[M]/a = f/a at rho = 0 minus f/a at rho = 1
                        r_model = f/(f - model(n, j, list(x), 1.0, C)[0])
                    dev_dp = max(dev_dp, abs(r_model - direct_ratio(at, pr, n))); n_dp += 1
                # AD gradient at the point vs central differences, away from the pole D = c - W_{n-1} = 0 of
                # T/D (where central differences are inaccurate)
                if j == n - 1 or C - info['W'][n - 2] >= 0.05:
                    xs = [I(np.array([xi])) for xi in x]
                    g_ad = model(n, j, [AD(xs[i], [I(np.array([1.0 if q == i else 0.0])) for q in range(d)])
                                        for i in range(d)], rho, C)[0].g
                    for i in range(d):
                        h = 1e-6; xp = x.copy(); xp[i] += h; xm = x.copy(); xm[i] -= h
                        fd = (model(n, j, list(xp), 0.85, C)[0] - model(n, j, list(xm), 0.85, C)[0])/(2*h)
                        dev_ad = max(dev_ad, abs(fd - g_ad[i].mid[0])/(1 + abs(fd))); n_ad += 1
                # a random box around x and a random point in it
                w = rng.uniform(0, 0.05, d)*(hi - lo)
                bl = np.maximum(lo, x - w); bh = np.minimum(hi, x + w)
                pt = rng.uniform(bl, bh)
                fb, cb, _, _ = model(n, j, [I(np.array([bl[i]]), np.array([bh[i]])) for i in range(d)], rho, C)
                fp_, cp_, _, _ = model(n, j, [I(np.array([t])) for t in pt], rho, C)
                n_box += 1
                for B, P in [(fb, fp_)] + list(zip(cb, cp_)):
                    if np.isfinite(B.lo[0]) and np.isfinite(B.hi[0]) and not (B.lo[0] <= P.lo[0] and P.hi[0] <= B.hi[0]):
                        bad_box += 1
                # gradient enclosures over the box contain the gradient at the point (f and every constraint)
                one_ = np.ones(1); zero_ = np.zeros(1)
                ab = model(n, j, [AD(I(np.array([bl[i]]), np.array([bh[i]])),
                                     [I(one_ if q == i else zero_) for q in range(d)]) for i in range(d)], rho, C)
                ap = model(n, j, [AD(I(np.array([pt[i]])), [I(one_ if q == i else zero_) for q in range(d)])
                                  for i in range(d)], rho, C)
                for GB, GP in [(ab[0], ap[0])] + list(zip(ab[1], ap[1])):
                    for i in range(d):
                        B = GB.g[i]; P = GP.g[i]
                        if np.isfinite(B.lo[0]) and np.isfinite(B.hi[0]) and not (B.lo[0] <= P.lo[0] and P.hi[0] <= B.hi[0]):
                            bad_box += 1
    return (f"self-tests: reduced model vs dynamic program at {n_dp} feasible random points, max deviation of the "
            f"ratio {dev_dp:.1e}; AD vs central differences at {n_ad} gradient components, max relative deviation "
            f"{dev_ad:.1e}; {n_box} random boxes, enclosures (values and gradients) missing the value at a random "
            f"point: {bad_box}"), \
        dev_dp < 1e-9 and dev_ad < 1e-5 and bad_box == 0


LOWER_78 = {7: '0.845355', 8: '0.842265'}
CONTROL_78 = {7: '0.8453560', 8: '0.8422660'}
UPPER_LAWS_78 = {
    7: (['0', '1', '1.095868', '1.1796028', '3'], ['0.7027116', '0.1707257', '0.0456362', '0.0386509', '0.0422756']),
    8: (['0', '1', '1.098169', '1.1821', '1.257266', '3'], ['0.708441', '0.146514', '0.040619', '0.034578', '0.02973', '0.040118']),
}


CONTROL_CASE_78 = {7: 3, 8: 4}      # a case containing the numerical minimizer (a threshold equals 1 there)


def run_78(nproc):
    """Horizons 7 and 8, certification with the parallel driver and the exact upper bounds; writes
    rhon_certify_78.txt.  The controls are a separate step (run_78_control), appended to the same file."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    out = []
    def say(t): print(t, flush=True); out.append(t)
    for n in (7, 8):
        t0 = time.time(); tot = {}
        for j in range(n):
            st, fails = certify_parallel(n, j, LOWER_78[n], nproc=nproc, parts=8*nproc)
            assert not fails, (n, j, fails[:1])
            for k, v in st.items(): tot[k] = tot.get(k, 0) + v
            print(f"   n={n} case j={j}: {st}  [{time.time() - t0:.0f}s]", flush=True)
        say(f"n={n}: OPT_n - {LOWER_78[n]} E[M_n] >= 0 certified in all {n} cases: {tot['evaluated']} boxes evaluated "
            f"({tot['natural']} accepted by the interval extension, {tot['meanvalue']} by the mean-value form, "
            f"{tot['lagrange']} by the Lagrangian relaxation, {tot['infeasible']} infeasible) [{time.time() - t0:.0f}s]")
        atoms, probs = UPPER_LAWS_78[n]
        A = [Fr(a) for a in atoms]; P = [Fr(q) for q in probs]; assert sum(P) == 1 and min(P) > 0
        R = exact_ratio(A, P, n)
        say(f"      upper bound: atoms {atoms}, probabilities {probs}: OPT_{n}/E[M_{n}] = {float(R):.10f} (exact)")
        say(f"      hence {LOWER_78[n]} <= rho_{n}(3) <= {float(R):.10f}")
    open('rhon_certify_78.txt', 'w').write("\n".join(out) + "\n")


def run_78_control():
    """Controls for n = 7, 8: the bounds CONTROL_78 are refuted (sequential branch and bound on the case that
    contains the numerical minimizer, stopping at the first feasible box midpoint with f < 0)."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    out = []
    for n in (7, 8):
        j = CONTROL_CASE_78[n]; t0 = time.time()
        st, fails = certify(n, j, const(CONTROL_78[n]))
        cex = [f for f in fails if f[0] == 'counterexample']
        assert cex, (n, fails[:1])
        t = (f"      control n={n}: the bound {CONTROL_78[n]} is refuted (case j={j}, feasible point "
             f"{[round(v, 6) for v in cex[0][1]]} with f = {cex[0][2]:.1e}; {st['evaluated']} boxes, {time.time() - t0:.0f}s)")
        print(t, flush=True); out.append(t)
    open('rhon_certify_78.txt', 'a').write("\n".join(out) + "\n")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '78':
        run_78(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == '78control':
        run_78_control()
        sys.exit(0)
    if len(sys.argv) > 2:
        n = int(sys.argv[1]); rho = const(sys.argv[2])
        tot = 0; t0 = time.time()
        for j in range(n):
            st, fails = certify(n, j, rho, verbose='-v' in sys.argv, lagrange='--nolag' not in sys.argv)
            tot += st['evaluated']
            print(f"n={n} rho_lo={sys.argv[2]} case j={j} (dim {dims(n, j)}): {st}; failures {fails[:1]}  "
                  f"[{time.time() - t0:.1f}s]", flush=True)
        print(f"total boxes evaluated: {tot}")
        sys.exit(0)
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # 0*inf -> NaN only leaves boxes undecided
    out = []
    def say(t): print(t, flush=True); out.append(t)
    msg, ok = self_tests(np.random.default_rng(0)); say(msg); assert ok
    for n in (3, 4, 5, 6):
        t0 = time.time(); tot = {}
        for j in range(n):
            st, fails = certify(n, j, const(LOWER[n]))
            assert not fails, (n, j, fails)
            for k, v in st.items(): tot[k] = tot.get(k, 0) + v
        say(f"n={n}: OPT_n - {LOWER[n]} E[M_n] >= 0 certified in all {n} cases: {tot['evaluated']} boxes evaluated "
            f"({tot['natural']} accepted by the interval extension, {tot['meanvalue']} by the mean-value form, "
            f"{tot['lagrange']} by the Lagrangian relaxation, {tot['infeasible']} infeasible) [{time.time() - t0:.0f}s]")
        atoms, probs = UPPER_LAWS[n]
        A = [Fr(a) for a in atoms]; P = [Fr(p) for p in probs]; assert sum(P) == 1 and min(P) > 0
        R = exact_ratio(A, P, n)
        say(f"      upper bound: atoms {atoms}, probabilities {probs}: OPT_{n}/E[M_{n}] = {float(R):.10f} (exact)")
        say(f"      hence {LOWER[n]} <= rho_{n}(3) <= {float(R):.10f}")
        refuted = None
        for j in range(n):
            st, fails = certify(n, j, const(CONTROL[n]))
            if fails and fails[0][0] == 'counterexample':
                refuted = (j, fails[0][1], fails[0][2]); break
        assert refuted is not None
        say(f"      control: the bound {CONTROL[n]} is refuted (case j={refuted[0]}, feasible point "
            f"{[round(v, 6) for v in refuted[1]]} with f = {refuted[2]:.1e})")
    open('rhon_certify.txt', 'w').write("\n".join(out) + "\n")
