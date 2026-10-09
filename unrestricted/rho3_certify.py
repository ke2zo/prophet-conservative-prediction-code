"""Rigorous two-sided bounds on the horizon-3 band constant rho_3(3) (manuscript, finite horizons).

rho_n(c) = inf over band laws F (positive support in [1, c], mass at 0 allowed) of OPT_n(F)/E[M_n].
With W_1 = E[X], W_2 = E[max(X, W_1)] and OPT_3 = E[max(X, W_2)], averaging F between consecutive
points of {1, W_1, W_2, c} keeps W_1, W_2, OPT_3 and does not decrease E[M_3] = int (1 - F^3)
(Jensen). So it suffices to consider step laws with atoms at 0 and at points of {1, W_1, W_2, c}.
Write u = W_1, v = W_2 (so v = u(1 + F(0)) whenever u <= 1 or the law has no atom in (0, u)).
Three cases cover everything:
  A  1 <= u:      atoms 0, 1, u, v, c.  On [1, u) the mass at 1 can be removed (keeping F(0) + (u-1)F(1)
                  fixed, which keeps u, v, OPT; Jensen again), so atoms 0, u, v, c.
                  Coordinates (u, F0, F2): F0 = F on [0, u), F2 = F on [u, v), v = u(1 + F0),
                  F3 = F on [v, c) = 1 - u F0 F2/(c - v);  OPT = u(1 + F0 + F0 F2).
  B  u < 1 <= v:  atoms 0, 1, v, c.  Coordinates (u, F0, F1): v = u(1 + F0),
                  s = (c - v)(1 - F3) = u - (1 - F0) - (v - 1)(1 - F1);  OPT = v + s.
  C  v < 1:       atoms 0, 1, c.  Coordinates a = 1 - F0 in (0, 1], t = (1 - F1)/a in [0, 1];
                  u = a(1 + (c-1)t), v = u(2 - a); we certify (OPT - rho E[M])/a.
In each case we certify  f = OPT - rho_lo * E[M] >= 0  on the whole (constrained) coordinate box by
branch and bound with interval arithmetic (outward rounding, ../certify/ivec.py): a box is discarded
if a defining constraint (probabilities >= 0, the case conditions) is violated on all of it, and
accepted if the natural interval extension or the mean-value form (gradient enclosed by forward-mode
automatic differentiation over intervals) gives f >= 0, or if v >= rho_lo*c there (then f >= 0 because
OPT >= v and E[M] <= c).  The upper bound comes from an explicit law, evaluated in exact rational
arithmetic."""
import os, sys, time
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'certify'))
from ivec import I, const

C = 3.0          # band width (exact binary)
RHO_LO = '0.8733'


class AD:
    """Forward-mode automatic differentiation over interval arrays."""
    __slots__ = ('v', 'g')
    def __init__(self, v, g): self.v = v; self.g = g
    @staticmethod
    def _l(x):
        return x if isinstance(x, AD) else None
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
            inv = b.v.recip()
            val = a.v*inv
            return AD(val, [(p - val*q)*inv for p, q in zip(a.g, b.g)])
        inv = I(1.0)/b if not isinstance(b, I) else b.recip()
        return AD(a.v*inv, [p*inv for p in a.g])
    def __rtruediv__(a, b):
        inv = a.v.recip()
        val = b*inv
        return AD(val, [-(val*inv)*p for p in a.g])
    def __pow__(a, n):
        assert n == 3
        sq = a.v*a.v
        return AD(sq*a.v, [(3*sq)*p for p in a.g])


def case_A(x, rho):
    u, F0, F2 = x
    v = u*(1 + F0)
    D = C - v
    s = u*F0*F2
    F3 = 1 - s/D
    OPT = u*(1 + F0 + F0*F2)
    EM = u*(1 - F0**3) + u*F0*(1 - F2**3) + D*(1 - F3**3)
    # p_u >= 0, p_v >= 0 (F3 >= F2, multiplied by c - v > 0), v <= c
    cons = [F2 - F0, D*(1 - F2) - s, D]
    # sufficient condition: OPT >= v and E[M] <= c, so v >= rho*c implies f >= 0
    return OPT - EM*rho, cons, [v - rho*C]

def case_B(x, rho):
    u, F0, F1 = x
    v = u*(1 + F0)
    D = C - v
    s = u - (1 - F0) - (v - 1)*(1 - F1)            # = p_c (c - v)
    F3 = 1 - s/D
    OPT = v + s
    EM = (1 - F0**3) + (v - 1)*(1 - F1**3) + D*(1 - F3**3)
    # p_1, p_c >= 0, p_v >= 0 (F3 >= F1, multiplied by c - v > 0), v >= 1, v <= c
    cons = [F1 - F0, s, D*(1 - F1) - s, v - 1, D]
    return OPT - EM*rho, cons, [v - rho*C]

def case_C(x, rho):
    a, t = x
    F0 = 1 - a
    F1 = 1 - a*t
    u = a*(1 + (C - 1)*t)
    v = u*(2 - a)
    q0 = 1 + F0 + F0*F0
    q1 = 1 + F1 + F1*F1
    ft = (1 + (C - 1)*t)*q0 - (q0 + (C - 1)*t*q1)*rho   # (OPT - rho E[M]) / a
    cons = [1 - v]                                         # v <= 1
    return ft, cons, []

CASES = {
    'A': (case_A, [1.0, 0.0, 0.0], [C, 1.0, 1.0]),
    'B': (case_B, [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]),
    'C': (case_C, [0.0, 0.0], [1.0, 1.0]),
}


def certify(name, rho, min_width=1e-9, batch=200000, verbose=True):
    f, lo0, hi0 = CASES[name]
    d = len(lo0)
    stack_lo = [np.array([lo0])]; stack_hi = [np.array([hi0])]
    n_cert = n_infeas = n_eval = 0; fails = []
    t0 = time.time(); it = 0
    while stack_lo:
        lo = np.concatenate(stack_lo); hi = np.concatenate(stack_hi)
        stack_lo = []; stack_hi = []
        for k0 in range(0, len(lo), batch):
            L = lo[k0:k0 + batch]; H = hi[k0:k0 + batch]
            n_eval += len(L)
            X = [I(L[:, i], H[:, i]) for i in range(d)]
            val, cons, extra = f(X, rho)
            infeas = np.zeros(len(L), bool)
            for g in cons:
                infeas |= g.hi < 0
            ok = val.lo >= 0
            for e in extra:
                ok |= e.lo >= 0
            # mean-value form for the undecided boxes
            und = ~(ok | infeas)
            if np.any(und):
                Lu = L[und]; Hu = H[und]
                m = 0.5*(Lu + Hu)
                # radius rounded outward, so that every box lies inside m + [-r, r]
                r = np.nextafter(np.maximum(m - Lu, Hu - m), np.inf)
                Xm = [I(m[:, i]) for i in range(d)]
                vm, _, _ = f(Xm, rho)
                # a feasible midpoint with f < 0 refutes the bound: stop at once
                _, cm, _ = f(Xm, rho)
                feas_m = np.ones(len(Lu), bool)
                for g in cm:
                    feas_m &= g.lo >= 0
                bad = feas_m & (vm.hi < 0)
                if np.any(bad):
                    k = np.where(bad)[0][0]
                    return n_eval, n_cert, n_infeas, [('counterexample', m[k])]
                zero = np.zeros(len(Lu)); one = np.ones(len(Lu))
                Xad = [AD(I(Lu[:, i], Hu[:, i]), [I(one if j == i else zero) for j in range(d)]) for i in range(d)]
                vad, _, _ = f(Xad, rho)
                lb = vm
                for i in range(d):
                    lb = lb + vad.g[i]*I(-r[:, i], r[:, i])
                okm = lb.lo >= 0
                ok_idx = np.where(und)[0][okm]
                ok[ok_idx] = True
            n_cert += int(np.sum(ok & ~infeas)); n_infeas += int(np.sum(infeas))
            split = ~(ok | infeas)
            if np.any(split):
                Ls = L[split]; Hs = H[split]
                w = Hs - Ls
                j = np.argmax(w, axis=1)
                tiny = w[np.arange(len(w)), j] < min_width
                if np.any(tiny):
                    for k in np.where(tiny)[0]:
                        fails.append((Ls[k].copy(), Hs[k].copy()))
                    Ls = Ls[~tiny]; Hs = Hs[~tiny]; j = j[~tiny]
                mid = 0.5*(Ls[np.arange(len(Ls)), j] + Hs[np.arange(len(Ls)), j])
                L1 = Ls.copy(); H1 = Hs.copy(); H1[np.arange(len(Ls)), j] = mid
                L2 = Ls.copy(); H2 = Hs.copy(); L2[np.arange(len(Ls)), j] = mid
                stack_lo += [L1, L2]; stack_hi += [H1, H2]
        it += 1
        if verbose:
            pend = sum(len(s) for s in stack_lo)
            print(f"  case {name} round {it}: evaluated {n_eval}, certified {n_cert}, infeasible {n_infeas}, "
                  f"pending {pend}, failed {len(fails)}  [{time.time() - t0:.1f}s]", flush=True)
    return n_eval, n_cert, n_infeas, fails


def exact_ratio(atoms, probs, n=3):
    """OPT_n / E[M_n] in exact rational arithmetic."""
    W = Fr(0)
    for _ in range(n):
        W = sum((p*max(a, W) for a, p in zip(atoms, probs)), Fr(0))
    order = sorted(range(len(atoms)), key=lambda i: atoms[i])
    EM = Fr(0); Fc = Fr(0)
    for i in order:
        EM += atoms[i]*((Fc + probs[i])**n - Fc**n); Fc += probs[i]
    return W, EM, W/EM


if __name__ == '__main__':
    out = []
    if len(sys.argv) > 1 and sys.argv[1] == '--sanity':
        # the procedure must refute a bound slightly above the true value 0.8733033...
        for name in ('A', 'B'):
            ne, nc, ni, fails = certify(name, const('0.87331'), verbose=False)
            print(f"sanity, rho_lo = 0.87331, case {name}: {fails[:1]}")
        sys.exit(0)
    rho = const(RHO_LO)
    # self-test of the automatic differentiation: AD gradients vs central differences at random points
    rng = np.random.default_rng(0); worst = 0.0
    for name, (f, lo0, hi0) in CASES.items():
        d = len(lo0)
        for _ in range(200):
            x = rng.uniform(lo0, hi0)
            xs = [I(np.array([xi])) for xi in x]
            g_ad = f([AD(xs[i], [I(np.array([1.0 if j == i else 0.0])) for j in range(d)]) for i in range(d)], rho)[0].g
            for i in range(d):
                h = 1e-6
                xp = x.copy(); xp[i] += h; xm = x.copy(); xm[i] -= h
                fp = f([I(np.array([t])) for t in xp], rho)[0].mid[0]
                fm = f([I(np.array([t])) for t in xm], rho)[0].mid[0]
                fd = (fp - fm)/(2*h)
                if np.isfinite(fd) and abs(fd) < 1e6:
                    worst = max(worst, abs(fd - g_ad[i].mid[0])/(1 + abs(fd)))
    s0 = f"AD self-test: max relative deviation of AD gradients from central differences {worst:.1e} (600 points)"
    print(s0); out.append(s0)
    assert worst < 1e-5
    for name in ('C', 'B', 'A'):
        ne, nc, ni, fails = certify(name, rho)
        s = (f"case {name}: OPT_3 - {RHO_LO} E[M_3] >= 0 certified on {nc} boxes ({ni} infeasible, "
             f"{ne} evaluated); failed boxes: {len(fails)}")
        print(s, flush=True); out.append(s)
        assert not fails
    # upper bound: an explicit law close to the minimizer (atoms 0, 1, 1 + p0, 3; then E[X] = 1)
    p0 = Fr(32065, 100000); p1 = Fr(30640, 100000); pc = Fr(11973, 100000); pv = 1 - p0 - p1 - pc
    atoms = [Fr(0), Fr(1), 1 + p0, Fr(3)]; probs = [p0, p1, pv, pc]
    W, EM, R = exact_ratio(atoms, probs)
    s = (f"upper bound: the law with atoms 0, 1, {float(1 + p0)}, 3 and probabilities {[str(p) for p in probs]} "
         f"has OPT_3/E[M_3] = {float(R):.10f} (exact rational)")
    print(s); out.append(s)
    s = f"hence {RHO_LO} <= rho_3(3) <= {float(R):.10f}"
    print(s); out.append(s)
    for name in ('A', 'B'):
        ne, nc, ni, fails = certify(name, const('0.87331'), verbose=False)
        s = f"control: the bound 0.87331 is refuted in case {name}: feasible point {fails[0][1].tolist()} with f < 0"
        print(s); out.append(s)
    open('rho3_certify.txt', 'w').write("\n".join(out) + "\n")
