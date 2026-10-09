"""Vectorized interval arithmetic on numpy float64 arrays.

Every basic operation (+, -, *, /) is computed in round-to-nearest and the result is widened
outward by one ulp (np.nextafter).  Since IEEE-754 round-to-nearest has error <= 1/2 ulp, the
widened interval contains the exact result.  Logarithms, exponentials and decimal constants
(`const(s)`) are computed in mpmath's interval arithmetic at 80 bits, element by element; each
endpoint is rounded outward to a double and then moved one more ulp outward.  The extra ulp
absorbs any error of the 80-bit endpoints up to a relative 2^-54, about 2^26 units of their last
place, so the enclosures do not depend on mpmath's directed rounding being exact."""
import numpy as np
from mpmath import iv, libmp
iv.prec = 80
_NI, _PI = -np.inf, np.inf

def _dn(x): return np.nextafter(x, _NI)
def _up(x): return np.nextafter(x, _PI)

class I:
    __slots__ = ('lo', 'hi')
    __array_priority__ = 1000
    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(lo if hi is None else hi, dtype=float)
    # ---------------------------------------------------------------- basics
    @property
    def mid(self): return 0.5*(self.lo + self.hi)
    @property
    def rad(self): return 0.5*(self.hi - self.lo)
    @property
    def wid(self): return self.hi - self.lo
    def __getitem__(self, k): return I(self.lo[k], self.hi[k])
    def __len__(self): return len(self.lo)
    def __repr__(self): return f"I({self.lo}, {self.hi})"
    def copy(self): return I(self.lo.copy(), self.hi.copy())
    # ---------------------------------------------------------------- arithmetic
    def __add__(a, b):
        b = _c(b); return I(_dn(a.lo + b.lo), _up(a.hi + b.hi))
    __radd__ = __add__
    def __sub__(a, b):
        b = _c(b); return I(_dn(a.lo - b.hi), _up(a.hi - b.lo))
    def __rsub__(a, b): return _c(b) - a
    def __neg__(a): return I(-a.hi, -a.lo)
    def __mul__(a, b):
        b = _c(b)
        p1 = a.lo*b.lo; p2 = a.lo*b.hi; p3 = a.hi*b.lo; p4 = a.hi*b.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return I(_dn(lo), _up(hi))
    __rmul__ = __mul__
    def recip(a):
        bad = (a.lo <= 0) & (a.hi >= 0)
        lo = np.where(bad, _NI, _dn(1.0/np.where(bad, 1.0, a.hi)))
        hi = np.where(bad, _PI, _up(1.0/np.where(bad, 1.0, a.lo)))
        return I(lo, hi)
    def __truediv__(a, b):
        b = _c(b)
        bad = (b.lo <= 0) & (b.hi >= 0)
        bl = np.where(bad, 1.0, b.lo); bh = np.where(bad, 1.0, b.hi)
        p1 = a.lo/bl; p2 = a.lo/bh; p3 = a.hi/bl; p4 = a.hi/bh
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return I(np.where(bad, _NI, _dn(lo)), np.where(bad, _PI, _up(hi)))
    def __rtruediv__(a, b): return _c(b)/a
    def sqr(a):
        l2 = a.lo*a.lo; h2 = a.hi*a.hi
        lo = np.where((a.lo <= 0) & (a.hi >= 0), 0.0, _dn(np.minimum(l2, h2)))
        return I(np.maximum(lo, 0.0), _up(np.maximum(l2, h2)))
    def __pow__(a, n):
        assert isinstance(n, int) and n >= 1
        if n == 1: return a
        if n == 2: return a.sqr()
        h = a**(n//2); h = h.sqr()
        return h*a if n % 2 else h
    def abs_hi(a): return np.maximum(np.abs(a.lo), np.abs(a.hi))
    def hull(a, b):
        b = _c(b); return I(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))
    def meet(a, b):
        b = _c(b); return I(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))

def _c(x):
    if isinstance(x, I): return x
    x = np.asarray(x, dtype=float)
    return I(x, x)

# endpoints of an 80-bit mpmath interval as doubles: rounded outward, then one more ulp outward
# (float() would truncate toward zero; to_float is not directed at gradual underflow and overflow, which the
# extra ulp covers: nextafter(+-inf) is the largest finite double)
def _lo80(v): return _dn(libmp.to_float(v._mpi_[0], rnd=libmp.round_floor))
def _hi80(v): return _up(libmp.to_float(v._mpi_[1], rnd=libmp.round_ceiling))

def const(s):
    """Rigorous enclosure of a decimal constant given as a string."""
    v = iv.mpf(s)
    return I(_lo80(v), _hi80(v))

def ilog(a):
    """Rigorous log of an interval array (log is increasing)."""
    lo = np.atleast_1d(a.lo); hi = np.atleast_1d(a.hi)
    assert np.all(lo > 0), "log of nonpositive interval"
    L = np.empty_like(lo); H = np.empty_like(hi)
    for k in range(lo.size):
        L.flat[k] = _lo80(iv.log(iv.mpf(lo.flat[k])))
        H.flat[k] = _hi80(iv.log(iv.mpf(hi.flat[k])))
    if np.ndim(a.lo) == 0: return I(L[0], H[0])
    return I(L, H)

def iexp(a):
    lo = np.atleast_1d(a.lo); hi = np.atleast_1d(a.hi)
    L = np.empty_like(lo); H = np.empty_like(hi)
    for k in range(lo.size):
        L.flat[k] = _lo80(iv.exp(iv.mpf(lo.flat[k])))
        H.flat[k] = _hi80(iv.exp(iv.mpf(hi.flat[k])))
    if np.ndim(a.lo) == 0: return I(L[0], H[0])
    return I(L, H)

def where(mask, a, b):
    a = _c(a); b = _c(b)
    return I(np.where(mask, a.lo, b.lo), np.where(mask, a.hi, b.hi))

if __name__ == '__main__':
    # self-test against mpmath.iv on random data; endpoints are compared exactly (as mpmath numbers)
    import mpmath
    from mpmath.libmp import from_float, mpf_le
    def inside(z, Z, k):    # the 80-bit mpmath interval z lies in [Z.lo[k], Z.hi[k]]
        return mpf_le(from_float(float(Z.lo[k])), z._mpi_[0]) and mpf_le(z._mpi_[1], from_float(float(Z.hi[k])))
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 3, 200); y = rng.uniform(0.1, 3, 200); w = rng.uniform(0, 1e-3, 200)
    X = I(x - w, x + w); Y = I(y, y + w)
    ok = True
    for op, f in (('+', lambda a, b: a + b), ('-', lambda a, b: a - b), ('*', lambda a, b: a*b), ('/', lambda a, b: a/b)):
        Z = f(X, Y)
        for k in range(200):
            if not inside(f(iv.mpf([X.lo[k], X.hi[k]]), iv.mpf([Y.lo[k], Y.hi[k]])), Z, k):
                ok = False; print("FAIL", op, k)
    for name, F, f, A in (('log', ilog, iv.log, X), ('exp', iexp, iv.exp, X), ('exp', iexp, iv.exp, -X - 5.0)):
        Z = F(A)
        for k in range(200):
            if not inside(f(iv.mpf([A.lo[k], A.hi[k]])), Z, k):
                ok = False; print("FAIL", name, k)
    with mpmath.workprec(200):
        # independent reference: plain mpmath at 200 bits (not the interval context)
        for v in np.concatenate([x - w, x + w]):
            v = float(v); L = ilog(I(v)); E = iexp(I(v)); E2 = iexp(I(-5.0 - v))
            t = mpmath.log(v); u = mpmath.exp(v); u2 = mpmath.exp(-5.0 - v)
            ok &= bool(float(L.lo) <= t <= float(L.hi) and float(E.lo) <= u <= float(E.hi) and float(E2.lo) <= u2 <= float(E2.hi))
        for c in ('0.1', '0.001', '1.0001', '2.2', '1e-7', '-0.3'):
            C = const(c); ok &= bool(float(C.lo) <= mpmath.mpf(c) <= float(C.hi))
        # the extra ulp absorbs an error of up to a relative 2^-54 of the 80-bit endpoints (on the wrong side),
        # for generic values and next to powers of 2
        d = mpmath.mpf(2)**-54; vals = []
        for s, t, e in zip(rng.uniform(1, 2, 300), rng.uniform(-1, 1, 300), rng.integers(-60, 60, 300)):
            vals += [mpmath.ldexp(mpmath.mpf(s) + mpmath.mpf(t)*2.0**-60, int(e)), mpmath.ldexp(1 + mpmath.mpf(t)*2.0**-60, int(e))]
        vals += [-v for v in vals]
        for v in vals:
            ok &= bool(float(_lo80(iv.mpf(v + d*abs(v)))) <= v <= float(_hi80(iv.mpf(v - d*abs(v)))))
        # control: the same test detects the truncating conversion float() followed by one ulp
        ok &= any(not bool(float(_dn(float(iv.mpf(v + d*abs(v)).a))) <= v <= float(_up(float(iv.mpf(v - d*abs(v)).b))))
                  for v in vals)
    print("ivec self-test:", "OK" if ok else "FAILED")
