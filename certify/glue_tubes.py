"""Gluing of two certified tubes on their common r-range (extended certification, Theorem thm:cert).
Each tube gives, for every r in its range, the unique zero Z(r) of Gamma(., r) in the y1-interval Y of the
cell containing r; Z is continuous (tube.check_glue).  On the common range I, the set {r in I: Z_A(r) = Z_B(r)}
is closed, and it is open: if Z_A(r') = Z_B(r'), this point lies in the interior of Y_a for every cell a of A
containing r', so Z_B(r) lies in Y_a for r near r' and equals Z_A(r) by uniqueness.  Hence one common point
suffices.  We check it at a few r* in I: the Krawczyk image K*(r*) of the B-cell containing r* (an enclosure of
Z_B(r*), tube._kstar) lies inside Y_a for the A-cell containing r*.
Usage: python3 glue_tubes.py A B   (tube caches tube_*.json or certification records cert_*.json.gz)"""
import sys, json
import numpy as np
import tube as TB

def load(p):
    """a tube cache (tube_*.json) or a certification record (cert_*.json.gz, which contains its tube)"""
    if p.endswith('.gz'):
        import gzip
        T = json.load(gzip.open(p))['tube']
    else:
        T = json.load(open(p)); T = T if isinstance(T, list) else T['cells']
    return sorted(T, key=lambda d: d['rlo'])

def cell_at(T, r):
    for d in T:
        if d['rlo'] < r < d['rhi']: return d
    return None

if __name__ == '__main__':
    A = load(sys.argv[1]); B = load(sys.argv[2])
    lo = max(A[0]['rlo'], B[0]['rlo']); hi = min(A[-1]['rhi'], B[-1]['rhi'])
    print(f"tube A: r in [{A[0]['rlo']}, {A[-1]['rhi']}], {len(A)} cells; tube B: r in [{B[0]['rlo']}, {B[-1]['rhi']}], "
          f"{len(B)} cells; common range [{lo}, {hi}]")
    ok_any = False
    for t in (0.25, 0.5, 0.75):
        rs = lo + t*(hi - lo)
        a = cell_at(A, rs); b = cell_at(B, rs)
        Kb, okb = TB._kstar([b], np.array([rs]))
        inside = bool(okb[0]) and bool(a['m'] - a['w'] < Kb.lo[0] and Kb.hi[0] < a['m'] + a['w'])
        ok_any |= inside
        print(f"  r* = {rs:.8f}: K*_B = [{Kb.lo[0]:.12f}, {Kb.hi[0]:.12f}] inside Y_A = [{a['m'] - a['w']:.12f}, "
              f"{a['m'] + a['w']:.12f}]: {inside}")
    print("the two zero curves coincide on the common range" if ok_any else "NOT SHOWN")
