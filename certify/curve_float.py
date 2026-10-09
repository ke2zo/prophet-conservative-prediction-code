"""Float continuation of the zero curve {Gamma = 0}: a coarse sequential trace, then vectorized
Newton on a fine r-grid.  Uses the midpoints of the thin interval evaluator (planar_iv)."""
import numpy as np, math
from ivec import I
import planar_iv as P

def thin(y, r):
    E = P.evaluate(I(np.atleast_1d(np.asarray(y, float))), I(np.atleast_1d(np.asarray(r, float))), derivs=True)
    return E

def newton_vec(y, r, iters=8):
    y = np.array(y, float)
    for it in range(iters):
        E = thin(y, r)
        dy = -E['Gamma'].mid/E['G_y'].mid
        dy = np.where(E['ok'], dy, 0.0)
        y = y + dy
        if np.all(np.abs(dy) < 1e-15*np.maximum(1.0, y)): break
    return y, E

def coarse(r0, y0, r_end, h=0.004):
    pts = []; r = r0; y = y0
    sgn = 1 if r_end > r0 else -1
    while sgn*(r_end - r) > 0:
        ys, E = newton_vec([y], [r])
        y = ys[0]
        dYdr = -E['G_r'].mid[0]/E['G_y'].mid[0]
        pts.append((r, y))
        hh = min(h, 0.01*y/abs(dYdr), 0.2*r)
        r_new = r + sgn*hh
        y = y + dYdr*(r_new - r); r = r_new
    return pts

if __name__ == '__main__':
    pts = coarse(0.1605454, 0.23648, 0.76) + coarse(0.1605454, 0.23648, 4e-5)[1:]
    pts.sort()
    pr = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
    rg = np.concatenate([np.geomspace(5e-5, 0.05, 2000, endpoint=False), np.linspace(0.05, 0.75, 7000)])
    y0 = np.interp(rg, pr, py)
    y, E = newton_vec(y0, rg)
    E = thin(y, rg)
    c = E['c'].mid; rho = E['rho'].mid
    dYdr = -E['G_r'].mid/E['G_y'].mid; dcdr = E['c_r'].mid + E['c_y'].mid*dYdr
    A = np.column_stack([rg, y, c, rho, dYdr, dcdr, E['Gamma'].mid, E['ok']])
    np.save('curve_float.npy', A)
    print("grid points:", len(A), " all ok:", bool(E['ok'].all()), " max|Gamma|:", np.abs(E['Gamma'].mid).max())
    for cc in (1.02, 1.05, 1.1, 1.25, 1.5, 2, 5, 10, 20, 50, 100, 200, 300):
        k = np.argmin(abs(c - cc))
        print(f"c~{c[k]:9.4f}: r={rg[k]:.6f} y1={y[k]:.6f} rho={rho[k]:.7f} dY/dr={dYdr[k]:+.4f} dc/dr={dcdr[k]:10.3f}")
    print("min dc/dr along curve:", dcdr.min(), " at r =", rg[np.argmin(dcdr)])
