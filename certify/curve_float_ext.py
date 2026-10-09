"""Extended float continuation of the zero curve (input for the tube cells of the extended certification,
c in [1.0001, 1e4]; not part of the proof).  Same method as curve_float.py, on r in [1e-5, 0.84]."""
import numpy as np
from curve_float import coarse, newton_vec, thin

if __name__ == '__main__':
    pts = coarse(0.1605454, 0.23648, 0.845) + coarse(0.1605454, 0.23648, 8e-6)[1:]
    pts.sort()
    pr = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
    rg = np.concatenate([np.geomspace(1e-5, 0.05, 3000, endpoint=False), np.linspace(0.05, 0.84, 9000)])
    y0 = np.interp(rg, pr, py)
    y, E = newton_vec(y0, rg)
    E = thin(y, rg)
    c = E['c'].mid; rho = E['rho'].mid
    dYdr = -E['G_r'].mid/E['G_y'].mid; dcdr = E['c_r'].mid + E['c_y'].mid*dYdr
    A = np.column_stack([rg, y, c, rho, dYdr, dcdr, E['Gamma'].mid, E['ok']])
    np.save('curve_float_ext.npy', A)
    print("grid points:", len(A), " all ok:", bool(E['ok'].all()), " max|Gamma|:", np.abs(E['Gamma'].mid).max(),
          " c range:", c.min(), c.max(), " min dc/dr:", dcdr.min())
