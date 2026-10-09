import math, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reduce2d import reduce
d = np.load('gamma_grid.npz'); ly, lr, G = d['ly'], d['lr'], d['G']
fig, ax = plt.subplots()
cs = ax.contour(lr, ly, np.ma.masked_invalid(G), levels=[0.0])
segs = cs.allsegs[0]
print("number of zero-contour components:", len(segs))
for k, seg in enumerate(segs):
    # seg columns: (log r, log y1)
    cvals = []; rhos = []
    for (b, a) in seg[::max(1, len(seg)//400)]:
        rr = reduce(math.exp(a), math.exp(b))
        cvals.append(rr['c'] if rr else np.nan); rhos.append(rr['rho'] if rr else np.nan)
    cvals = np.array(cvals); rhos = np.array(rhos); ok = np.isfinite(cvals)
    dc = np.diff(cvals[ok])
    mono = "increasing" if np.all(dc > 0) else ("decreasing" if np.all(dc < 0) else f"NOT monotone ({int(np.sum(dc>0))} up, {int(np.sum(dc<0))} down)")
    print(f"component {k}: {len(seg)} points; y1 in [{math.exp(seg[:,1].min()):.3g}, {math.exp(seg[:,1].max()):.3g}], "
          f"r in [{math.exp(seg[:,0].min()):.3g}, {math.exp(seg[:,0].max()):.3g}]; c from {np.nanmin(cvals):.4g} to {np.nanmax(cvals):.4g}; "
          f"rho from {np.nanmin(rhos):.5f} to {np.nanmax(rhos):.5f}; c along the curve is {mono}")
ax.set_xlabel('log r'); ax.set_ylabel('log y1'); fig.savefig('gamma_zero.png', dpi=110)
