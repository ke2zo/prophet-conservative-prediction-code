"""Self-check: box enclosures (planar_box.evaluate_box) contain the values of the independent
floating-point evaluator float_ref.evalp at random points of random boxes.  Boxes are drawn
log-uniformly in y1 in [1e-6, 0.6] and uniformly in logit r in [-9, 3] (r in [1.2e-4, 0.95]), with
relative sizes 1e-6 .. 3e-2; only boxes on which the evaluator succeeds are tested."""
import sys, math, numpy as np
from ivec import I
from float_ref import evalp
from planar_box import evaluate_box

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
NB, NP = 400, 12
tested = boxes_ok = viol = 0
worst = {}
for batch in range(5):
    yc = np.exp(rng.uniform(math.log(1e-6), math.log(0.6), NB))
    rc = 1/(1 + np.exp(-rng.uniform(-9, 3, NB)))
    h = np.exp(rng.uniform(math.log(1e-6), math.log(3e-2), NB))
    Y = I(yc*(1 - h), yc*(1 + h)); R = I(rc*(1 - h), np.minimum(rc*(1 + h), 0.999))
    V, G, ok, C = evaluate_box(Y, R)
    for k in np.nonzero(ok)[0]:
        boxes_ok += 1
        for t in range(NP):
            y = rng.uniform(Y.lo[k], Y.hi[k]); r = rng.uniform(R.lo[k], R.hi[k])
            f = evalp(y, r)
            if not f['feas']: continue
            tested += 1
            for key in ('c', 'Gamma', 'rho'):
                X = V[key][k]
                if not (X.lo <= f[key] <= X.hi):
                    viol += 1; print("VIOLATION", key, y, r, f[key], X)
print(f"boxes with successful enclosure: {boxes_ok}; points tested: {tested}; violations: {viol}")
