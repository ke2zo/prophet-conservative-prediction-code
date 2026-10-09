"""Independent check (separate mpmath implementation, indep_planar.py) of Lemma 5.3 (a), (b), (c)
at random points of F, with Gamma from its original definition (with K3)."""
import random, mpmath as mp
from indep_planar import *
print("K =", mp.nstr(K, 20), " beta =", mp.nstr(beta, 20))
random.seed(11)
n = 0; worst = [0, 0, 0]; tried = 0
while n < 40 and tried < 400:
    tried += 1
    y1 = mp.e**mp.mpf(random.uniform(mp.log(1e-5), mp.log(0.7)))
    r = 1/(1 + mp.e**mp.mpf(-random.uniform(-8, 4)))
    V = reduced(y1, r)
    if V is None: continue
    da, db, dc, G = checks(V)
    n += 1
    worst = [max(worst[0], abs(da)), max(worst[1], abs(db)), max(worst[2], abs(dc))]
    if n <= 8:
        print(f"y1={mp.nstr(y1,5):>10} r={mp.nstr(r,5):>10} c={mp.nstr(V['c'],6):>10} Gamma={mp.nstr(G,6):>12} "
              f"(a)err={mp.nstr(da,3):>9} (b)err={mp.nstr(db,3):>9} (c)err={mp.nstr(dc,3):>9}")
print(f"{n} feasible points of {tried}: max errors (a) {mp.nstr(worst[0],3)}, (b) {mp.nstr(worst[1],3)}, (c) {mp.nstr(worst[2],3)}")
