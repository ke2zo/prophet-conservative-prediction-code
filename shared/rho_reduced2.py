"""
EXACT reduced continuum ratio, with s represented as a PIECEWISE-CONSTANT tail-rate
(natural for atomic distributions and for the worst-case which has atoms at endpoints).

Tail-rate s(x), nonincreasing, on x>=0:
  - s(x)=S0 for x in [0,1)            (no positive mass below 1; only atom at 0)
  - on [1,c): piecewise constant, levels x[0]=1<x[1]<...<x[K]=c, with s = sval[i] on
    [x[i], x[i+1]).  sval nonincreasing, sval ends at sval[K-1] = rate of the top atom at c.
  - s(x)=0 for x>=c.
Atoms (Poisson rates): atom at 0 carries the rest; atom at x[i] (i>=1) has rate sval[i-1]-sval[i]
(drop); atom at c has rate sval[K-1]; atom at 1 has rate S0 - sval[0]. (All >=0 by monotonicity.)

Reduced formulas (derived, exact):
  E[M] = c - INT_0^c e^{-s(x)} dx
       = c - e^{-S0}*1   (x in [0,1))   - SUM_i e^{-sval[i]} * (x[i+1]-x[i])   (x in [1,c))
  R(w) = INT_w^c s(x) dx   (piecewise linear in w)
  ALG solves INT_0^{ALG} dw/R(w) = 1.

VALIDATE vs discrete DP, then minimize ratio over the piecewise-constant s -> rho(c).
"""
import numpy as np
from scipy import integrate, optimize
np.seterr(all='ignore')

def kertz_beta():
    def eq(K):
        f=lambda y:1.0/(y*(1.0-np.log(y))+(K-1.0))
        return integrate.quad(f,0,1,limit=400)[0]-1.0
    return 1.0/optimize.brentq(eq,1.0001,3.0)
BETA=kertz_beta()

class PWS:
    """piecewise-constant tail rate: breakpoints xb (ascending, xb[0]=1, xb[-1]=c),
    sval (len = len(xb)-1) = s on [xb[i],xb[i+1]); plus S0 = s on [0,1)."""
    def __init__(self, xb, sval, S0):
        self.xb=np.asarray(xb,float); self.sval=np.asarray(sval,float); self.S0=float(S0)
        self.c=self.xb[-1]
        # R at breakpoints: R(xb[i]) = INT_{xb[i]}^c s = sum_{j>=i} sval[j]*(xb[j+1]-xb[j])
        w=np.diff(self.xb)*self.sval
        self.R_break=np.concatenate([np.cumsum(w[::-1])[::-1],[0.0]])  # len = len(xb)
    def R(self,w):
        c=self.c
        if w>=c: return 0.0
        if w<=1.0:
            return self.S0*(1.0-w)+self.R_break[0]
        i=int(np.searchsorted(self.xb,w,side='right')-1)
        i=max(0,min(i,len(self.sval)-1))
        return self.sval[i]*(self.xb[i+1]-w)+self.R_break[i+1]
    def Emax(self):
        c=self.c
        block0=np.exp(-self.S0)*1.0
        seg=np.sum(np.exp(-self.sval)*np.diff(self.xb))
        return c - block0 - seg
    def alg(self):
        c=self.c
        # INT_0^{ALG} dw/R(w)=1. R(w) piecewise linear, positive on [0,c). Integrate
        # analytically per piece: on a piece where R(w)=a-b*w (b=s slope), INT dw/(a-bw)=
        # -(1/b)ln(a-bw). Build the pieces.
        # pieces: [0,1] with R=S0(1-w)+R_break[0] => R= (S0+R_break[0]) - S0 w, slope b=S0.
        # then [xb[i],xb[i+1]] with R = sval[i](xb[i+1]-w)+R_break[i+1]
        #      = (sval[i] xb[i+1]+R_break[i+1]) - sval[i] w, slope b=sval[i].
        pieces=[]  # (wlo,whi, a, b) with R=a-b w
        pieces.append((0.0,1.0, self.S0+self.R_break[0], self.S0))
        for i in range(len(self.sval)):
            a=self.sval[i]*self.xb[i+1]+self.R_break[i+1]; b=self.sval[i]
            pieces.append((self.xb[i],self.xb[i+1],a,b))
        # cumulative G up to each breakpoint, find where G crosses 1.
        def piece_int(wlo,whi,a,b):
            # INT_wlo^whi dw/(a-b w)
            if abs(b)<1e-14:
                return (whi-wlo)/max(a,1e-15)
            Rlo=a-b*wlo; Rhi=a-b*whi
            Rlo=max(Rlo,1e-15); Rhi=max(Rhi,1e-15)
            return (1.0/b)*np.log(Rlo/Rhi)
        G=0.0
        for (wlo,whi,a,b) in pieces:
            inc=piece_int(wlo,whi,a,b)
            if G+inc>=1.0:
                # solve within this piece: G + INT_wlo^A dw/(a-bw)=1
                need=1.0-G
                if abs(b)<1e-14:
                    A=wlo+need*max(a,1e-15)
                else:
                    Rlo=max(a-b*wlo,1e-15)
                    # (1/b)ln(Rlo/(a-bA))=need => a-bA = Rlo*exp(-b*need)
                    A=(a-Rlo*np.exp(-b*need))/b
                return min(A,self.c)
            G+=inc
        return self.c   # saturates (gambler reaches top)
    def ratio(self):
        em=self.Emax()
        if em<=1e-12: return 1.0
        return self.alg()/em

# ---- discrete DP cross-check ----
def Emax_dp(vals,probs,n):
    o=np.argsort(vals);v=vals[o];p=probs[o];cum=np.cumsum(p);cb=cum-p
    return float(np.sum(v*(cum**n-cb**n)))
def alg_dp(vals,probs,n):
    W=0.0;v=np.asarray(vals,float);p=np.asarray(probs,float)
    for _ in range(n):W=W+float(np.dot(p,np.maximum(v-W,0.0)))
    return W

if __name__=="__main__":
    print(f"beta={BETA:.8f}\n")
    print("cross-check reduced-continuum vs discrete DP, 2 positive atoms {1,B}:")
    print("  (s = S0=r1+rB on [0,1); s=rB on [1,B); 0 above)")
    for (B,r1,rB) in [(50.,1.,.05),(10.,2.,.2),(1000.,1.,.01),(5.,3.,.5),(2.,5.,2.)]:
        m=PWS(xb=[1.0,B], sval=[rB], S0=r1+rB)
        rr=m.ratio()
        for n in [8000,32000,128000]:
            p1=r1/n;pB=rB/n
            vals=np.array([0.,1.,B]);probs=np.array([1-p1-pB,p1,pB])
            dd=alg_dp(vals,probs,n)/Emax_dp(vals,probs,n)
        print(f"  B={B:7.1f} r1={r1} rB={rB}: reduced={rr:.6f}  DP(n={n})={dd:.6f}  diff={rr-dd:+.1e}")
