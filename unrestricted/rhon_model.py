"""The reduced band laws for rho_n(c) at a fixed horizon n (manuscript, Proposition prop:finite).

rho_n(c) = inf over band laws F (positive support in [1, c], mass at 0 allowed) of OPT_n(F)/E[M_n].
The gambler's thresholds are W_1 = E[X] and W_{k+1} = W_k + T(W_k), where T(t) = int_t^c (1 - F);
OPT_n = W_n.  Averaging F between consecutive points of {1, W_1, ..., W_{n-1}, c} keeps F constant
on [0, 1), keeps T at every threshold (hence all W_k and OPT_n), and by Jensen does not decrease
E[M_n] = int_0^c (1 - F^n).  Let j be the number of thresholds below 1.  Then it suffices to
consider the following laws (one case for each j = 0, ..., n-1):

  j = 0        atoms 0, W_1, ..., W_{n-1}, c   (if W_1 >= 1 the atom at 1 is averaged away as well)
               coordinates u = W_1 in [1, c], G_0, ..., G_{n-2} in [0, 1]  (G_k = F on [W_k, W_{k+1}),
               G_0 = F on [0, W_1)); then T(W_1) = u G_0, T(W_{k+1}) = T(W_k) G_k.
  1<=j<=n-2    atoms 0, 1, W_{j+1}, ..., W_{n-1}, c
               coordinates u = W_1 in [1/(j+1), 1], G_0 = F on [0, 1), H = F on [1, W_{j+1}),
               G_{j+1}, ..., G_{n-2}; below 1, W_{k+1} = u + G_0 W_k, so W_k = u (1 + G_0 + ... + G_0^{k-1})
               for k <= j+1; T(1) = u - (1 - G_0), T(W_{j+1}) = T(1) - (W_{j+1} - 1)(1 - H).
  j = n-1      atoms 0, 1, c;  coordinates a = 1 - F(0) in [0, 1], t = P(X = c | X > 0) in [0, 1];
               f is divided by a (the degenerate corner a -> 0 costs nothing).

In every case the CDF on the last interval [W_{n-1}, c) is determined by D (1 - G_{n-1}) = T(W_{n-1}),
D = c - W_{n-1} (at the corner D = 0 the law is the point mass at c, with ratio 1; boxes near it are
accepted through OPT_n - rho c >= 0).  model() returns f = OPT_n - rho E[M_n] (divided by a when
j = n-1), the list of constraints g >= 0 that define the case (monotone CDF with values in [0, 1],
W_j <= 1 <= W_{j+1} (only W_{n-1} <= 1 when j = n-1), T(W_{j+1}) >= 0, D >= 0), and the
sufficient condition OPT_n - rho c >= 0 (then f >= 0 because E[M_n] <= c).  All formulas use only
+, -, *, / and integer powers, so they evaluate on floats, on intervals (../certify/ivec.py) and on
interval automatic differentiation alike."""
import math


def dims(n, j):
    return n if j == 0 else (n + 1 - j if j <= n - 2 else 2)


def domain(n, j, c):
    if j == 0:
        return [1.0] + [0.0]*(n - 1), [float(c)] + [1.0]*(n - 1)
    if j <= n - 2:
        d = n + 1 - j
        # W_1 >= 1/(j+1); the float bound is rounded down, so that the box contains every law of the case
        return [math.nextafter(1.0/(j + 1), 0.0)] + [0.0]*(d - 1), [1.0] + [1.0]*(d - 1)
    return [0.0, 0.0], [1.0, 1.0]


def geom(G, k):
    """1 + G + ... + G^(k-1)."""
    s = 1.0 + 0.0*G
    p = 1.0 + 0.0*G
    for _ in range(k - 1):
        p = p*G
        s = s + p
    return s


def model(n, j, x, rho, c, one=1.0):
    """x: list of coordinates (floats, intervals or AD numbers).  Returns (f, cons, extra, info)."""
    cons = []
    if j == n - 1:
        a, t = x
        G0 = one - a
        H = one - a*t
        u = a*(one + (c - 1)*t)
        cons.append(one - u*geom(G0, n - 1))           # W_{n-1} <= 1
        SG = geom(G0, n); SH = geom(H, n)
        f = (one + (c - 1)*t)*SG - (SG + (c - 1)*t*SH)*rho
        return f, cons, [], {'W': [u*geom(G0, k) for k in range(1, n + 1)]}
    if j == 0:
        u = x[0]; G = list(x[1:])                        # G_0 .. G_{n-2}
        W = u; Ws = [u]
        EM = u*(one - G[0]**n)
        T = u*G[0]                                       # T(W_1)
        prev = G[0]
        for k in range(1, n - 1):                        # interval [W_k, W_{k+1}) with CDF G_k
            EM = EM + T*(one - G[k]**n)
            cons.append(G[k] - prev); prev = G[k]
            W = W + T; Ws.append(W)
            T = T*G[k]
    else:
        u, G0, H = x[0], x[1], x[2]; G = list(x[3:])      # G_{j+1} .. G_{n-2}
        Wj = u*geom(G0, j)
        W = u*geom(G0, j + 1)                            # W_{j+1}
        cons += [one - Wj, W - one, H - G0]
        Ws = [u*geom(G0, k) for k in range(1, j + 2)]
        T = (u - (one - G0)) - (W - one)*(one - H)       # T(W_{j+1})
        cons.append(T)
        EM = (one - G0**n) + (W - one)*(one - H**n)
        prev = H
        for k in range(len(G)):                          # intervals [W_{j+1+k}, W_{j+2+k}) with CDF G[k]
            EM = EM + T*(one - G[k]**n)
            cons.append(G[k] - prev); prev = G[k]
            W = W + T; Ws.append(W)
            T = T*G[k]
    # last interval [W_{n-1}, c): D (1 - G_{n-1}) = T
    D = c - W
    cons += [D*(one - prev) - T, D]
    Gl = one - T/D
    EM = EM + D*(one - Gl**n)
    OPT = W + T
    Ws.append(OPT)
    return OPT - EM*rho, cons, [OPT - rho*c], {'W': Ws, 'OPT': OPT, 'EM': EM}


def law(n, j, x, c):
    """The reduced law as (atoms, probabilities), floats; for checking against a direct DP."""
    if j == n - 1:
        a, t = x
        return [0.0, 1.0, float(c)], [1 - a, a*(1 - t), a*t]
    f, cons, extra, info = model(n, j, list(x), 0.0, c)
    Ws = info['W']
    if j == 0:
        G = list(x[1:])
        atoms = [0.0] + Ws[:n - 1] + [float(c)]
        T = info['OPT'] - Ws[n - 2]
        D = c - Ws[n - 2]
        cdf = G + [1 - T/D, 1.0]
    else:
        G0, H = x[1], x[2]; G = list(x[3:])
        atoms = [0.0, 1.0] + Ws[j:n - 1] + [float(c)]
        T = info['OPT'] - Ws[n - 2]
        D = c - Ws[n - 2]
        cdf = [G0, H] + G + [1 - T/D, 1.0]
    probs = [cdf[0]] + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))]
    return atoms, probs


def direct_ratio(atoms, probs, n):
    """OPT_n / E[M_n] by the dynamic program and the order statistics, floats."""
    W = 0.0
    for _ in range(n):
        W = sum(p*max(a, W) for a, p in zip(atoms, probs))
    order = sorted(range(len(atoms)), key=lambda i: atoms[i])
    EM = 0.0; Fc = 0.0
    for i in order:
        EM += atoms[i]*((Fc + probs[i])**n - Fc**n); Fc += probs[i]
    return W/EM
