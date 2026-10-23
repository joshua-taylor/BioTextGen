"""E2 (fast): sparse vs dense, exact forward-only credit vs random per-token perturbation (K draws)."""
import numpy as np, json, time
from common import load, LN2
from fast import Sparse1
from e5 import build
def run(H=1024, kw=16, K=0, steps=5000, T=256, nseq=4, lr=3e-3, seed=0, evals=5, log=False, **_):
    trn, val, itos = load(2_000_000); V = len(itos); space = itos.index(' ')
    g = {"feats": [("c", i) for i in range(1, 9)], "H": H, "kw": kw}
    rng = np.random.default_rng(seed); net = Sparse1(8 * V, H, kw, V, lr=lr, seed=seed); hist = []; t0 = time.time()
    for it in range(1, steps + 1):
        F, y, _ = build(g, trn, rng.integers(0, len(trn) - T - 2, nseq), T, V, space)
        net.step(F, y, K=K, rng=rng)
        if it % (steps // evals) == 0:
            tot = cnt = 0.0
            for s0 in range(0, 60_000 - T - 1, T):
                F, yv, _ = build(g, val, [s0], T, V, space); tot += net.loss(F, yv).sum(); cnt += len(yv)
            hist.append((it, it * nseq * T, round(time.time() - t0, 1), round(tot / cnt / LN2, 4)))
    return hist
