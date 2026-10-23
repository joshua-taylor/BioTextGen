"""E1: how does evolution-style (perturbation) learning cost scale with model shape?

Model: char MLP language model. Last k chars -> embeddings (k*de) -> L tanh layers of width H -> logits.
Rules
  bp : exact backprop (reference)
  np : per-token node perturbation. Each token gets its OWN random nudge on every hidden unit
       (antithetic +/-), scored by its own loss. Batch = population. Readout by delta rule.
  es : classic ES/EGGROLL. N population members, each a rank-1 nudge of the WEIGHTS shared by
       all tokens in the batch, scored by the batch's total loss. Readout by delta rule.
Metric: validation bits/char vs full forward passes per token (and wall-clock).
"""
import numpy as np, sys, json, argparse
from common import load, contexts, xent, logsoftmax, Adam, LN2, Timer

def init(V, k, de, H, L, rng):
    p = {"E": rng.normal(0, 0.3, (V, de)).astype(np.float32)}
    din = k * de
    for l in range(L):
        p[f"W{l}"] = (rng.normal(0, 1, (din, H)) / np.sqrt(din)).astype(np.float32)
        p[f"b{l}"] = np.zeros(H, np.float32); din = H
    p["Wo"] = np.zeros((din, V), np.float32); p["bo"] = np.zeros(V, np.float32)
    return p

def forward(p, X, L, noise=None, s=0.0):
    """noise: dict layer-> (B,width) added to pre-activations (layer -1 = input vector). returns acts list, logits"""
    B = X.shape[0]
    a = p["E"][X].reshape(B, -1)
    if noise is not None: a = a + s * noise[-1]
    acts = [a]
    for l in range(L):
        z = a @ p[f"W{l}"] + p[f"b{l}"]
        if noise is not None: z = z + s * noise[l]
        a = np.tanh(z); acts.append(a)
    return acts, a @ p["Wo"] + p["bo"]

def evaluate(p, val, k, L, n=50_000):
    idx = np.arange(k, min(len(val), k + n))
    tot = 0.0
    for i in range(0, len(idx), 10000):
        X, y = contexts(val, idx[i:i+10000], k)
        tot += xent(forward(p, X, L)[1], y).sum()
    return tot / len(idx) / LN2

def run(rule, H=128, L=1, k=8, de=16, B=256, steps=2000, lr=3e-3, sigma=0.05, K=1, N=32,
        seed=0, ntrain=2_000_000, evals=10, budget=None, log=True, r=16, frac=0.25, alpha=0.1, refresh=25):
    trn, val, itos = load(ntrain); V = len(itos)
    rng = np.random.default_rng(seed)
    p = init(V, k, de, H, L, rng); opt = Adam(p, lr)
    T = Timer(); fwd = 0.0; hist = []
    widths = {-1: k * de, **{l: H for l in range(L)}}
    if rule == "gnp":   # guided node perturbation: learned low-rank noise subspace per layer
        Cov = {l: np.eye(w, dtype=np.float32) for l, w in widths.items()}
        U = {l: np.linalg.qr(rng.standard_normal((w, min(r, w))))[0].astype(np.float32) for l, w in widths.items()}
        niso = int(B * frac)
    for step in range(1, steps + 1):
        idx = rng.integers(k, len(trn), B)
        X, y = contexts(trn, idx, k)
        acts, logits = forward(p, X, L)
        P = np.exp(logsoftmax(logits)); D = P.copy(); D[np.arange(B), y] -= 1; D /= B   # dl/dlogits
        g = {"Wo": acts[-1].T @ D, "bo": D.sum(0)}
        if rule == "bp":
            fwd += 3
            d = D @ p["Wo"].T
            for l in range(L - 1, -1, -1):
                dz = d * (1 - acts[l + 1] ** 2)
                g[f"W{l}"] = acts[l].T @ dz; g[f"b{l}"] = dz.sum(0)
                d = dz @ p[f"W{l}"].T
            gE = np.zeros_like(p["E"]); np.add.at(gE, X.reshape(-1), d.reshape(-1, de)); g["E"] = gE
        elif rule == "np":
            dz = {l: np.zeros((B, w), np.float32) for l, w in widths.items()}
            for r in range(K):
                xi = {l: rng.standard_normal((B, w)).astype(np.float32) for l, w in widths.items()}
                lp = xent(forward(p, X, L, xi, +sigma)[1], y)
                lm = xent(forward(p, X, L, xi, -sigma)[1], y)
                c = ((lp - lm) / (2 * sigma))[:, None] / (B * K)
                for l in widths: dz[l] += c * xi[l]
            fwd += 1 + 2 * K
            for l in range(L):
                g[f"W{l}"] = acts[l].T @ dz[l]; g[f"b{l}"] = dz[l].sum(0)
            gE = np.zeros_like(p["E"]); np.add.at(gE, X.reshape(-1), dz[-1].reshape(-1, de)); g["E"] = gE
        elif rule == "gnp":
            dz = {}; xi = {}
            for l, w in widths.items():
                rr = U[l].shape[1]
                iso = rng.standard_normal((niso, w)).astype(np.float32)
                gui = np.sqrt(1 - alpha) * np.sqrt(w / rr) * (rng.standard_normal((B - niso, rr)).astype(np.float32) @ U[l].T) \
                      + np.sqrt(alpha) * rng.standard_normal((B - niso, w)).astype(np.float32)
                xi[l] = np.concatenate([iso, gui], 0)
            lp = xent(forward(p, X, L, xi, +sigma)[1], y)
            lm = xent(forward(p, X, L, xi, -sigma)[1], y)
            c = ((lp - lm) / (2 * sigma))[:, None]
            for l, w in widths.items():
                gh = c * xi[l]
                dz[l] = gh / B
                gi = gh[:niso]
                Cov[l] = 0.95 * Cov[l] + 0.05 * (gi.T @ gi) / niso
                if step % refresh == 0:
                    ev, evec = np.linalg.eigh(Cov[l]); U[l] = evec[:, -U[l].shape[1]:].astype(np.float32)
            fwd += 3
            for l in range(L):
                g[f"W{l}"] = acts[l].T @ dz[l]; g[f"b{l}"] = dz[l].sum(0)
            gE = np.zeros_like(p["E"]); np.add.at(gE, X.reshape(-1), dz[-1].reshape(-1, de)); g["E"] = gE
        elif rule == "ro":
            fwd += 1
        elif rule == "es":
            keys = ["E"] + [f"W{l}" for l in range(L)] + [f"b{l}" for l in range(L)]
            acc = {kk: np.zeros_like(p[kk]) for kk in keys}
            base = {kk: p[kk].copy() for kk in keys}
            for n in range(N):
                eps = {}
                for kk in keys:
                    sh = p[kk].shape
                    if len(sh) == 2:   # rank-1 EGGROLL-style
                        eps[kk] = np.outer(rng.standard_normal(sh[0]), rng.standard_normal(sh[1])).astype(np.float32)
                    else:
                        eps[kk] = rng.standard_normal(sh).astype(np.float32)
                fs = []
                for sgn in (+1, -1):
                    for kk in keys: p[kk] = base[kk] + sgn * sigma * eps[kk]
                    fs.append(xent(forward(p, X, L)[1], y).mean())
                c = (fs[0] - fs[1]) / (2 * sigma * N)
                for kk in keys: acc[kk] += c * eps[kk]
            for kk in keys: p[kk] = base[kk]; g[kk] = acc[kk]
            fwd += 1 + 2 * N
        opt.step(g)
        if step % max(1, steps // evals) == 0 or step == steps:
            v = evaluate(p, val, k, L)
            hist.append((step, fwd, round(T(), 1), round(v, 4)))
            if log: print(rule, f"H{H} L{L}", hist[-1], flush=True)
        if budget and T() > budget: break
    return hist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a, t, d in [("rule", str, "bp"), ("H", int, 128), ("L", int, 1), ("B", int, 256), ("steps", int, 2000),
                    ("lr", float, 3e-3), ("sigma", float, 0.05), ("K", int, 1), ("N", int, 32), ("seed", int, 0),
                    ("budget", float, 0)]:
        ap.add_argument("--" + a, type=t, default=d)
    a = ap.parse_args()
    h = run(a.rule, H=a.H, L=a.L, B=a.B, steps=a.steps, lr=a.lr, sigma=a.sigma, K=a.K, N=a.N, seed=a.seed,
            budget=a.budget or None)
    print("RESULT", json.dumps(dict(vars(a), hist=h)))
