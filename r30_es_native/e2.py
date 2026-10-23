"""E2: does SPARSE activation make perturbation learning cheap?

Model: last k chars (one row per (position, char), plus optional hashed n-gram rows) -> sum of table rows
-> L hidden layers of width H, each keeping only its top-kw units (k-winners-take-all; kw=H means dense ReLU)
-> readout.  Rules:
  bp : backprop through the active units (winner selection treated as fixed, as is standard)
  np : per-token node perturbation on ACTIVE units only (antithetic, batch = population); readout by delta rule
  ro : readout only (random fixed hidden), control
Cone law prediction: np's cost relative to bp ~ active units per token = kw * L, not H.
Sparse layers are computed sparsely (cost ~ kw, not H).
"""
import numpy as np, json, argparse
from common import load, contexts, xent, logsoftmax, Adam, LN2, Timer

def feats(X, V, ngram, R):
    B, k = X.shape
    rows = [X[:, -i - 1] + i * V for i in range(k)]
    if ngram >= 2:
        h = np.zeros(B, np.int64)
        for o in range(1, ngram + 1):
            h = (h * 1000003 + X[:, -o] + 1) % (1 << 61)
            if o >= 2: rows.append(k * V + (h * 2654435761 + o) % R)
    return np.stack(rows, 1)

def init(nrows, H, L, V, rng, tscale=0.1):
    p = {"T": (rng.normal(0, 1, (nrows, H)) * tscale).astype(np.float32), "b0": np.zeros(H, np.float32)}
    for l in range(1, L):
        p[f"W{l}"] = (rng.normal(0, 1, (H, H)) * np.sqrt(2.0 / H)).astype(np.float32); p[f"b{l}"] = np.zeros(H, np.float32)
    p["Wo"] = np.zeros((H, V), np.float32); p["bo"] = np.zeros(V, np.float32)
    return p

def topk(z, kw):
    idx = np.argpartition(z, -kw, axis=1)[:, -kw:]
    return idx, np.take_along_axis(z, idx, 1)

class Net:
    """layer outputs as (idx (B,kw), val (B,kw)); dense case kw==H uses idx=None."""
    def __init__(s, p, H, kw, L): s.p, s.H, s.kw, s.L = p, H, kw, L
    def pre(s, l, prev):  # full pre-activation of layer l given previous sparse output
        p = s.p
        if l == 0: return p["T"][prev].sum(1) + p["b0"]
        idx, val = prev
        if idx is None: return val @ p[f"W{l}"] + p[f"b{l}"]
        return np.einsum("bk,bkh->bh", val, p[f"W{l}"][idx]) + p[f"b{l}"]
    def pre_at(s, l, prev, sel):  # pre-activation of layer l only at units sel (B,kw)
        p = s.p
        if l == 0:
            return np.take_along_axis(p["T"][prev].sum(1), sel, 1) + p["b0"][sel]
        idx, val = prev
        W = p[f"W{l}"]
        return np.einsum("bk,bkj->bj", val, W[idx[:, :, None], sel[:, None, :]]) + p[f"b{l}"][sel]
    def readout(s, out):
        idx, val = out; p = s.p
        if idx is None: return val @ p["Wo"] + p["bo"]
        return np.einsum("bk,bkv->bv", val, p["Wo"][idx]) + p["bo"]
    def forward(s, F):
        outs, prev = [], F
        for l in range(s.L):
            z = s.pre(l, prev)
            if s.kw >= s.H: out = (None, np.maximum(z, 0))
            else:
                idx, v = topk(z, s.kw); out = (idx, np.maximum(v, 0))
            outs.append(out); prev = out
        return outs, s.readout(outs[-1])
    def forward_noisy(s, F, outs, xi, sg):
        """same winners as `outs`; noise added to active pre-activations."""
        prev, new = F, []
        for l in range(s.L):
            idx = outs[l][0]
            if idx is None:
                z = s.pre(l, prev); act = (z > 0) if l == 0 or True else None
                z = z + sg * xi[l] * (outs[l][1] > 0)
                out = (None, np.maximum(z, 0) * (outs[l][1] > 0))
            else:
                z = s.pre_at(l, prev, idx) + sg * xi[l] * (outs[l][1] > 0)
                out = (idx, np.maximum(z, 0) * (outs[l][1] > 0))
            new.append(out); prev = out
        return s.readout(new[-1])

def evaluate(net, val, k, V, ngram, R, n=50_000):
    idx = np.arange(k, min(len(val), k + n)); tot = 0.0
    for i in range(0, len(idx), 5000):
        X, y = contexts(val, idx[i:i + 5000], k)
        tot += xent(net.forward(feats(X, V, ngram, R))[1], y).sum()
    return tot / len(idx) / LN2

def scatter_rows(shape, idx, upd):
    """sum upd (B,kw,C) into rows idx (B,kw) of a (R,C) zero matrix."""
    g = np.zeros(shape, np.float32); np.add.at(g, idx.reshape(-1), upd.reshape(-1, shape[1])); return g

def run(rule, H=128, kw=128, L=1, k=8, ngram=0, R=1 << 16, B=256, steps=2000, lr=3e-3, sigma=0.05,
        seed=0, ntrain=2_000_000, evals=10, log=True, tscale=0.1, **_):
    trn, val, itos = load(ntrain); V = len(itos)
    rng = np.random.default_rng(seed)
    nrows = k * V + (R if ngram >= 2 else 0)
    p = init(nrows, H, L, V, rng, tscale); opt = Adam(p, lr); net = Net(p, H, kw, L)
    T = Timer(); fwd = 0.0; hist = []
    ar = np.arange(B)
    for step in range(1, steps + 1):
        idx = rng.integers(k, len(trn), B)
        X, y = contexts(trn, idx, k); F = feats(X, V, ngram, R)
        outs, logits = net.forward(F)
        D = np.exp(logsoftmax(logits)); D[ar, y] -= 1; D /= B
        oi, ov = outs[-1]
        g = {"bo": D.sum(0), "Wo": (ov.T @ D) if oi is None else scatter_rows(p["Wo"].shape, oi, ov[:, :, None] * D[:, None, :])}
        dz = None
        if rule == "bp":
            fwd += 3; dz = [None] * L
            ii, vv = outs[-1]
            d = D @ p["Wo"].T if ii is None else np.einsum("bv,bkv->bk", D, p["Wo"][ii])
            for l in range(L - 1, -1, -1):
                dz[l] = d * (outs[l][1] > 0)
                if l > 0:
                    pi, pv = outs[l - 1]; ci = outs[l][0]; W = p[f"W{l}"]
                    if ci is None: d = dz[l] @ W.T
                    else: d = np.einsum("bj,bkj->bk", dz[l], W[pi[:, :, None], ci[:, None, :]])
        elif rule == "np":
            w = [o[1].shape[1] for o in outs]
            xi = [rng.standard_normal((B, w[l])).astype(np.float32) for l in range(L)]
            lp = xent(net.forward_noisy(F, outs, xi, +sigma), y)
            lm = xent(net.forward_noisy(F, outs, xi, -sigma), y)
            c = ((lp - lm) / (2 * sigma))[:, None] / B
            dz = [c * xi[l] * (outs[l][1] > 0) for l in range(L)]
            fwd += 3
        else:
            fwd += 1
        if dz is not None:
            i0 = outs[0][0]
            if i0 is None:
                g["T"] = scatter_rows(p["T"].shape, F, np.repeat(dz[0][:, None, :], F.shape[1], 1))
            else:   # only the active columns of the touched rows
                gT = np.zeros_like(p["T"])
                np.add.at(gT, (np.repeat(F[:, :, None], kw, 2), np.repeat(i0[:, None, :], F.shape[1], 1)),
                          np.repeat(dz[0][:, None, :], F.shape[1], 1))
                g["T"] = gT
            g["b0"] = np.zeros(H, np.float32)
            if i0 is None: g["b0"] = dz[0].sum(0)
            else: np.add.at(g["b0"], i0.reshape(-1), dz[0].reshape(-1))
            for l in range(1, L):
                pi, pv = outs[l - 1]; ci = outs[l][0]
                if ci is None and pi is None:
                    g[f"W{l}"] = pv.T @ dz[l]; g[f"b{l}"] = dz[l].sum(0)
                else:
                    gW = np.zeros_like(p[f"W{l}"])
                    np.add.at(gW, (np.repeat(pi[:, :, None], kw, 2), np.repeat(ci[:, None, :], kw, 1)),
                              pv[:, :, None] * dz[l][:, None, :])
                    g[f"W{l}"] = gW
                    gb = np.zeros(H, np.float32); np.add.at(gb, ci.reshape(-1), dz[l].reshape(-1)); g[f"b{l}"] = gb
        opt.step(g)
        if step % max(1, steps // evals) == 0 or step == steps:
            v = evaluate(net, val, k, V, ngram, R)
            hist.append((step, fwd, round(T(), 1), round(v, 4)))
            if log: print(rule, f"H{H} kw{kw} L{L}", hist[-1], flush=True)
    return hist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a, t, d in [("rule", str, "bp"), ("H", int, 128), ("kw", int, 128), ("L", int, 1), ("ngram", int, 0),
                    ("B", int, 256), ("steps", int, 2000), ("lr", float, 3e-3), ("sigma", float, 0.05), ("seed", int, 0)]:
        ap.add_argument("--" + a, type=t, default=d)
    a = ap.parse_args()
    print("RESULT", json.dumps(dict(vars(a), hist=run(**vars(a)))))
