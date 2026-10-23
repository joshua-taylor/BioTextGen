"""E3: a hard-retrieval memory as the attention replacement, inside the sparse (k-WTA) network.

Training/eval on contiguous windows of T chars. For every position t and every order n in ORDERS, look up
the most recent earlier position in the same window whose previous n chars match exactly (an exact-match
'induction' lookup, O(1) with a hash map), and add a table row for (n, the char that followed there).
The memory itself has no parameters; the network learns how much to trust each retrieval.
Rules as in e2: bp (backprop through active units) or np (per-token node perturbation, batch = population).
"""
import numpy as np, json, argparse
from common import load, xent, logsoftmax, Adam, LN2, Timer
from e2 import Net, init, scatter_rows

ORDERS = (3, 5, 8, 12)

def window_feats(seq, k, V, use_mem, orders=ORDERS):
    """seq: (T+1,) chars; returns F (T, nfeat) row ids, y (T,), matched (T,) bool for order>=8."""
    T = len(seq) - 1
    cols = [np.concatenate([np.zeros(max(0, i + 1 - 0), np.int64)[:0], seq[:T]]) for i in range(0)]
    pad = np.concatenate([np.zeros(k, np.int64), seq])          # k leading pads (char 0)
    rows = [pad[k - i - 1: k - i - 1 + T] + i * V for i in range(k)]   # char at t-1-i for target index t (t=1..T)
    y = seq[1:]
    matched = np.zeros(T, bool)
    if use_mem:
        base = k * V
        for j, n in enumerate(orders):
            last = {}; r = np.full(T, base + j * (V + 1) + V, np.int64)   # 'no match' row
            s = pad.tobytes() if False else None
            for t in range(T):      # predicting seq[t+1] from seq[..t]
                if t + 1 >= n:
                    key = seq[t + 1 - n: t + 1].tobytes()
                    prev = last.get(key)
                    if prev is not None:
                        r[t] = base + j * (V + 1) + seq[prev + 1]
                        if n >= 8: matched[t] = True
                    last[key] = t
            rows.append(r)
    return np.stack(rows, 1), y, matched

def batch(stream, starts, T, k, V, use_mem):
    Fs, ys, ms = [], [], []
    for s in starts:
        F, y, m = window_feats(stream[s:s + T + 1], k, V, use_mem); Fs.append(F); ys.append(y); ms.append(m)
    return np.concatenate(Fs), np.concatenate(ys), np.concatenate(ms)

def evaluate(net, val, T, k, V, use_mem, n=100_000):
    tot = 0.0; cnt = 0; tm = 0.0; cm = 0
    for s in range(0, min(n, len(val) - T - 1), T):
        F, y, m = window_feats(val[s:s + T + 1], k, V, use_mem)
        l = xent(net.forward(F)[1], y); tot += l.sum(); cnt += len(y); tm += l[m].sum(); cm += m.sum()
    return tot / cnt / LN2, tm / max(cm, 1) / LN2

def run(rule, H=1024, kw=16, L=1, k=8, T=256, nseq=4, steps=5000, lr=3e-3, sigma=0.05, mem=1, seed=0,
        ntrain=2_000_000, evals=10, log=True, **_):
    trn, val, itos = load(ntrain); V = len(itos)
    rng = np.random.default_rng(seed)
    nrows = k * V + len(ORDERS) * (V + 1)
    p = init(nrows, H, L, V, rng); opt = Adam(p, lr); net = Net(p, H, kw, L)
    tm = Timer(); hist = []
    for step in range(1, steps + 1):
        F, y, _ = batch(trn, rng.integers(0, len(trn) - T - 2, nseq), T, k, V, mem)
        B = len(y); ar = np.arange(B)
        outs, logits = net.forward(F)
        D = np.exp(logsoftmax(logits)); D[ar, y] -= 1; D /= B
        oi, ov = outs[-1]
        g = {"bo": D.sum(0), "Wo": (ov.T @ D) if oi is None else scatter_rows(p["Wo"].shape, oi, ov[:, :, None] * D[:, None, :])}
        if rule == "bp":
            dz = [None] * L; ii = outs[-1][0]
            d = D @ p["Wo"].T if ii is None else np.einsum("bv,bkv->bk", D, p["Wo"][ii])
            for l in range(L - 1, -1, -1):
                dz[l] = d * (outs[l][1] > 0)
                if l > 0:
                    pi = outs[l - 1][0]; ci = outs[l][0]; W = p[f"W{l}"]
                    d = dz[l] @ W.T if ci is None else np.einsum("bj,bkj->bk", dz[l], W[pi[:, :, None], ci[:, None, :]])
        else:
            xi = [rng.standard_normal(o[1].shape).astype(np.float32) for o in outs]
            lp = xent(net.forward_noisy(F, outs, xi, +sigma), y); lm = xent(net.forward_noisy(F, outs, xi, -sigma), y)
            c = ((lp - lm) / (2 * sigma))[:, None] / B
            dz = [c * xi[l] * (outs[l][1] > 0) for l in range(L)]
        i0 = outs[0][0]
        if i0 is None: g["T"] = scatter_rows(p["T"].shape, F, np.repeat(dz[0][:, None, :], F.shape[1], 1))
        else:
            gT = np.zeros_like(p["T"]); nf = F.shape[1]
            np.add.at(gT, (np.repeat(F[:, :, None], kw, 2), np.repeat(i0[:, None, :], nf, 1)), np.repeat(dz[0][:, None, :], nf, 1))
            g["T"] = gT
        gb = np.zeros(H, np.float32)
        if i0 is None: gb = dz[0].sum(0)
        else: np.add.at(gb, i0.reshape(-1), dz[0].reshape(-1))
        g["b0"] = gb
        for l in range(1, L):
            pi, pv = outs[l - 1]; ci = outs[l][0]; gW = np.zeros_like(p[f"W{l}"])
            np.add.at(gW, (np.repeat(pi[:, :, None], kw, 2), np.repeat(ci[:, None, :], kw, 1)), pv[:, :, None] * dz[l][:, None, :])
            g[f"W{l}"] = gW; gbl = np.zeros(H, np.float32); np.add.at(gbl, ci.reshape(-1), dz[l].reshape(-1)); g[f"b{l}"] = gbl
        opt.step(g)
        if step % max(1, steps // evals) == 0:
            v, vm = evaluate(net, val, T, k, V, mem)
            hist.append((step, step * nseq * T, round(tm(), 1), round(v, 4), round(vm, 4)))
            if log: print(rule, f"H{H} kw{kw} mem{mem}", hist[-1], flush=True)
    return hist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a, t, d in [("rule", str, "bp"), ("H", int, 1024), ("kw", int, 16), ("L", int, 1), ("mem", int, 1),
                    ("steps", int, 5000), ("lr", float, 3e-3), ("seed", int, 0)]:
        ap.add_argument("--" + a, type=t, default=d)
    a = ap.parse_args(); print("RESULT", json.dumps(dict(vars(a), hist=run(**vars(a)))))
