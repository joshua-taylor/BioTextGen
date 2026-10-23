"""E4: attention replacement = nearest-neighbour lookup over the model's own SPARSE CODES.

Net A (sparse k-WTA, no memory) predicts the next char from the last k chars; its active-unit code c_t is the key.
For each position t, find the earlier position s<t in the window whose code overlaps c_t most (sparse dot product;
an inverted index makes this O(kw * entries per unit), not O(T*d)), and give net B a table row for
(overlap strength bin, char that followed s).  Net B also gets the exact-match rows of e3.
Retrieval is a hard, non-differentiable choice; nothing trains through it. B learns how to use what comes back.
Credit for both nets: exact per-unit finite differences (= backprop numbers on this fixed-selection net; see e2b 'fdo').
"""
import numpy as np, json
from common import load, xent, logsoftmax, Adam, LN2, Timer
from e2 import Net, init, scatter_rows
from e3 import window_feats, ORDERS

NB = 4  # overlap bins

def codes(netA, FA, H):
    outs, logits = netA.forward(FA)
    idx, val = outs[0]
    C = np.zeros((len(FA), H), np.float32); np.put_along_axis(C, idx, val, 1)
    return C, outs, logits

def nn_rows(C, seq_next, base, V, kw):
    """C: (T,H) codes for one window; seq_next[s] = char that followed position s. returns (T,) row ids."""
    T = C.shape[0]
    n = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-6)
    S = n @ n.T
    S[np.triu_indices(T, 0)] = -1.0                      # only s < t
    s = S.argmax(1); best = S[np.arange(T), s]
    b = np.clip((best * NB).astype(int), 0, NB - 1)
    r = base + b * (V + 1) + seq_next[s]
    r[best <= 0] = base + NB * (V + 1) - 1             # nothing similar: shared 'none' row (last row of block)
    r[0] = base + NB * (V + 1) - 1
    return r

def step_net(net, p, F, y, outs, logits, rng, B):
    """backprop-equivalent credit on a fixed-selection sparse net (identical to forward-only fdo)."""
    D = np.exp(logsoftmax(logits)); D[np.arange(B), y] -= 1; D /= B
    oi, ov = outs[-1]; kw = oi.shape[1]
    g = {"bo": D.sum(0), "Wo": scatter_rows(p["Wo"].shape, oi, ov[:, :, None] * D[:, None, :])}
    dz = np.einsum("bv,bkv->bk", D, p["Wo"][oi]) * (ov > 0)
    gT = np.zeros_like(p["T"]); nf = F.shape[1]
    np.add.at(gT, (np.repeat(F[:, :, None], kw, 2), np.repeat(oi[:, None, :], nf, 1)), np.repeat(dz[:, None, :], nf, 1))
    g["T"] = gT; gb = np.zeros_like(p["b0"]); np.add.at(gb, oi.reshape(-1), dz.reshape(-1)); g["b0"] = gb
    return g

def build(stream, starts, T, k, V, netA, H, kw, use_nn, use_exact):
    FA_l, FB_l, y_l, m_l = [], [], [], []
    for st in starts:
        seq = stream[st:st + T + 1]
        F, y, m = window_feats(seq, k, V, use_exact)
        FA = F[:, :k]
        if use_nn:
            C, _, _ = codes(netA, FA, H)
            base = k * V + len(ORDERS) * (V + 1)
            r = nn_rows(C, seq[1:], base, V, kw)
            F = np.concatenate([F, r[:, None]], 1) if use_exact else np.concatenate([FA, r[:, None]], 1)
        FA_l.append(FA); FB_l.append(F); y_l.append(y); m_l.append(m)
    return np.concatenate(FA_l), np.concatenate(FB_l), np.concatenate(y_l), np.concatenate(m_l)

def run(H=1024, kw=16, k=8, T=256, nseq=4, steps=5000, lr=3e-3, use_nn=1, use_exact=1, seed=0, ntrain=2_000_000,
        evals=10, log=True, **_):
    trn, val, itos = load(ntrain); V = len(itos)
    rng = np.random.default_rng(seed)
    pA = init(k * V, H, 1, V, rng); netA = Net(pA, H, kw, 1); optA = Adam(pA, lr)
    nrows = k * V + len(ORDERS) * (V + 1) + NB * (V + 1)
    pB = init(nrows, H, 1, V, rng); netB = Net(pB, H, kw, 1); optB = Adam(pB, lr)
    tm = Timer(); hist = []
    for it in range(1, steps + 1):
        FA, FB, y, _ = build(trn, rng.integers(0, len(trn) - T - 2, nseq), T, k, V, netA, H, kw, use_nn, use_exact)
        B = len(y)
        oA, lA = netA.forward(FA); optA.step(step_net(netA, pA, FA, y, oA, lA, rng, B))
        oB, lB = netB.forward(FB); optB.step(step_net(netB, pB, FB, y, oB, lB, rng, B))
        if it % max(1, steps // evals) == 0:
            tot = cnt = tmm = cm = 0.0
            for s0 in range(0, 100_000 - T - 1, T):
                FA, FB, yv, m = build(val, [s0], T, k, V, netA, H, kw, use_nn, use_exact)
                l = xent(netB.forward(FB)[1], yv); tot += l.sum(); cnt += len(yv); tmm += l[m].sum(); cm += m.sum()
            hist.append((it, it * B, round(tm(), 1), round(tot / cnt / LN2, 4), round(tmm / max(cm, 1) / LN2, 4)))
            if log: print("e4", f"nn{use_nn} ex{use_exact}", hist[-1], flush=True)
    return hist

if __name__ == "__main__":
    import sys
    a = dict(arg.split("=") for arg in sys.argv[1:])
    a = {kk: (float(v) if "." in v else int(v)) for kk, v in a.items()}
    print("RESULT", json.dumps(dict(a, hist=run(**a))))
