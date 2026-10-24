"""Fast sparse bench: sparse k-WTA network + hard own-code lookup, trained with exact forward-only credit.

Model (per character position p, predicting s[p+1]):
  x  = sum of 64-dim rows for the input features                     (the 'summary')
  z1 = x W1 + b1;  layer-1 code = top-kw units of z1 (ReLU)           (the 'fingerprint')
  lookup: among remembered positions, the one whose layer-1 code has highest cosine with this one;
          returns r = (similarity bin, char that followed it)         (no parameters, no gradient)
  L=2: z2 = (x + M[r]) W2 + a1 R2 + b2; layer-2 code = top-kw of z2; active set = layer1 + layer2 units
  logits = sum over active units a_j Wo[j] + bo  (+ Mo[r] when L=1)

Credit: the per-unit credit for active units is computed in closed form; on this network with the winner sets
and the lookup choice held fixed it is identical to the forward-only one-unit-at-a-time measurement (verified in
round 30, cosine 1.000000). Updates are local: (credit at a unit) x (that unit's input).

Memory for the lookup: per training stream, a ring of the last `horizon` positions; per unit a posting list of the
last `cap` positions where it fired. Codes are stored as computed at the time (they are not refreshed).

Optional mechanisms beyond backprop (cfg keys):
  swap     : m>0 -> for each token try each of the next-m runner-up units in place of the weakest winner; if the
             token's loss would fall, push the runner-up up and the winner down (credit for the selection itself)
  ste      : straight-through control: give runner-ups the credit they would get if they were active
  regrow   : every G steps, units that fired < thr times are rewritten in one shot to fire for a recent
             high-loss token and vote for its answer ('neurogenesis')
"""
import numpy as np, time, json, sys, os
import scipy.sparse as sp
from numba import njit
import sbdata
from common import logsoftmax, LN2

NB = 4

# ------------------------------------------------------------------ lookup memory (numba)
@njit(cache=True)
def _lookup2(act, val, nxt, gid0, pos_store, pval_store, cnt, head, ent_next, ent_norm, ent_gid, score, touched,
             horizon, V, topm, out_r, out_cos):
    T, k = act.shape; H, cap = pos_store.shape; M = ent_next.shape[0]
    best_s = np.empty(topm, np.int64); best_c = np.empty(topm, np.float64)
    for t in range(T):
        g = gid0 + t
        nrm = 0.0
        for u in range(k):
            if val[t, u] > 0: nrm += val[t, u] * val[t, u]
        nrm = np.sqrt(nrm) + 1e-6
        nt = 0
        for u in range(k):
            v = val[t, u]
            if v <= 0: continue
            j = act[t, u]
            for q in range(cnt[j]):
                e = pos_store[j, q]
                slot = e % M
                if ent_gid[slot] != e or g - e > horizon: continue
                if score[slot] == 0.0:
                    touched[nt] = slot; nt += 1
                score[slot] += v * pval_store[j, q]
        for m in range(topm):
            best_s[m] = -1; best_c[m] = 0.0
        for i in range(nt):
            slot = touched[i]
            cs = score[slot] / (nrm * ent_norm[slot])
            score[slot] = 0.0
            # insert into top-m
            if cs > best_c[topm - 1]:
                m = topm - 1
                while m > 0 and cs > best_c[m - 1]:
                    best_c[m] = best_c[m - 1]; best_s[m] = best_s[m - 1]; m -= 1
                best_c[m] = cs; best_s[m] = slot
        for m in range(topm):
            if best_s[m] < 0 or best_c[m] <= 0:
                out_r[t, m] = NB * (V + 1); out_cos[t, m] = 0.0
            else:
                b = int(best_c[m] * NB)
                if b > NB - 1: b = NB - 1
                out_r[t, m] = b * (V + 1) + ent_next[best_s[m]]; out_cos[t, m] = best_c[m]
        # write this position into memory
        slot = g % M
        ent_gid[slot] = g; ent_next[slot] = nxt[t]; ent_norm[slot] = nrm
        for u in range(k):
            v = val[t, u]
            if v <= 0: continue
            j = act[t, u]
            if cnt[j] < cap:
                pos_store[j, cnt[j]] = g; pval_store[j, cnt[j]] = v; cnt[j] += 1
            else:
                h = head[j]; pos_store[j, h] = g; pval_store[j, h] = v; head[j] = (h + 1) % cap

class Memory:
    def __init__(self, H, horizon, cap):
        M = 1
        while M < horizon + 2: M *= 2
        self.pos = np.zeros((H, cap), np.int64); self.pval = np.zeros((H, cap), np.float64)
        self.cnt = np.zeros(H, np.int64); self.head = np.zeros(H, np.int64)
        self.next = np.zeros(M, np.int64); self.norm = np.ones(M); self.gid = np.full(M, -1, np.int64)
        self.score = np.zeros(M); self.touched = np.zeros(M, np.int64); self.g = 0; self.horizon = horizon
    def query_write(self, act, val, nxt, V, topm):
        T = act.shape[0]; r = np.empty((T, topm), np.int64); c = np.empty((T, topm))
        _lookup2(act.astype(np.int64), val.astype(np.float64), nxt.astype(np.int64), self.g, self.pos, self.pval,
                 self.cnt, self.head, self.next, self.norm, self.gid, self.score, self.touched, self.horizon, V, topm, r, c)
        self.g += T
        return r, c

# ------------------------------------------------------------------ optimiser
class Adam:
    def __init__(s, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        s.m = np.zeros(shape, np.float32); s.v = np.zeros(shape, np.float32); s.lr, s.b1, s.b2, s.eps, s.t = lr, b1, b2, eps, 0
    def dense(s, W, g):
        s.t += 1; s.m *= s.b1; s.m += (1 - s.b1) * g; s.v *= s.b2; s.v += (1 - s.b2) * g * g
        W -= s.lr * (s.m / (1 - s.b1 ** s.t)) / (np.sqrt(s.v / (1 - s.b2 ** s.t)) + s.eps)
    def rows(s, W, rows, g):
        s.t += 1; m = s.m[rows] * s.b1 + (1 - s.b1) * g; v = s.v[rows] * s.b2 + (1 - s.b2) * g * g
        s.m[rows] = m; s.v[rows] = v
        W[rows] -= s.lr * (m / (1 - s.b1 ** s.t)) / (np.sqrt(v / (1 - s.b2 ** s.t)) + s.eps)

def topk(z, k):
    idx = np.argpartition(z, -k, axis=1)[:, -k:]
    return idx, np.take_along_axis(z, idx, 1)

def scatter_rows(rows, g, nrows_unused=None):
    u, inv = np.unique(rows, return_inverse=True)
    out = np.zeros((len(u), g.shape[-1]), np.float32); np.add.at(out, inv.reshape(-1), g.reshape(-1, g.shape[-1]))
    return u, out

def csr(idx, val, H):
    B, k = idx.shape
    return sp.csr_matrix((val.reshape(-1), idx.reshape(-1), np.arange(0, B * k + 1, k)), shape=(B, H))

DEFAULT = dict(feats=[["c", i] for i in range(1, 9)], H=1024, kw=32, L=2, de=64, lr=3e-3, lookup=1, horizon=1024,
               cap=128, topm=1, swap=0, swap_lr=1.0, ste=0, regrow=0, regrow_every=200, regrow_thr=1, seed=0,
               T=512, S=2, steps=800, evals=4, nval=40_000, wbits=0, wbits_from=0)

class Model:
    def __init__(s, cfg, V):
        s.c = cfg; s.V = V; rng = np.random.default_rng(cfg["seed"]); s.rng = rng
        H, de = cfg["H"], cfg["de"]
        s.feats = [tuple(f) for f in cfg["feats"]]
        s.nrows = sum(sbdata.block_size(f, V) for f in s.feats)
        f32 = np.float32
        s.p = {"E": (rng.normal(0, 0.05, (s.nrows, de))).astype(f32),
               "W1": (rng.normal(0, 1, (de, H)) / np.sqrt(de)).astype(f32), "b1": np.zeros(H, f32),
               "Wo": np.zeros((H, V), f32), "bo": np.zeros(V, f32)}
        nr = (NB + 1) * (V + 1)
        if cfg["L"] == 2:
            s.p.update({"W2": (rng.normal(0, 1, (de, H)) / np.sqrt(de)).astype(f32), "b2": np.zeros(H, f32),
                        "R2": (rng.normal(0, 1, (H, H)) * np.sqrt(2.0 / H) * 0.5).astype(f32),
                        "M": (rng.normal(0, 0.05, (nr, de))).astype(f32)})
        else:
            s.p["Mo"] = np.zeros((nr, V), f32)
        s.opt = {k: Adam(v.shape, cfg["lr"]) for k, v in s.p.items()}
        s.usage = np.zeros(H, np.int64); s.hard = []   # recent high-loss tokens for regrowth
        s.stats = {"swap_tried": 0, "swap_good": 0, "regrown": 0}

    # ---------------------------------------------------------------- forward
    def forward(s, F, mems, nxt, nseg):
        """F (B,nf); tokens ordered as nseg consecutive segments, one per memory in mems."""
        p, c = s.p, s.c; H, kw = c["H"], c["kw"]; B = F.shape[0]
        x = p["E"][F].sum(1)
        z1 = x @ p["W1"] + p["b1"]
        i1, v1 = topk(z1, kw); a1 = np.maximum(v1, 0)
        out = {"x": x, "z1": z1, "i1": i1, "a1": a1}
        if c["lookup"]:
            T = B // nseg; rs, cs = [], []
            for g in range(nseg):
                r, co = mems[g].query_write(i1[g * T:(g + 1) * T], a1[g * T:(g + 1) * T], nxt[g * T:(g + 1) * T], s.V, c["topm"])
                rs.append(r); cs.append(co)
            r = np.concatenate(rs); out["r"] = r
        else:
            r = np.full((B, 1), NB * (s.V + 1)); out["r"] = r
        A1 = csr(i1, a1, H); out["A1"] = A1
        if c["L"] == 2:
            x2 = x + p["M"][r].sum(1)
            z2 = x2 @ p["W2"] + p["b2"] + A1 @ p["R2"]
            i2, v2 = topk(z2, kw); a2 = np.maximum(v2, 0)
            out.update(x2=x2, z2=z2, i2=i2, a2=a2)
            A = A1 + csr(i2, a2, H)
            logits = A @ p["Wo"] + p["bo"]
        else:
            A = A1
            logits = A @ p["Wo"] + p["bo"] + p["Mo"][r].sum(1)
        out["A"] = A; out["logits"] = np.asarray(logits)
        return out

    def loss(s, out, y):
        ls = logsoftmax(out["logits"]); return -ls[np.arange(len(y)), y]

    # ---------------------------------------------------------------- learning step
    def step(s, F, y, mems, nxt, nseg):
        p, c = s.p, s.c; H, kw = c["H"], c["kw"]; B = len(y)
        o = s.forward(F, mems, nxt, nseg)
        lp = logsoftmax(o["logits"]); tokl = -lp[np.arange(B), y]
        P = np.exp(lp); D = P.copy(); D[np.arange(B), y] -= 1; D /= B
        g = {}
        g["Wo"] = np.asarray(o["A"].T @ D); g["bo"] = D.sum(0)
        dWo = D @ p["Wo"].T                                    # (B,H): credit each unit would get from the readout
        i1, a1 = o["i1"], o["a1"]
        m1 = (a1 > 0)
        dz1 = np.take_along_axis(dWo, i1, 1) * m1
        if c["L"] == 2:
            i2, a2 = o["i2"], o["a2"]; m2 = a2 > 0
            dz2 = np.take_along_axis(dWo, i2, 1) * m2
            DZ2 = csr(i2, dz2, H)
            dz1 = dz1 + np.take_along_axis(np.asarray(DZ2 @ p["R2"].T), i1, 1) * m1
            g["W2"] = np.asarray((DZ2.T @ o["x2"]).T); g["b2"] = np.asarray(DZ2.sum(0)).ravel()
            g["R2"] = (o["A1"].T @ DZ2).toarray()
            dx2 = np.asarray(DZ2 @ p["W2"].T)
            u, gm = scatter_rows(o["r"], np.repeat(dx2[:, None, :], o["r"].shape[1], 1)); g["M"] = (u, gm)
        else:
            dx2 = 0.0
            u, gm = scatter_rows(o["r"], np.repeat(D[:, None, :], o["r"].shape[1], 1)); g["Mo"] = (u, gm)
        # --- beyond-backprop: credit for the selection itself (layer 1)
        extra_i, extra_d = None, None
        if c["swap"] or c["ste"]:
            m = max(c["swap"], c["ste"])
            z1 = o["z1"].copy(); np.put_along_axis(z1, i1, -np.inf, 1)
            ri, rv = topk(z1, m)                               # runner-ups
            if c["ste"]:
                extra_i = ri; extra_d = np.take_along_axis(dWo, ri, 1) * (rv > 0)
            else:
                # loss if runner-up j replaced the weakest active winner (readout path only)
                w = np.argmin(np.where(m1, a1, np.inf), 1); wi = i1[np.arange(B), w]; wa = a1[np.arange(B), w]
                base = o["logits"]; Wo = p["Wo"]
                ed = np.zeros((B, m), np.float32); dw = np.zeros(B, np.float32)
                for j in range(m):
                    aj = np.maximum(rv[:, j], 0) + 1e-3
                    lg = base + aj[:, None] * Wo[ri[:, j]] - wa[:, None] * Wo[wi]
                    lj = -logsoftmax(lg)[np.arange(B), y]
                    gain = tokl - lj                               # >0: swap would help
                    good = gain > 0
                    s.stats["swap_tried"] += B; s.stats["swap_good"] += int(good.sum())
                    ed[:, j] = -c["swap_lr"] * np.where(good, gain, 0) / B    # credit sign: negative = raise unit
                    dw += c["swap_lr"] * np.where(good, gain, 0) / B
                extra_i = np.concatenate([ri, wi[:, None]], 1); extra_d = np.concatenate([ed, dw[:, None]], 1)
        DZ1 = csr(i1, dz1, H)
        if extra_i is not None: DZ1 = DZ1 + csr(extra_i, extra_d.astype(np.float32), H)
        g["W1"] = np.asarray((DZ1.T @ o["x"]).T); g["b1"] = np.asarray(DZ1.sum(0)).ravel()
        dx = np.asarray(DZ1 @ p["W1"].T) + dx2
        u, ge = scatter_rows(F, np.repeat(dx[:, None, :], F.shape[1], 1)); g["E"] = (u, ge)
        for k, v in g.items():
            if isinstance(v, tuple): s.opt[k].rows(p[k], v[0], v[1])
            else: s.opt[k].dense(p[k], v.astype(np.float32))
        # --- usage and hard examples
        np.add.at(s.usage, i1[m1], 1)
        if c["regrow"]:
            hi = np.argsort(-tokl)[:8]
            for h in hi: s.hard.append((o["x"][h].copy(), int(y[h])))
            s.hard = s.hard[-512:]
        return tokl

    def regrow(s):
        """neurogenesis: rewrite rarely-used layer-1 units, in one shot, to fire for a recent hard token."""
        c, p = s.c, s.p
        dead = np.where(s.usage < c["regrow_thr"])[0]
        if len(dead) == 0 or not s.hard: s.usage[:] = 0; return
        dead = dead[:max(1, len(s.hard) // 4)]
        pick = s.rng.choice(len(s.hard), len(dead), replace=False if len(dead) <= len(s.hard) else True)
        for j, h in zip(dead, pick):
            xh, yh = s.hard[h]
            nrm = np.linalg.norm(xh) + 1e-6
            p["W1"][:, j] = (xh / nrm) * 3.0                     # fires strongly for this context
            p["b1"][j] = -0.5 * 3.0 * nrm                        # fires when the context is close to this one
            p["Wo"][j] = 0.0; p["Wo"][j, yh] = 1.0               # votes for the answer it was born for
            for k in ("W1", "b1", "Wo"):
                o = s.opt[k]
                if k == "W1": o.m[:, j] = 0; o.v[:, j] = 0
                else: o.m[j] = 0; o.v[j] = 0
            s.stats["regrown"] += 1
        s.usage[:] = 0

def evaluate(model, split="val"):
    V = model.V; c = model.c; T = c["T"]; nval = c["nval"]
    mem = [Memory(c["H"], c["horizon"], c["cap"])]
    s = sbdata.get(split, "s"); ex8 = sbdata.get(split, "ex8")
    tot = 0.0; n = 0; tc = 0.0; nc = 0
    for st in range(16, nval - T - 1, T):
        pos = np.arange(st, st + T)
        F, _ = sbdata.feature_rows(model.feats, split, pos, V)
        y = np.asarray(s[pos + 1]).astype(np.int64)
        o = model.forward(F, mem, y, 1)
        l = model.loss(o, y); tot += l.sum(); n += len(l)
        cm = np.asarray(ex8[pos]) != V; tc += l[cm].sum(); nc += cm.sum()
    return tot / n / LN2, tc / max(nc, 1) / LN2, (tot - tc) / max(n - nc, 1) / LN2

def run(**kw):
    c = dict(DEFAULT); c.update(kw)
    V, space = sbdata.meta(); model = Model(c, V); model.it = 0
    s = sbdata.get("trn", "s"); N = len(s)
    rng = np.random.default_rng(c["seed"] + 1000)
    starts = rng.integers(16, N - c["T"] * (c["steps"] + 2) - 1, c["S"])
    mems = [Memory(c["H"], c["horizon"], c["cap"]) for _ in range(c["S"])]
    hist = []; t0 = time.time(); trl = []
    for it in range(1, c["steps"] + 1):
        model.it = it
        pos = np.concatenate([np.arange(st + (it - 1) * c["T"], st + it * c["T"]) for st in starts])
        F, _ = sbdata.feature_rows(model.feats, "trn", pos, V)
        y = np.asarray(s[pos + 1]).astype(np.int64)
        trl.append(model.step(F, y, mems, y, c["S"]).mean())
        if c["regrow"] and it % c["regrow_every"] == 0: model.regrow()
        if not c["regrow"] and it == (3 * c["steps"]) // 4: model.usage[:] = 0   # usage over the last quarter
        if it % max(1, c["steps"] // c["evals"]) == 0:
            v, vc, vo = evaluate(model)
            dead = float((model.usage == 0).mean()) if not c["regrow"] else None
            hist.append((it, it * c["S"] * c["T"], round(time.time() - t0, 1), round(v, 4), round(vc, 4), round(vo, 4)))
    u = model.usage.astype(float); pu = u / max(u.sum(), 1)
    eff = float(np.exp(-(pu[pu > 0] * np.log(pu[pu > 0])).sum())) if u.sum() > 0 else 0.0
    return dict(hist=hist, stats=model.stats, train_bpc=round(float(np.mean(trl[-50:])) / LN2, 4),
                dead_frac=round(float((u == 0).mean()), 4), eff_units=round(eff, 1))

if __name__ == "__main__":
    a = dict(x.split("=", 1) for x in sys.argv[1:])
    a = {k: json.loads(v) for k, v in a.items()}
    r = run(**a); print(json.dumps(dict(a, **r)))
