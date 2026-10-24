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
def _lookup3(act, val, nxt, gid0, pos_store, pval_store, cnt, head, ent_next, ent_norm, ent_gid, code_i, code_v,
             score, touched, horizon, V, topm, out_r, out_cos, out_si, out_sv, do_query=True, write_thr=2.0):
    T, k = act.shape; H, cap = pos_store.shape; M = ent_next.shape[0]
    best_s = np.empty(topm, np.int64); best_c = np.empty(topm, np.float64)
    store_code = code_i.shape[0] == M
    for t in range(T):
        g = gid0 + t
        nrm = 0.0
        for u in range(k):
            if val[t, u] > 0: nrm += val[t, u] * val[t, u]
        nrm = np.sqrt(nrm) + 1e-6
        nt = 0
        for u in range(k):
            v = val[t, u]
            if v <= 0 or not do_query: continue
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
        # successor code of the best match (the model's own fingerprint one step after the matched moment)
        for u in range(k):
            out_si[t, u] = 0; out_sv[t, u] = 0.0
        if store_code and best_s[0] >= 0 and best_c[0] > 0:
            sg = ent_gid[best_s[0]] + 1
            if sg == g:
                for u in range(k):
                    out_si[t, u] = act[t, u]; out_sv[t, u] = val[t, u] if val[t, u] > 0 else 0.0
            else:
                ss = sg % M
                if ent_gid[ss] == sg:
                    for u in range(k):
                        out_si[t, u] = code_i[ss, u]; out_sv[t, u] = code_v[ss, u]
        # write this position (novelty gate: skip if an almost identical moment is already stored)
        if best_c[0] >= write_thr: continue
        slot = g % M
        ent_gid[slot] = g; ent_next[slot] = nxt[t]; ent_norm[slot] = nrm
        for u in range(k):
            v = val[t, u]
            if store_code:
                code_i[slot, u] = act[t, u]; code_v[slot, u] = v if v > 0 else 0.0
            if v <= 0: continue
            j = act[t, u]
            if cnt[j] < cap:
                pos_store[j, cnt[j]] = g; pval_store[j, cnt[j]] = v; cnt[j] += 1
            else:
                h = head[j]; pos_store[j, h] = g; pval_store[j, h] = v; head[j] = (h + 1) % cap

class Memory:
    def __init__(self, H, horizon, cap, k=32, codes=False):
        M = 1
        while M < horizon + 2: M *= 2
        self.pos = np.zeros((H, cap), np.int64); self.pval = np.zeros((H, cap), np.float64)
        self.cnt = np.zeros(H, np.int64); self.head = np.zeros(H, np.int64)
        self.next = np.zeros(M, np.int64); self.norm = np.ones(M); self.gid = np.full(M, -1, np.int64)
        self.score = np.zeros(M); self.touched = np.zeros(M, np.int64); self.g = 0; self.horizon = horizon
        Mc = M if codes else 1
        self.code_i = np.zeros((Mc, k), np.int64); self.code_v = np.zeros((Mc, k))
    def query_write(self, act, val, nxt, V, topm, do_query=True):
        wt = getattr(self, "write_thr", 2.0)
        T, k = act.shape; r = np.empty((T, topm), np.int64); c = np.empty((T, topm))
        si = np.zeros((T, k), np.int64); sv = np.zeros((T, k))
        _lookup3(act.astype(np.int64), val.astype(np.float64), nxt.astype(np.int64), self.g, self.pos, self.pval,
                 self.cnt, self.head, self.next, self.norm, self.gid, self.code_i, self.code_v, self.score, self.touched,
                 self.horizon, V, topm, r, c, si, sv, do_query, wt)
        self.g += T
        return r, c, si, sv

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
               T=512, S=2, steps=800, evals=4, nval=40_000, rv=0, lk2=0, horizon2=1024, kw2=None,
               homeo=0.0, prefill=0, prefill_cap=None, lm={}, explore=0.0, boost=0.0, write_thr=2.0, hebb=0.0, mem_kind="unit", P=8, bcap=64, keys=[], kkw=32, kH=1024)

def sparse_rows(G, rows):
    """dense rows of a scipy sparse gradient."""
    return np.asarray(G.tocsr()[rows].toarray(), np.float32)

class Model:
    def __init__(s, cfg, V):
        s.c = cfg; s.V = V; rng = np.random.default_rng(cfg["seed"]); s.rng = rng
        H, de = cfg["H"], cfg["de"]
        s.feats = [tuple(f) for f in cfg["feats"]]
        s.nrows = sum(sbdata.block_size(f, V) for f in s.feats)
        f32 = np.float32; nr = (NB + 1) * (V + 1)
        s.p = {"E": (rng.normal(0, 0.05, (s.nrows, de))).astype(f32),
               "W1": (rng.normal(0, 1, (de, H)) / np.sqrt(de)).astype(f32), "b1": np.zeros(H, f32),
               "Wo": np.zeros((H, V), f32), "bo": np.zeros(V, f32)}
        if cfg["L"] == 2:
            s.p.update({"W2": (rng.normal(0, 1, (de, H)) / np.sqrt(de)).astype(f32), "b2": np.zeros(H, f32),
                        "R2": (rng.normal(0, 1, (H, H)) * np.sqrt(2.0 / H) * 0.5).astype(f32),
                        "M": (rng.normal(0, 0.05, (nr, de))).astype(f32)})
            if cfg["rv"]: s.p["Rv"] = (rng.normal(0, 1, (H, H)) * np.sqrt(2.0 / H) * 0.25).astype(f32)
        else:
            s.p["Mo"] = np.zeros((nr, V), f32)
        if cfg["lk2"]: s.p["Mo2"] = np.zeros((nr, V), f32)
        # extra lookups keyed on chosen feature subsets: key code = top-kkw of a FIXED random projection (an LSH),
        # so what each memory is keyed on is a discrete design choice, not something learned by gradient
        s.keys = [[tuple(f) for f in kf] for kf in cfg["keys"]]
        s.key_rows = [sum(sbdata.block_size(f, V) for f in kf) for kf in s.keys]
        s.key_fixed = []
        for j, kf in enumerate(s.keys):
            Ek = rng.normal(0, 1, (s.key_rows[j], de)).astype(f32); Wk = (rng.normal(0, 1, (de, cfg["kH"])) / np.sqrt(de)).astype(f32)
            s.key_fixed.append((Ek, Wk))
            if cfg["L"] == 2: s.p[f"MK{j}"] = (rng.normal(0, 0.05, (nr, de))).astype(f32)
            else: s.p[f"MKo{j}"] = np.zeros((nr, V), f32)
        grp = {"E": "E", "M": "E", "W1": "W", "b1": "W", "W2": "W", "b2": "W", "R2": "W", "Rv": "W",
               "Wo": "O", "bo": "O", "Mo": "O", "Mo2": "O"}
        for j in range(len(s.keys)): grp[f"MK{j}"] = "E"; grp[f"MKo{j}"] = "O"
        s.opt = {k: Adam(v.shape, cfg["lr"] * cfg["lm"].get(grp[k], 1.0)) for k, v in s.p.items()}
        s.usage = np.zeros(H, np.int64); s.hard = []
        s.stats = {"swap_tried": 0, "swap_good": 0, "regrown": 0, "match_rate": [], "cos": []}

    def rows(s, split, pos):
        F, _ = sbdata.feature_rows(s.feats, split, pos, s.V)
        if not s.keys: return F
        return [F] + [sbdata.feature_rows(kf, split, pos, s.V)[0] for kf in s.keys]

    def key_codes(s, Fk, j):
        Ek, Wk = s.key_fixed[j]; z = Ek[Fk].sum(1) @ Wk
        i, v = topk(z, s.c["kkw"]); return i, np.maximum(v, 0)

    def new_mems(s):
        c = s.c
        if c["mem_kind"] == "pair":
            from pairmem import PairMemory
            m1 = PairMemory(c["horizon"], P=c["P"], bcap=c["bcap"])
        else:
            m1 = Memory(c["H"], c["horizon"], c["cap"], c["kw"], codes=bool(c["rv"])); m1.write_thr = c["write_thr"]
        if c["horizon"] >= 8192:      # long key memories use the cheaper pair index
            from pairmem import PairMemory
            d = {f"k{j}": PairMemory(c["horizon"], P=c["P"], bcap=c["bcap"]) for j in range(len(s.keys))}
        else:
            d = {f"k{j}": Memory(c["kH"], c["horizon"], int(min(1024, max(64, 2 * c["horizon"] * c["kkw"] / c["kH"]))), c["kkw"])
                 for j in range(len(s.keys))}
        return {**d, "m1": m1,
                "m2": Memory(c["H"], c["horizon2"], c["cap"], c["kw"]) if c["lk2"] else None}

    def _lookup(s, mems, key, i, a, nxt, nseg):
        B = i.shape[0]; T = B // nseg; out = [[], [], [], []]
        for g in range(nseg):
            sl = slice(g * T, (g + 1) * T)
            res = mems[g][key].query_write(i[sl], a[sl], nxt[sl], s.V, s.c["topm"])
            for q in range(4): out[q].append(res[q])
        return [np.concatenate(z) for z in out]

    def forward(s, F, mems, nxt, nseg):
        p, c = s.p, s.c; H, kw = c["H"], c["kw"]; V = s.V
        Fk = []
        if isinstance(F, list): F, Fk = F[0], F[1:]
        B = F.shape[0]
        x = p["E"][F].sum(1)
        z1 = x @ p["W1"] + p["b1"]
        training = getattr(s, "training", False)
        if (c["explore"] and training) or c["boost"]:
            # selection score != activation: exploration noise (training only) and/or homeostatic boosting
            zs = z1.copy(); sd = z1.std(1, keepdims=True)
            if c["explore"] and training:
                zs += c["explore"] * sd * s.rng.standard_normal(z1.shape).astype(np.float32)
            if c["boost"]:
                if not hasattr(s, "rate"): s.rate = np.full(H, kw / H)
                zs += c["boost"] * sd * np.clip(np.log((kw / H) / (s.rate + 1e-9)), -4, 4).astype(np.float32)
            i1 = np.argpartition(zs, -kw, axis=1)[:, -kw:]; a1 = np.maximum(np.take_along_axis(z1, i1, 1), 0)
        else:
            i1, v1 = topk(z1, kw); a1 = np.maximum(v1, 0)
        o = {"x": x, "z1": z1, "i1": i1, "a1": a1}
        if c["lookup"]:
            r, co, si, sv = s._lookup(mems, "m1", i1, a1, nxt, nseg)
            s.stats["match_rate"].append(float((co[:, 0] > 0).mean())); s.stats["cos"].append(float(co[:, 0].mean()))
        else:
            r = np.full((B, 1), NB * (V + 1)); si = np.zeros((B, kw), np.int64); sv = np.zeros((B, kw))
        o["r"] = r
        A1 = csr(i1, a1, H); o["A1"] = A1
        rk = []
        for j in range(len(Fk)):
            ik, ak = s.key_codes(Fk[j], j)
            rj, _, _, _ = s._lookup(mems, f"k{j}", ik, ak, nxt, nseg); rk.append(rj)
        o["rk"] = rk
        if c["L"] == 2:
            x2 = x + p["M"][r].sum(1)
            for j, rj in enumerate(rk): x2 = x2 + p[f"MK{j}"][rj].sum(1)
            z2 = x2 @ p["W2"] + p["b2"] + A1 @ p["R2"]
            if c["rv"]:
                AS = csr(si, sv.astype(np.float32), H); o["AS"] = AS; z2 = z2 + AS @ p["Rv"]
            i2, v2 = topk(np.asarray(z2), kw); a2 = np.maximum(v2, 0)
            o.update(x2=x2, i2=i2, a2=a2)
            A = A1 + csr(i2, a2, H)
            logits = np.asarray(A @ p["Wo"]) + p["bo"]
            if c["lk2"]:
                r2, _, _, _ = s._lookup(mems, "m2", i2, a2, nxt, nseg); o["r2"] = r2
                logits = logits + p["Mo2"][r2].sum(1)
        else:
            A = A1
            logits = np.asarray(A @ p["Wo"]) + p["bo"] + p["Mo"][r].sum(1)
            for j, rj in enumerate(rk): logits = logits + p[f"MKo{j}"][rj].sum(1)
        o["A"] = A; o["logits"] = logits
        return o

    def loss(s, o, y):
        ls = logsoftmax(o["logits"]); return -ls[np.arange(len(y)), y]

    def step(s, F, y, mems, nxt, nseg):
        p, c = s.p, s.c; H, kw = c["H"], c["kw"]; B = len(y)
        s.training = True
        o = s.forward(F, mems, nxt, nseg)
        s.training = False
        if isinstance(F, list): F = F[0]
        lp = logsoftmax(o["logits"]); tokl = -lp[np.arange(B), y]
        D = np.exp(lp); D[np.arange(B), y] -= 1; D /= B
        g = {"Wo": np.asarray(o["A"].T @ D), "bo": D.sum(0)}
        dWo = D @ p["Wo"].T
        i1, a1 = o["i1"], o["a1"]; m1 = a1 > 0
        dz1 = np.take_along_axis(dWo, i1, 1) * m1
        dx2 = 0.0
        if c["L"] == 2:
            i2, a2 = o["i2"], o["a2"]; m2 = a2 > 0
            dz2 = np.take_along_axis(dWo, i2, 1) * m2
            DZ2 = csr(i2, dz2, H)
            dz1 = dz1 + np.take_along_axis(np.asarray(DZ2 @ p["R2"].T), i1, 1) * m1
            g["W2"] = np.asarray((DZ2.T @ o["x2"]).T); g["b2"] = np.asarray(DZ2.sum(0)).ravel()
            rows1 = np.unique(i1); g["R2"] = (rows1, sparse_rows(o["A1"].T @ DZ2, rows1))
            if c["rv"]:
                rs = np.unique(o["AS"].indices); g["Rv"] = (rs, sparse_rows(o["AS"].T @ DZ2, rs))
            dx2 = np.asarray(DZ2 @ p["W2"].T)
            g["M"] = scatter_rows(o["r"], np.repeat(dx2[:, None, :], o["r"].shape[1], 1))
            for j, rj in enumerate(o["rk"]): g[f"MK{j}"] = scatter_rows(rj, np.repeat(dx2[:, None, :], rj.shape[1], 1))
            if c["lk2"]: g["Mo2"] = scatter_rows(o["r2"], np.repeat(D[:, None, :], o["r2"].shape[1], 1))
        else:
            g["Mo"] = scatter_rows(o["r"], np.repeat(D[:, None, :], o["r"].shape[1], 1))
            for j, rj in enumerate(o["rk"]): g[f"MKo{j}"] = scatter_rows(rj, np.repeat(D[:, None, :], rj.shape[1], 1))
        extra_i = extra_d = None
        if c["swap"] or c["ste"]:
            m = max(c["swap"], c["ste"])
            z1 = o["z1"].copy(); np.put_along_axis(z1, i1, -np.inf, 1)
            ri, rv = topk(z1, m)
            if c["ste"]:
                extra_i = ri; extra_d = np.take_along_axis(dWo, ri, 1) * (rv > 0)
            else:
                w = np.argmin(np.where(m1, a1, np.inf), 1); wi = i1[np.arange(B), w]; wa = a1[np.arange(B), w]
                base = o["logits"]; Wo = p["Wo"]
                ed = np.zeros((B, m), np.float32); dw = np.zeros(B, np.float32)
                for j in range(m):
                    aj = np.maximum(rv[:, j], 0) + 1e-3
                    lg = base + aj[:, None] * Wo[ri[:, j]] - wa[:, None] * Wo[wi]
                    gain = tokl + logsoftmax(lg)[np.arange(B), y]
                    good = gain > 0
                    s.stats["swap_tried"] += B; s.stats["swap_good"] += int(good.sum())
                    ed[:, j] = -c["swap_lr"] * np.where(good, gain, 0) / B
                    dw += c["swap_lr"] * np.where(good, gain, 0) / B
                extra_i = np.concatenate([ri, wi[:, None]], 1); extra_d = np.concatenate([ed, dw[:, None]], 1)
        DZ1 = csr(i1, dz1, H)
        if extra_i is not None: DZ1 = DZ1 + csr(extra_i, extra_d.astype(np.float32), H)
        g["W1"] = np.asarray((DZ1.T @ o["x"]).T); g["b1"] = np.asarray(DZ1.sum(0)).ravel()
        dx = np.asarray(DZ1 @ p["W1"].T) + dx2
        g["E"] = scatter_rows(F, np.repeat(dx[:, None, :], F.shape[1], 1))
        if c.get("grad_only"):
            s.last_grads = g; return tokl
        for k, v in g.items():
            if isinstance(v, tuple): s.opt[k].rows(p[k], v[0], v[1])
            else: s.opt[k].dense(p[k], np.asarray(v, np.float32))
        np.add.at(s.usage, i1[m1], 1)
        if c["hebb"]:
            # competitive Hebbian learning (no error signal): each winning unit's input weights move toward the
            # average context it won, keeping their length -> units tile the space of contexts
            cntj = np.bincount(i1.ravel(), minlength=H).astype(np.float32)
            sumx = np.asarray(csr(i1, np.ones_like(a1, dtype=np.float32), H).T @ o["x"]).T   # (de, H)
            won = cntj > 0
            cen = sumx[:, won] / cntj[won]
            Wj = p["W1"][:, won]; nw = np.linalg.norm(Wj, axis=0, keepdims=True) + 1e-6
            tgt = cen / (np.linalg.norm(cen, axis=0, keepdims=True) + 1e-6) * nw
            p["W1"][:, won] = Wj + c["hebb"] * (tgt - Wj)
        if c["boost"]:
            if not hasattr(s, "rate"): s.rate = np.full(H, kw / H)
            fired = np.zeros(H); np.add.at(fired, i1.ravel(), 1.0); s.rate = 0.995 * s.rate + 0.005 * fired / B
        if c["homeo"]:
            # intrinsic plasticity: each unit's threshold drifts so its firing rate approaches kw/H
            if not hasattr(s, "rate"): s.rate = np.full(H, kw / H)
            fired = np.zeros(H); np.add.at(fired, i1.ravel(), 1.0); s.rate = 0.99 * s.rate + 0.01 * fired / B
            p["b1"] -= c["homeo"] * (s.rate / (kw / H) - 1.0).astype(np.float32)
        if c["regrow"]:
            hi = np.argsort(-tokl)[:8]
            for h in hi: s.hard.append((o["x"][h].copy(), int(y[h])))
            s.hard = s.hard[-512:]
        return tokl

    def regrow(s):
        c, p = s.c, s.p
        dead = np.where(s.usage < c["regrow_thr"])[0]
        if len(dead) == 0 or not s.hard: s.usage[:] = 0; return
        dead = dead[:max(1, len(s.hard) // 4)]
        pick = s.rng.choice(len(s.hard), len(dead), replace=len(dead) > len(s.hard))
        for j, h in zip(dead, pick):
            xh, yh = s.hard[h]; nrm = np.linalg.norm(xh) + 1e-6
            p["W1"][:, j] = (xh / nrm) * 3.0; p["b1"][j] = -0.5 * 3.0 * nrm
            p["Wo"][j] = 0.0; p["Wo"][j, yh] = 1.0
            s.opt["W1"].m[:, j] = 0; s.opt["W1"].v[:, j] = 0
            for k in ("b1", "Wo"): s.opt[k].m[j] = 0; s.opt[k].v[j] = 0
            s.stats["regrown"] += 1
        s.usage[:] = 0

def write_only(model, mem, split, pos):
    """read text into all memories without scoring or learning (codes from the main layer and from each key)."""
    V = model.V; c = model.c; p = model.p
    s = sbdata.get(split, "s"); yy = np.asarray(s[pos + 1]).astype(np.int64)
    F = model.rows(split, pos); Fk = []
    if isinstance(F, list): F, Fk = F[0], F[1:]
    x = p["E"][F].sum(1); z1 = x @ p["W1"] + p["b1"]
    i1, v1 = topk(z1, c["kw"]); mem["m1"].query_write(i1, np.maximum(v1, 0), yy, V, 1, do_query=False)
    for j in range(len(Fk)):
        ik, ak = model.key_codes(Fk[j], j); mem[f"k{j}"].query_write(ik, ak, yy, V, 1, do_query=False)

def long_mems(model, hz, cap, kind="unit", P=8, bcap=32):
    c = model.c
    if kind == "pair":
        from pairmem import PairMemory
        m = {"m1": PairMemory(hz, P=P, bcap=bcap), "m2": None}
    else:
        m = {"m1": Memory(c["H"], hz, cap, c["kw"], codes=bool(c["rv"])), "m2": None}; m["m1"].write_thr = c["write_thr"]
    for j in range(len(model.keys)):
        if kind == "pair": m[f"k{j}"] = PairMemory(hz, P=P, bcap=bcap)
        else: m[f"k{j}"] = Memory(c["kH"], hz, int(cap * c["kkw"] / c["kw"] * c["H"] / c["kH"]) or 64, c["kkw"])
    return m

def evaluate(model, split="val", prefill=None):
    V = model.V; c = model.c; T = c["T"]; nval = c["nval"]
    prefill = c["prefill"] if prefill is None else prefill
    if prefill:
        cc = dict(c); cc["horizon"] = prefill + nval + T; cc["cap"] = c["prefill_cap"] or c["cap"]
        mem = [long_mems(model, cc["horizon"], cc["cap"])]
        st_tr = sbdata.get("trn", "s"); start = len(st_tr) - prefill - T - 2
        for st in range(start, start + prefill, T):
            write_only(model, mem[0], "trn", np.arange(st, st + T))
    else:
        mem = [model.new_mems()]
    s = sbdata.get(split, "s"); ex8 = sbdata.get(split, "ex8")
    tot = 0.0; n = 0; tc = 0.0; nc = 0
    for st in range(16, nval - T - 1, T):
        pos = np.arange(st, st + T)
        F = model.rows(split, pos)
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
    mems = [model.new_mems() for _ in range(c["S"])]
    hist = []; t0 = time.time(); trl = []
    for it in range(1, c["steps"] + 1):
        model.it = it
        pos = np.concatenate([np.arange(st + (it - 1) * c["T"], st + it * c["T"]) for st in starts])
        F = model.rows("trn", pos)
        y = np.asarray(s[pos + 1]).astype(np.int64)
        trl.append(model.step(F, y, mems, y, c["S"]).mean())
        if c["regrow"] and it % c["regrow_every"] == 0: model.regrow()
        if not c["regrow"] and it == (3 * c["steps"]) // 4: model.usage[:] = 0
        if it % max(1, c["steps"] // c["evals"]) == 0:
            v, vc, vo = evaluate(model)
            hist.append((it, it * c["S"] * c["T"], round(time.time() - t0, 1), round(v, 4), round(vc, 4), round(vo, 4)))
    u = model.usage.astype(float); pu = u / max(u.sum(), 1)
    eff = float(np.exp(-(pu[pu > 0] * np.log(pu[pu > 0])).sum())) if u.sum() > 0 else 0.0
    st = model.stats
    st = dict(swap_good_frac=round(st["swap_good"] / max(st["swap_tried"], 1), 4), regrown=st["regrown"],
              match_rate=round(float(np.mean(st["match_rate"][-100:])), 3) if st["match_rate"] else None,
              mean_cos=round(float(np.mean(st["cos"][-100:])), 3) if st["cos"] else None)
    res = dict(hist=hist, stats=st, train_bpc=round(float(np.mean(trl[-50:])) / LN2, 4),
                dead_frac=round(float((u == 0).mean()), 4), eff_units=round(eff, 1))
    if c.get("return_model"): res["model"] = model
    return res

if __name__ == "__main__":
    a = dict(x.split("=", 1) for x in sys.argv[1:])
    a = {k: json.loads(v) for k, v in a.items()}
    r = run(**a); print(json.dumps(dict(a, **r)))

def evaluate_long(model, history, cap, nval=40_000, split="vlong", horizon=None, kind="unit", P=8, bcap=32,
                  eval_start=None):
    """Read `history` chars of a long validation stream into memory (write-only, no learning), then score the
    next `nval` chars with lookups over the whole history. Measures retrieval over long text."""
    V = model.V; c = model.c; T = c["T"]; p = model.p
    hz = horizon or (history + nval + T)
    mem = [long_mems(model, hz, cap, kind, P, bcap)]
    s = sbdata.get(split, "s"); ex8 = sbdata.get(split, "ex8")
    e0 = (16 + history) if eval_start is None else eval_start      # fixed scoring window when eval_start is given
    for st in range(e0 - history, e0, T):
        write_only(model, mem[0], split, np.arange(st, st + T))
    tot = 0.0; n = 0; tc = 0.0; nc = 0
    for st in range(e0, e0 + nval, T):
        pos = np.arange(st, st + T)
        F = model.rows(split, pos)
        y = np.asarray(s[pos + 1]).astype(np.int64)
        o = model.forward(F, mem, y, 1)
        l = model.loss(o, y); tot += l.sum(); n += len(l)
        cm = np.asarray(ex8[pos]) != V; tc += l[cm].sum(); nc += cm.sum()
    return tot / n / LN2, tc / max(nc, 1) / LN2, (tot - tc) / max(n - nc, 1) / LN2
