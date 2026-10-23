"""E7: hard non-differentiable retrieval vs soft attention, inside the same sparse backbone.

Windows of T chars. Backbone: feature-embedding sum -> sparse layer 1 (code a1) -> context mixer -> sparse layer 2 -> readout.
Mixers (all read 'what followed' an earlier position s<t, i.e. the char at s+1, which is known at time t):
  none    : no context mixing beyond the features
  hard    : s* = argmax_s cos(a1_t, a1_s) on the model's own sparse codes (stop-gradient, non-differentiable);
            add embedding row of (similarity bin, char after s*). O(kw * postings) with an inverted index.
  hardq   : same, but the code used for lookup is a separate learned sparse projection (trained only by the
            prediction loss through the retrieved value, i.e. it gets NO gradient through the choice)
  soft    : one softmax attention head, query/key = projections of a1, value = embedding of char after s
            (backprop through the selection; the standard relaxation)
  soft4   : 4 heads
Credit for everything except the hard choice equals the forward-only per-unit finite-difference credit.
"""
import os; os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
from common import load, LN2
from e5 import build
from jsparse import kwta, BASE, RICH, MEM

NB = 4

def init(key, nrows, de, H, V, mixer, dk=32, nh=1):
    ks = jax.random.split(key, 10)
    p = {"E": jax.random.normal(ks[0], (nrows, de)) * 0.05,
         "W1": jax.random.normal(ks[1], (de, H)) / np.sqrt(de), "b1": jnp.zeros(H),
         "W2": jax.random.normal(ks[2], (de, H)) / np.sqrt(de), "R2": jax.random.normal(ks[3], (H, H)) * np.sqrt(2.0 / H) * 0.5,
         "b2": jnp.zeros(H), "Wo": jnp.zeros((H, V)), "bo": jnp.zeros(V),
         "M": jax.random.normal(ks[4], ((NB + 1) * (V + 1), de)) * 0.05}
    if mixer.startswith("soft"):
        p["Q"] = jax.random.normal(ks[5], (nh, H, dk)) / np.sqrt(H); p["K"] = jax.random.normal(ks[6], (nh, H, dk)) / np.sqrt(H)
        p["Mv"] = jax.random.normal(ks[7], (nh, V + 1, de)) * 0.05
    if mixer in ("hardc", "hardr"):
        p["Q"] = jax.random.normal(ks[5], (H, dk)) / np.sqrt(H); p["K"] = jax.random.normal(ks[6], (H, dk)) / np.sqrt(H)
    if mixer == "hardq":
        p["P"] = jax.random.normal(ks[8], (H, H)) / np.sqrt(H)
    return p

def mix(p, a1, nxt, mixer, kw, V):
    """a1: (S,T,H) codes; nxt: (S,T) char after each position. returns (S,T,de) retrieved embedding."""
    S, T, H = a1.shape
    causal = jnp.tril(jnp.ones((T, T), bool), -1)            # s < t
    if mixer in ("hard", "hardq"):
        c = jax.lax.stop_gradient(a1) if mixer == "hard" else kwta((jax.lax.stop_gradient(a1) @ p["P"]).reshape(S * T, H), kw).reshape(S, T, H)
        c = jax.lax.stop_gradient(c)
        n = c / (jnp.linalg.norm(c, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.where(causal, n @ n.transpose(0, 2, 1), -1.0)
        s = sim.argmax(-1); best = jnp.take_along_axis(sim, s[..., None], -1)[..., 0]
        ch = jnp.take_along_axis(nxt, s, 1)
        b = jnp.clip((best * NB).astype(jnp.int32), 0, NB - 1)
        row = jnp.where(best > 0, b * (V + 1) + ch, NB * (V + 1))
        return p["M"][row]
    if mixer.startswith("soft"):
        q = jnp.einsum("sth,nhd->sntd", a1, p["Q"]); k = jnp.einsum("sth,nhd->sntd", a1, p["K"])
        att = jnp.where(causal, q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1]), -1e9)
        w = jax.nn.softmax(att, -1) * causal.any(-1)[:, None]          # row 0 attends nowhere
        vals = p["Mv"][:, nxt]                                            # (nh,S,T,de)
        return jnp.einsum("snts,nsue->ste", w, vals.transpose(1, 0, 2, 3)) if False else \
               jnp.einsum("snto,nsoe->ste", w, vals)
    return 0.0

def apply(p, F, nxt, mixer, kw, V, S, T):
    x = p["E"][F].sum(1)                                    # (S*T, de)
    a1 = kwta(x @ p["W1"] + p["b1"], kw)
    r = mix(p, a1.reshape(S, T, -1), nxt, mixer, kw, V)
    x2 = x + (r.reshape(S * T, -1) if mixer != "none" else 0.0)
    a2 = kwta(x2 @ p["W2"] + a1 @ p["R2"] + p["b2"], kw) + a1
    return a2 @ p["Wo"] + p["bo"]

def xent_rows(lg, y):
    return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[..., None], -1)[..., 0]

def apply_hardc(p, F, y, nxt, kw, V, S, T, m=4):
    """hard top-1 retrieval; the scorer learns from counterfactual losses of its top-m candidates."""
    x = p["E"][F].sum(1); a1 = kwta(x @ p["W1"] + p["b1"], kw)
    a1s = jax.lax.stop_gradient(a1).reshape(S, T, -1)
    q = a1s @ p["Q"]; k = a1s @ p["K"]
    causal = jnp.tril(jnp.ones((T, T), bool), -1)
    sim = jnp.where(causal, q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]), -1e9)
    top, cand = jax.lax.top_k(jax.lax.stop_gradient(sim), m)                 # (S,T,m)
    valid = (jnp.arange(T)[:, None] > jnp.arange(m)[None, :])[None]          # position t has >= j+1 earlier positions
    ch = jnp.take_along_axis(nxt[:, :, None].repeat(m, 2), cand, 1) if False else jnp.take_along_axis(jnp.broadcast_to(nxt[:, None, :], (S, T, T)), cand, 2)
    vals = jnp.where(valid[..., None], p["M"][ch], 0.0)                       # (S,T,m,de)
    def head(r):  # r: (S*T, de)
        a2 = kwta((x + r) @ p["W2"] + a1 @ p["R2"] + p["b2"], kw) + a1
        return a2 @ p["Wo"] + p["bo"]
    lg = head(vals[:, :, 0].reshape(S * T, -1))
    main = xent_rows(lg, y)
    # counterfactual: loss under each candidate (no gradient), teach the scorer to rank the best first
    cf = jnp.stack([xent_rows(head(jax.lax.stop_gradient(vals[:, :, j]).reshape(S * T, -1)), y) for j in range(m)], -1)
    cf = jax.lax.stop_gradient(jnp.where(valid.reshape(1, T, m).repeat(S, 0).reshape(S * T, m), cf, 1e9))
    best = cf.argmin(-1)
    sc = jnp.take_along_axis(sim, cand, 2).reshape(S * T, m)
    sc = jnp.where(valid.reshape(1, T, m).repeat(S, 0).reshape(S * T, m), sc, -1e9)
    has2 = (jnp.arange(T) >= 2)[None].repeat(S, 0).reshape(-1)
    rank = -jnp.take_along_axis(jax.nn.log_softmax(sc), best[:, None], 1)[:, 0] * has2
    return main, rank

def apply_hardr(p, F, y, nxt, kw, V, S, T, m=4):
    """candidates = top-m earlier positions by cosine of the model's own sparse codes (as 'hard');
    a learned scorer re-ranks them, taught by counterfactual losses; value row carries the cosine bin (confidence)."""
    x = p["E"][F].sum(1); a1 = kwta(x @ p["W1"] + p["b1"], kw)
    a1s = jax.lax.stop_gradient(a1).reshape(S, T, -1)
    causal = jnp.tril(jnp.ones((T, T), bool), -1)
    n = a1s / (jnp.linalg.norm(a1s, axis=-1, keepdims=True) + 1e-6)
    cos = jnp.where(causal, n @ n.transpose(0, 2, 1), -1.0)
    cosm, cand = jax.lax.top_k(cos, m)                                         # (S,T,m)
    q = a1s @ p["Q"]; k = a1s @ p["K"]
    sc = jnp.einsum("std,stmd->stm", q, k[jnp.arange(S)[:, None, None], cand]) / np.sqrt(q.shape[-1]) + 4.0 * cosm
    valid = cosm > 0
    sc = jnp.where(valid, sc, -1e9)
    pick = jax.lax.stop_gradient(sc).argmax(-1)                                # (S,T)
    ch = jnp.take_along_axis(jnp.broadcast_to(nxt[:, None, :], (S, T, T)), cand, 2)
    b = jnp.clip((cosm * NB).astype(jnp.int32), 0, NB - 1)
    rows = jnp.where(valid, b * (V + 1) + ch, NB * (V + 1))                   # (S,T,m)
    def head(r):
        a2 = kwta((x + r) @ p["W2"] + a1 @ p["R2"] + p["b2"], kw) + a1
        return a2 @ p["Wo"] + p["bo"]
    used = jnp.take_along_axis(rows, pick[..., None], 2)[..., 0].reshape(-1)
    main = xent_rows(head(p["M"][used]), y)
    cf = jnp.stack([xent_rows(head(jax.lax.stop_gradient(p["M"][rows[..., j].reshape(-1)])), y) for j in range(m)], -1)
    cf = jax.lax.stop_gradient(jnp.where(valid.reshape(-1, m), cf, 1e9)); best = cf.argmin(-1)
    rank = -jnp.take_along_axis(jax.nn.log_softmax(sc.reshape(-1, m)), best[:, None], 1)[:, 0] * valid.reshape(-1, m).sum(-1).__ge__(2)
    return main, rank

def loss(p, F, y, nxt, mixer, kw, V, S, T):
    if mixer == "hardr":
        main, rank = apply_hardr(p, F, y, nxt, kw, V, S, T)
        return main + rank - jax.lax.stop_gradient(rank)
    if mixer == "hardc":
        main, rank = apply_hardc(p, F, y, nxt, kw, V, S, T)
        return main + rank - jax.lax.stop_gradient(rank)      # value = main loss; gradient also teaches the scorer
    lg = apply(p, F, nxt, mixer, kw, V, S, T)
    return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1)[:, 0]

def copy_mask(seq, n=8):
    """positions whose previous n chars already occurred earlier in the window (copyable content)."""
    T = len(seq) - 1; seen = set(); m = np.zeros(T, bool)
    for t in range(T):
        if t + 1 >= n:
            k = seq[t + 1 - n:t + 1].tobytes(); m[t] = k in seen; seen.add(k)
    return m

def run(mixer="none", feats="base", H=1024, kw=32, de=64, steps=5000, T=256, nseq=4, lr=3e-3, seed=0, evals=5, nh=1,
        log=True, ntrain=2_000_000, nval=60_000, **_):
    fs = [tuple(f) for f in feats] if isinstance(feats, list) else {"base": BASE, "rich": RICH, "richmem": RICH + MEM, "basemem": BASE + MEM}[feats]
    trn, val, itos = load(ntrain); V = len(itos); space = itos.index(' ')
    g = {"feats": fs, "H": H, "kw": kw}
    _, _, nrows = build(g, trn, [0], T, V, space)
    p = init(jax.random.PRNGKey(seed), nrows, de, H, V, mixer, nh=nh)
    opt = optax.adam(lr); st = opt.init(p)
    def batch(stream, starts):
        F, y, _ = build(g, stream, starts, T, V, space)
        return F, y, y.reshape(len(starts), T)               # y[t] = char after position t
    @jax.jit
    def step(p, st, F, y, nxt):
        l, gr = jax.value_and_grad(lambda p: loss(p, F, y, nxt, mixer, kw, V, nseq, T).mean())(p)
        u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    evl = jax.jit(lambda p, F, y, nxt: loss(p, F, y, nxt, mixer, kw, V, 1, T))
    vstarts = list(range(0, nval - T - 1, T))
    rng = np.random.default_rng(seed); hist = []; t0 = time.time()
    for it in range(1, steps + 1):
        F, y, nxt = batch(trn, rng.integers(0, len(trn) - T - 2, nseq))
        p, st, l = step(p, st, F, y, nxt)
        if it % max(1, steps // evals) == 0:
            ls = np.concatenate([np.asarray(evl(p, *batch(val, [s0]))) for s0 in vstarts])
            cm = np.concatenate([copy_mask(val[s0:s0 + T + 1]) for s0 in vstarts])
            hist.append((it, it * nseq * T, round(time.time() - t0, 1), round(float(ls.mean()) / LN2, 4),
                         round(float(ls[cm].mean()) / LN2, 4), round(float(ls[~cm].mean()) / LN2, 4)))
            if log: print(f"jmem {mixer} {feats}", hist[-1], flush=True)
    return hist

if __name__ == "__main__":
    a = dict(x.split("=") for x in sys.argv[1:])
    a = {k: (v if k in ("mixer", "feats") else (float(v) if "." in v else int(v))) for k, v in a.items()}
    print("RESULT", json.dumps(dict(a, hist=run(**a))))
