"""Forward-only training of the sparse net WITH hard retrieval (jmem mixers 'hard' and 'hardc').

Cone per token: 64 layer-1 input dims + 64 layer-2 input dims (embedding + retrieved value) + kw + kw active units.
Each cone item is probed with a forward-mode derivative of that token's own loss; weights update locally.
'hardc': the retrieval scorer (q.k on the layer-1 code) learns from counterfactual forward passes: each token tries
its top-m candidates, and the scorer is nudged to rank the one that gave the lowest loss first. That nudge is a
local update of the scorer's own bilinear form; the network's loss is never back-propagated.
"""
import os; os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
from common import load, LN2
from e5 import build
from jsparse import kwta, BASE
import jmem

def tok_loss(lg, y): return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1)[:, 0]

def retrieve(p, a1, nxt, mixer, S, T, V, m=4):
    """returns value rows (S*T,) used, plus (for hardc) candidates and validity."""
    a1s = jax.lax.stop_gradient(a1).reshape(S, T, -1)
    causal = jnp.tril(jnp.ones((T, T), bool), -1)
    if mixer == "hard":
        n = a1s / (jnp.linalg.norm(a1s, axis=-1, keepdims=True) + 1e-6)
        sim = jnp.where(causal, n @ n.transpose(0, 2, 1), -1.0)
        s = sim.argmax(-1); best = jnp.take_along_axis(sim, s[..., None], -1)[..., 0]
        ch = jnp.take_along_axis(nxt, s, 1); b = jnp.clip((best * jmem.NB).astype(jnp.int32), 0, jmem.NB - 1)
        return jnp.where(best > 0, b * (V + 1) + ch, jmem.NB * (V + 1)).reshape(-1), None
    q = a1s @ p["Q"]; k = a1s @ p["K"]
    sim = jnp.where(causal, q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]), -1e9)
    _, cand = jax.lax.top_k(sim, m)
    ch = jnp.take_along_axis(jnp.broadcast_to(nxt[:, None, :], (S, T, T)), cand, 2)        # (S,T,m) chars
    valid = jnp.broadcast_to((jnp.arange(T)[:, None] > jnp.arange(m)[None, :])[None], (S, T, m))
    rows = ch                                       # value rows: bin-0 block, row = char (invalid -> zero vector)
    return rows[..., 0].reshape(-1), (cand, rows, valid)

def masked(p, x1, x2, i1, i2, e1, e2):
    z1 = jnp.einsum("bd,bdk->bk", x1, p["W1"][:, i1].transpose(1, 0, 2)) + p["b1"][i1] + e1; a1 = jax.nn.relu(z1)
    R = p["R2"][i1[:, :, None], i2[:, None, :]]
    z2 = jnp.einsum("bd,bdk->bk", x2, p["W2"][:, i2].transpose(1, 0, 2)) + p["b2"][i2] + jnp.einsum("bi,bij->bj", a1, R) + e2
    a2 = jax.nn.relu(z2)
    return a1, a2, jnp.einsum("bk,bkv->bv", a1, p["Wo"][i1]) + jnp.einsum("bk,bkv->bv", a2, p["Wo"][i2]) + p["bo"]

def full_head(p, x2, a1f, kw):
    a2 = kwta(x2 @ p["W2"] + a1f @ p["R2"] + p["b2"], kw) + a1f
    return a2 @ p["Wo"] + p["bo"]

def fwd_grads(p, F, y, nxt, mixer, kw, V, S, T, m=4):
    B = F.shape[0]; x = p["E"][F].sum(1); de = x.shape[1]
    z1f = x @ p["W1"] + p["b1"]; a1f = kwta(z1f, kw); i1 = jax.lax.top_k(z1f, kw)[1]
    row, extra = retrieve(p, a1f, nxt, mixer, S, T, V, m)
    vmask = jnp.ones(B) if extra is None else extra[2][..., 0].reshape(-1).astype(jnp.float32)
    r = p["M"][row] * vmask[:, None]; x2 = x + r
    i2 = jax.lax.top_k(x2 @ p["W2"] + a1f @ p["R2"] + p["b2"], kw)[1]
    zk = jnp.zeros((B, kw))
    a1, a2, lg = masked(p, x, x2, i1, i2, zk, zk)
    f = lambda u1, u2, e1, e2: tok_loss(masked(p, u1, u2, i1, i2, e1, e2)[2], y)
    jv = lambda fn, base, n: jax.vmap(lambda t: jax.jvp(fn, (base,), (jnp.broadcast_to(t, base.shape),))[1])(jnp.eye(n)).T / B
    c_x1 = jv(lambda u: f(u, x2, zk, zk), x, de); c_x2 = jv(lambda u: f(x, u, zk, zk), x2, de)
    c1 = jv(lambda u: f(x, x2, u, zk), zk, kw) * (a1 > 0); c2 = jv(lambda u: f(x, x2, zk, u), zk, kw) * (a2 > 0)
    D = (jax.nn.softmax(lg) - jax.nn.one_hot(y, V)) / B
    H = p["Wo"].shape[0]
    g = {k: jnp.zeros_like(v) for k, v in p.items()}
    g["Wo"] = g["Wo"].at[i1].add(a1[:, :, None] * D[:, None, :]).at[i2].add(a2[:, :, None] * D[:, None, :])
    g["bo"] = D.sum(0)
    g["W1"] = g["W1"].at[:, i1].add(jnp.einsum("bd,bk->dbk", x, c1)); g["b1"] = g["b1"].at[i1].add(c1)
    g["W2"] = g["W2"].at[:, i2].add(jnp.einsum("bd,bk->dbk", x2, c2)); g["b2"] = g["b2"].at[i2].add(c2)
    g["R2"] = g["R2"].at[i1[:, :, None], i2[:, None, :]].add(a1[:, :, None] * c2[:, None, :])
    g["E"] = g["E"].at[F].add(jnp.broadcast_to((c_x1 + c_x2)[:, None, :], F.shape + (de,)))
    g["M"] = g["M"].at[row].add(c_x2 * vmask[:, None])
    if mixer == "hardc":
        cand, rows, valid = extra
        vflat = valid.reshape(B, m)
        cf = jnp.stack([tok_loss(full_head(p, x + p["M"][rows[..., j].reshape(-1)] * vflat[:, j:j + 1], a1f, kw), y) for j in range(m)], -1); cf = jnp.where(vflat, cf, 1e9); best = cf.argmin(-1)
        has2 = (jnp.arange(T) >= 2)[None].repeat(S, 0).reshape(-1)
        a1s = jax.lax.stop_gradient(a1f).reshape(S, T, -1)
        def rank(QK):   # the scorer's own local objective
            q = a1s @ QK[0]; k = a1s @ QK[1]
            sc = jnp.einsum("std,stmd->stm", q, k[jnp.arange(S)[:, None, None], cand]) / np.sqrt(q.shape[-1])
            sc = jnp.where(valid, sc, -1e9).reshape(B, m)
            return (-jnp.take_along_axis(jax.nn.log_softmax(sc), best[:, None], 1)[:, 0] * has2).mean()
        gQ, gK = jax.grad(rank)((p["Q"], p["K"])); g["Q"], g["K"] = gQ, gK
    return g, tok_loss(lg, y).mean()

def run(mixer="hard", mode="fwd", H=1024, kw=32, de=64, steps=5000, T=256, nseq=4, lr=3e-3, seed=0, evals=5, log=True, **_):
    trn, val, itos = load(2_000_000); V = len(itos); space = itos.index(' ')
    g = {"feats": BASE, "H": H, "kw": kw}; _, _, nrows = build(g, trn, [0], T, V, space)
    p = jmem.init(jax.random.PRNGKey(seed), nrows, de, H, V, mixer)
    opt = optax.adam(lr); st = opt.init(p)
    def batch(stream, starts):
        F, y, _ = build(g, stream, starts, T, V, space); return F, y, y.reshape(len(starts), T)
    @jax.jit
    def step_fwd(p, st, F, y, nxt):
        gr, l = fwd_grads(p, F, y, nxt, mixer, kw, V, nseq, T); u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    @jax.jit
    def step_bp(p, st, F, y, nxt):
        l, gr = jax.value_and_grad(lambda p: jmem.loss(p, F, y, nxt, mixer, kw, V, nseq, T).mean())(p)
        u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    step = step_fwd if mode == "fwd" else step_bp
    evl = jax.jit(lambda p, F, y, nxt: jmem.loss(p, F, y, nxt, mixer, kw, V, 1, T))
    rng = np.random.default_rng(seed); hist = []; t0 = time.time()
    for it in range(1, steps + 1):
        p, st, l = step(p, st, *batch(trn, rng.integers(0, len(trn) - T - 2, nseq)))
        if it % max(1, steps // evals) == 0:
            ls = np.concatenate([np.asarray(evl(p, *batch(val, [s0]))) for s0 in range(0, 60_000 - T - 1, T)])
            hist.append((it, it * nseq * T, round(time.time() - t0, 1), round(float(ls.mean()) / LN2, 4)))
            if log: print(f"jfdo_mem {mixer} {mode}", hist[-1], flush=True)
    return hist

if __name__ == "__main__":
    if sys.argv[1] == "check": check(sys.argv[2]); sys.exit()
    a = dict(x.split("=") for x in sys.argv[1:])
    a = {k: (v if k in ("mixer", "mode") else (float(v) if "." in v else int(v))) for k, v in a.items()}
    print("RESULT", json.dumps(dict(a, hist=run(**a))))

def check(mixer):
    trn, val, itos = load(2_000_000); V = len(itos); space = itos.index(' ')
    g = {"feats": BASE, "H": 1024, "kw": 32}; F, y, nrows = build(g, trn, [1000, 70000], 256, V, space)
    p = jmem.init(jax.random.PRNGKey(0), nrows, 64, 1024, V, mixer)
    p["Wo"] = jax.random.normal(jax.random.PRNGKey(1), p["Wo"].shape) * 0.05
    nxt = y.reshape(2, 256)
    gf, _ = jax.jit(lambda p: fwd_grads(p, F, y, nxt, mixer, 32, V, 2, 256))(p)
    gb = jax.grad(lambda p: jmem.loss(p, F, y, nxt, mixer, 32, V, 2, 256).mean())(p)
    for k in sorted(gb):
        a, b = np.ravel(gf[k]), np.ravel(gb[k])
        print(mixer, k, "cos", round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 6), "ratio", round(float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-12)), 4))
