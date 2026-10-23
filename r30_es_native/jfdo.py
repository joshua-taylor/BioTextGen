"""Forward-only exact credit for the 2-layer sparse net of jsparse.py (L=2).

No backward pass and no transposed weights. Every token gets a 'population' of perturbations, one per item in
its dependency cone: the de input-embedding dims, its kw active layer-1 units and its kw active layer-2 units.
Each perturbation's effect on that token's own loss is measured with forward-mode derivatives (jax.jvp, i.e. an
infinitesimal finite difference). Weight updates are then local: (credit at a unit) x (that unit's input).
The output error (softmax - onehot) is local to the readout, as in every backprop-free rule.
"""
import os; os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
from common import load, LN2
from e5 import build
import jsparse as J

def winners(p, x, kw):
    l1, l2 = p["layers"]
    z1 = x @ l1["Win"] + l1["b"]; i1 = jax.lax.top_k(z1, kw)[1]
    a1f = J.kwta(z1, kw)
    z2 = x @ l2["Win"] + l2["b"] + a1f @ l2["Wr"]; i2 = jax.lax.top_k(z2, kw)[1]
    return i1, i2

def masked(p, x, i1, i2, e1, e2):
    """forward over active units only; e1, e2 = injected perturbations at layer-1 / layer-2 active units."""
    l1, l2 = p["layers"]
    z1 = jnp.einsum("bd,bdk->bk", x, l1["Win"][:, i1].transpose(1, 0, 2)) + l1["b"][i1] + e1
    a1 = jax.nn.relu(z1)
    R = l2["Wr"][i1[:, :, None], i2[:, None, :]]                     # (B,kw,kw) active->active
    z2 = jnp.einsum("bd,bdk->bk", x, l2["Win"][:, i2].transpose(1, 0, 2)) + l2["b"][i2] + jnp.einsum("bi,bij->bj", a1, R) + e2
    a2 = jax.nn.relu(z2)
    lg = jnp.einsum("bk,bkv->bv", a1, p["Wo"][i1]) + jnp.einsum("bk,bkv->bv", a2, p["Wo"][i2]) + p["bo"]
    return a1, a2, lg

def tok_loss(lg, y): return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1)[:, 0]

def forward_only_grads(p, F, y, kw):
    B = F.shape[0]; x = p["E"][F].sum(1); de = x.shape[1]
    i1, i2 = winners(p, x, kw)
    z = jnp.zeros((B, kw))
    a1, a2, lg = masked(p, x, i1, i2, z, z)
    f = lambda xx, e1, e2: tok_loss(masked(p, xx, i1, i2, e1, e2)[2], y)
    # one forward-mode pass per cone coordinate; each token's own loss gives its own credit
    jx = jax.vmap(lambda t: jax.jvp(lambda u: f(u, z, z), (x,), (jnp.broadcast_to(t, (B, de)),))[1])(jnp.eye(de))   # (de,B)
    j1 = jax.vmap(lambda t: jax.jvp(lambda u: f(x, u, z), (z,), (jnp.broadcast_to(t, (B, kw)),))[1])(jnp.eye(kw))    # (kw,B)
    j2 = jax.vmap(lambda t: jax.jvp(lambda u: f(x, z, u), (z,), (jnp.broadcast_to(t, (B, kw)),))[1])(jnp.eye(kw))
    cx, c1, c2 = jx.T / B, j1.T / B, j2.T / B                         # credit per token
    c1 = c1 * (a1 > 0); c2 = c2 * (a2 > 0)
    D = jax.nn.softmax(lg) - jax.nn.one_hot(y, lg.shape[1]); D = D / B   # local output error
    H, V = p["Wo"].shape
    g = jax.tree_util.tree_map(jnp.zeros_like, p)
    gWo = jnp.zeros((H, V)).at[i1].add(a1[:, :, None] * D[:, None, :]).at[i2].add(a2[:, :, None] * D[:, None, :])
    l1 = {"Win": jnp.zeros_like(p["layers"][0]["Win"]).at[:, i1].add(jnp.einsum("bd,bk->dbk", x, c1)),
          "b": jnp.zeros(H).at[i1].add(c1), "Wr": jnp.zeros_like(p["layers"][0]["Wr"])}
    gR = jnp.zeros_like(p["layers"][1]["Wr"]).at[i1[:, :, None], i2[:, None, :]].add(a1[:, :, None] * c2[:, None, :])
    l2 = {"Win": jnp.zeros_like(p["layers"][1]["Win"]).at[:, i2].add(jnp.einsum("bd,bk->dbk", x, c2)),
          "b": jnp.zeros(H).at[i2].add(c2), "Wr": gR}
    gE = jnp.zeros_like(p["E"]).at[F].add(jnp.broadcast_to(cx[:, None, :], F.shape + (de,)))
    return {"E": gE, "Wo": gWo, "bo": D.sum(0), "layers": [l1, l2]}, tok_loss(lg, y).mean()

def run(feats=J.BASE, H=1024, kw=32, de=64, steps=1000, T=256, nseq=4, lr=3e-3, seed=0, evals=5, mode="fwd", log=True, **_):
    trn, val, itos = load(2_000_000); V = len(itos); space = itos.index(' ')
    g = {"feats": [tuple(f) for f in feats], "H": H, "kw": kw}
    _, _, nrows = build(g, trn, [0], T, V, space)
    p = J.init(jax.random.PRNGKey(seed), nrows, de, H, 2, V)
    opt = optax.adam(lr); st = opt.init(p)
    @jax.jit
    def step_fwd(p, st, F, y):
        gr, l = forward_only_grads(p, F, y, kw); u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    @jax.jit
    def step_bp(p, st, F, y):
        l, gr = jax.value_and_grad(J.loss)(p, F, y, kw); u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    step = step_fwd if mode == "fwd" else step_bp
    evl = jax.jit(lambda p, F, y: J.loss(p, F, y, kw))
    VF, Vy = [], []
    for s0 in range(0, 60_000 - T - 1, T):
        F, y, _ = build(g, val, [s0], T, V, space); VF.append(F); Vy.append(y)
    VF = np.concatenate(VF); Vy = np.concatenate(Vy)
    rng = np.random.default_rng(seed); hist = []; t0 = time.time()
    for it in range(1, steps + 1):
        F, y, _ = build(g, trn, rng.integers(0, len(trn) - T - 2, nseq), T, V, space)
        p, st, l = step(p, st, F, y)
        if it % max(1, steps // evals) == 0:
            v = float(np.mean([evl(p, VF[i:i + 4096], Vy[i:i + 4096]) for i in range(0, len(Vy) - 4095, 4096)])) / LN2
            hist.append((it, it * nseq * T, round(time.time() - t0, 1), round(v, 4)))
            if log: print(f"jfdo {mode}", hist[-1], flush=True)
    return hist

def check(seed=0, kw=32, H=1024):
    """cosine between forward-only credit and backprop gradient, per parameter block."""
    trn, val, itos = load(2_000_000); V = len(itos); space = itos.index(' ')
    g = {"feats": J.BASE, "H": H, "kw": kw}
    F, y, nrows = build(g, trn, [1000, 50000], 256, V, space)
    p = J.init(jax.random.PRNGKey(seed), nrows, 64, H, 2, V)
    p["Wo"] = jax.random.normal(jax.random.PRNGKey(1), p["Wo"].shape) * 0.05
    p["layers"][1]["Wr"] = jax.random.normal(jax.random.PRNGKey(2), p["layers"][1]["Wr"].shape) * 0.03
    gf, _ = jax.jit(lambda p: forward_only_grads(p, F, y, kw))(p)
    gb = jax.grad(J.loss)(p, F, y, kw)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(gf), jax.tree_util.tree_leaves(gb)):
        a, b = np.ravel(a), np.ravel(b)
        print(jax.tree_util.keystr(path), "cos", round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 6),
              "norm ratio", round(float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-12)), 4))

if __name__ == "__main__":
    if sys.argv[1:] == ["check"]: check()
    else:
        a = dict(x.split("=") for x in sys.argv[1:]); mode = a.pop("mode", "fwd")
        a = {k: (float(v) if "." in v else int(v)) for k, v in a.items()}
        print("RESULT", json.dumps(dict(a, mode=mode, hist=run(mode=mode, **a))))
