"""Backprop attention reference: tiny causal char transformer in JAX on the same char stream."""
import os; os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpy as np, time, json, argparse
from common import load, LN2

def init(rng, V, d, L, T, ff=4):
    k = jax.random.split(rng, 4 + 6 * L); i = iter(k)
    n = lambda shape, s: jax.random.normal(next(i), shape) * s
    p = {"E": n((V, d), 0.02), "P": n((T, d), 0.02), "blocks": []}
    for _ in range(L):
        p["blocks"].append({"qkv": n((d, 3 * d), d ** -0.5), "o": n((d, d), d ** -0.5) / np.sqrt(2 * L),
                            "w1": n((d, ff * d), d ** -0.5), "w2": n((ff * d, d), (ff * d) ** -0.5) / np.sqrt(2 * L),
                            "g1": jnp.ones(d), "g2": jnp.ones(d)})
    p["gf"] = jnp.ones(d)
    return p

def rms(x, g): return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * g

def apply(p, X, nh):
    B, T = X.shape; d = p["E"].shape[1]
    h = p["E"][X] + p["P"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for b in p["blocks"]:
        q, k, v = jnp.split(rms(h, b["g1"]) @ b["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, nh, d // nh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        a = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // nh), -1e9)
        o = (jax.nn.softmax(a, -1) @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        h = h + o @ b["o"]
        h = h + jax.nn.gelu(rms(h, b["g2"]) @ b["w1"]) @ b["w2"]
    return rms(h, p["gf"]) @ p["E"].T

def loss_fn(p, X, Y, nh):
    lg = apply(p, X, nh)
    return -jnp.take_along_axis(jax.nn.log_softmax(lg), Y[..., None], -1).mean()

def run(d=64, L=2, T=128, B=32, steps=3000, lr=3e-3, nh=4, seed=0, ntrain=2_000_000, evals=10, log=True, **kw):
    trn, val, itos = load(ntrain); V = len(itos)
    p = init(jax.random.PRNGKey(seed), V, d, L, T)
    import optax
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1), weight_decay=0.01)
    st = opt.init(p)
    @jax.jit
    def step(p, st, X, Y):
        l, g = jax.value_and_grad(loss_fn)(p, X, Y, nh)
        u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    ev = jax.jit(lambda p, X, Y: loss_fn(p, X, Y, nh))
    vX = np.stack([val[i:i + T] for i in range(0, 60_000 - T - 1, T)]); vY = np.stack([val[i + 1:i + T + 1] for i in range(0, 60_000 - T - 1, T)])
    rng = np.random.default_rng(seed); t0 = time.time(); hist = []
    nparam = sum(x.size for x in jax.tree_util.tree_leaves(p))
    for s in range(1, steps + 1):
        idx = rng.integers(0, len(trn) - T - 1, B)
        X = np.stack([trn[i:i + T] for i in idx]); Y = np.stack([trn[i + 1:i + T + 1] for i in idx])
        p, st, l = step(p, st, X, Y)
        if s % max(1, steps // evals) == 0:
            v = float(np.mean([ev(p, vX[i:i + 64], vY[i:i + 64]) for i in range(0, len(vX), 64)])) / LN2
            hist.append((s, s * B * T, round(time.time() - t0, 1), round(v, 4)))
            if log: print("tfm", d, L, hist[-1], flush=True)
    return dict(nparam=nparam, hist=hist)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a, t, dflt in [("d", int, 64), ("L", int, 2), ("T", int, 128), ("B", int, 32), ("steps", int, 3000), ("lr", float, 3e-3)]:
        ap.add_argument("--" + a, type=t, default=dflt)
    a = ap.parse_args(); print("RESULT", json.dumps(run(**vars(a))))
