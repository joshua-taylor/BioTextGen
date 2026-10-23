"""Deep sparse (k-WTA) char LM in JAX, for fast architecture exploration.

Gradients come from jax.grad with the winner sets held fixed (stop_gradient on the selection). On such a
fixed-selection net this equals the forward-only per-unit finite-difference credit (verified in e2b/fast.fdo),
so this file is a fast stand-in for the forward-only rule when exploring architectures.

Input per position: sum of embeddings of context features (chars at offsets 1..k, hashed n-grams, word hashes,
optional exact-match memory rows) -> L sparse layers (each also reads the input embedding) -> readout.
Optional 'mem': exact-match retrieval rows (orders 3,5,8,12) computed per window, no parameters.
"""
import os; os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
from common import load, LN2
from e5 import build

def init(key, nrows, de, H, L, V):
    ks = jax.random.split(key, 3 + 2 * L)
    p = {"E": jax.random.normal(ks[0], (nrows, de)) * 0.05,
         "Wo": jnp.zeros((H, V)), "bo": jnp.zeros(V), "layers": []}
    for l in range(L):
        p["layers"].append({"Win": jax.random.normal(ks[1 + 2 * l], (de, H)) / np.sqrt(de),
                            "Wr": jax.random.normal(ks[2 + 2 * l], (H, H)) * (np.sqrt(2.0 / H) if l else 0.0),
                            "b": jnp.zeros(H)})
    return p

def kwta(z, kw):
    if kw >= z.shape[-1]: return jax.nn.relu(z)
    thr = jax.lax.stop_gradient(jax.lax.top_k(z, kw)[0][:, -1:])
    return jnp.where(z >= thr, jax.nn.relu(z), 0.0)

def apply(p, F, kw):
    x = p["E"][F].sum(1)
    a = None
    for l, lay in enumerate(p["layers"]):
        z = x @ lay["Win"] + lay["b"] + (a @ lay["Wr"] if a is not None else 0.0)
        a = kwta(z, kw) + (a if a is not None else 0.0)      # sparse residual: units accumulate across layers
    return a @ p["Wo"] + p["bo"]

def loss(p, F, y, kw):
    lg = apply(p, F, kw)
    return -jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1).mean()

def run(feats, H=1024, kw=32, L=2, de=64, steps=5000, T=256, nseq=4, lr=3e-3, seed=0, evals=5, log=True,
        ntrain=2_000_000, nval=60_000, **_):
    trn, val, itos = load(ntrain); V = len(itos); space = itos.index(' ')
    g = {"feats": [tuple(f) for f in feats], "H": H, "kw": kw}
    _, _, nrows = build(g, trn, [0], T, V, space)
    p = init(jax.random.PRNGKey(seed), nrows, de, H, L, V)
    opt = optax.adam(lr); st = opt.init(p)
    @jax.jit
    def step(p, st, F, y):
        l, gr = jax.value_and_grad(loss)(p, F, y, kw)
        u, st = opt.update(gr, st, p); return optax.apply_updates(p, u), st, l
    evl = jax.jit(lambda p, F, y: loss(p, F, y, kw))
    VF, Vy = [], []
    for s0 in range(0, nval - T - 1, T):
        F, y, _ = build(g, val, [s0], T, V, space); VF.append(F); Vy.append(y)
    VF = np.concatenate(VF); Vy = np.concatenate(Vy)
    rng = np.random.default_rng(seed); hist = []; t0 = time.time()
    for it in range(1, steps + 1):
        F, y, _ = build(g, trn, rng.integers(0, len(trn) - T - 2, nseq), T, V, space)
        p, st, l = step(p, st, F, y)
        if it % max(1, steps // evals) == 0:
            v = float(np.mean([evl(p, VF[i:i + 4096], Vy[i:i + 4096]) for i in range(0, len(Vy) - 4095, 4096)])) / LN2
            hist.append((it, it * nseq * T, round(time.time() - t0, 1), round(v, 4)))
            if log: print(f"jsparse H{H} kw{kw} L{L}", hist[-1], flush=True)
    return hist

BASE = [("c", i) for i in range(1, 9)]
RICH = BASE + [("ng", o) for o in (2, 3, 4, 5, 6)] + [("wd",), ("pw",)]
MEM = [("ex", n) for n in (3, 5, 8, 12)]

if __name__ == "__main__":
    a = dict(x.split("=") for x in sys.argv[1:])
    fs = {"base": BASE, "rich": RICH, "richmem": RICH + MEM, "basemem": BASE + MEM}[a.pop("feats", "base")]
    a = {k: (float(v) if "." in v else int(v)) for k, v in a.items()}
    print("RESULT", json.dumps(dict(a, hist=run(fs, **a))))
