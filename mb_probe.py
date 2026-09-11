"""
The mushroom-body probe: does fly-style sparse expansion buy anything on
next-token prediction, and does the ACTUAL wiring beat a claw-matched random one?

Architecture (all fixed, no backprop anywhere -- readout is closed-form ridge):

    context (C chars, one-hot)
        -> AL   : fixed random projection to n_pn, divisive normalisation
                  (the antennal lobe's gain-control motif)
        -> KC   : sparse mask M (n_pn x n_kc), ~6 claws per KC
        -> APL  : winner-take-all, keep top 5% of KCs
        -> MBON : ridge regression to next-char logits

Conditions isolate one design choice each. Pre-registered kill criterion for the
headline claim is in README.md.
"""
import numpy as np
from dataclasses import dataclass

from connectome import mushroom_body_layer
from nulls import bipartite_degree_null


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_text(path, max_chars=1_200_000):
    s = open(path, encoding="utf8", errors="ignore").read()[:max_chars]
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    vocab = [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:47]]
    stoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 = UNK
    ids = np.array([stoi.get(c, 0) for c in s], np.int16)
    return ids, len(vocab) + 1


def make_windows(ids, V, context=8, n_train=60000, n_val=12000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids) - context - 1, n_train + n_val, replace=False)
    idx.sort()
    tr, va = idx[:n_train], idx[n_train:]

    def onehot(positions):
        X = np.zeros((len(positions), context * V), np.float32)
        for k in range(context):
            X[np.arange(len(positions)), k * V + ids[positions + k]] = 1.0
        y = ids[positions + context].astype(np.int64)
        return X, y

    return onehot(tr), onehot(va)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class MBConfig:
    n_pn: int = 200
    n_kc: int = 2000
    claws: int = 6
    sparsity: float = 0.05      # APL: fraction of KCs left active
    ridge: float = 1.0
    wta: bool = True
    expand: bool = True


def al_project(X, n_pn, seed):
    """Antennal lobe: fixed random projection + divisive normalisation."""
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(X.shape[1], n_pn)).astype(np.float32) / np.sqrt(X.shape[1])
    A = X @ P
    A = np.maximum(A, 0)
    A = A / (A.mean(1, keepdims=True) + 1e-3)   # divisive gain control
    return A


def kc_activate(A, M, cfg):
    """KC layer: sparse expansion then winner-take-all."""
    if not cfg.expand:
        return A
    K = A @ M
    K -= K.mean(1, keepdims=True)
    if cfg.wta:
        k = max(1, int(cfg.sparsity * K.shape[1]))
        thr = np.partition(K, -k, axis=1)[:, -k][:, None]
        K = np.where(K >= thr, K, 0.0)
    return K


def ridge_fit(H, y, V, lam):
    Y = np.zeros((len(y), V), np.float32)
    Y[np.arange(len(y)), y] = 1.0
    H = np.hstack([H, np.ones((len(H), 1), np.float32)])
    G = H.T @ H
    G[np.diag_indices_from(G)] += lam
    return np.linalg.solve(G, H.T @ Y)


def bits_per_char(H, W, y, V):
    H = np.hstack([H, np.ones((len(H), 1), np.float32)])
    S = H @ W
    best = np.inf
    for T in np.geomspace(0.05, 20, 25):          # 1-D temperature calibration
        Z = S / T
        Z -= Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        nll = (lse - Z[np.arange(len(y)), y]).mean()
        best = min(best, nll / np.log(2))
    return float(best)


def run_condition(M, train, val, V, cfg, seed):
    (Xtr, ytr), (Xva, yva) = train, val
    Atr = al_project(Xtr, cfg.n_pn, seed)
    Ava = al_project(Xva, cfg.n_pn, seed)
    Htr = kc_activate(Atr, M, cfg)
    Hva = kc_activate(Ava, M, cfg)
    W = ridge_fit(Htr, ytr, V, cfg.ridge)
    return {"val_bpc": bits_per_char(Hva, W, yva, V),
            "train_bpc": bits_per_char(Htr, W, ytr, V),
            "active_frac": float((Htr != 0).mean()),
            "dim": Htr.shape[1]}


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------
def build_conditions(cfg, cx=None, pn_region=None, kc_region=None, seed=0):
    """Returns {name: (mask, cfg)}. Each differs from the reference by ONE thing."""
    M_real = mushroom_body_layer(cx, pn_region, kc_region,
                                 n_pn=cfg.n_pn, n_kc=cfg.n_kc,
                                 claws=cfg.claws, seed=seed)
    M_null = bipartite_degree_null(M_real, seed=seed + 77)

    rng = np.random.default_rng(seed + 5)
    M_dense = rng.normal(size=(cfg.n_pn, cfg.n_kc)).astype(np.float64) / np.sqrt(cfg.n_pn)

    from dataclasses import replace
    return {
        "mb_connectome":    (M_real,  cfg),
        "mb_degree_null":   (M_null,  cfg),
        "mb_dense_proj":    (M_dense, cfg),
        "mb_no_wta":        (M_real,  replace(cfg, wta=False)),
        "no_expansion":     (M_real,  replace(cfg, expand=False)),
    }


def sweep(text_path, cfg=MBConfig(), seeds=(0, 1, 2, 3, 4), context=8,
          n_train=60000, n_val=12000, cx=None, pn_region=None, kc_region=None,
          verbose=True):
    ids, V = load_text(text_path)
    results = {}
    for seed in seeds:
        train, val = make_windows(ids, V, context, n_train, n_val, seed=seed)
        conds = build_conditions(cfg, cx, pn_region, kc_region, seed=seed)
        for name, (M, c) in conds.items():
            r = run_condition(M, train, val, V, c, seed)
            results.setdefault(name, []).append(r)
            if verbose:
                print(f"    seed {seed}  {name:16s} val_bpc={r['val_bpc']:.4f} "
                      f"dim={r['dim']} active={r['active_frac']:.3f}", flush=True)
    return summarise(results), results


def summarise(results):
    out = {}
    for name, rs in results.items():
        v = np.array([r["val_bpc"] for r in rs])
        out[name] = {"mean_bpc": float(v.mean()), "std_bpc": float(v.std(ddof=1)),
                     "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                     "n_seeds": len(v)}
    return out


def paired_delta(results, a="mb_connectome", b="mb_degree_null"):
    """Paired (same-seed) difference: negative means `a` is better."""
    va = np.array([r["val_bpc"] for r in results[a]])
    vb = np.array([r["val_bpc"] for r in results[b]])
    d = va - vb
    return {"mean_delta_bpc": float(d.mean()),
            "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
            "seeds": len(d),
            "a": a, "b": b}
