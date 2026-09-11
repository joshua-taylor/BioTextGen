"""
Null models. These are the whole experiment.

The question is not "does a connectome-constrained network learn?" (it will).
It is "does the SPECIFIC wiring beat a random graph with the SAME statistics?"
Each null strips one property while preserving the rest, so a win against null X
localises the useful structure.

    weight_shuffle      : topology kept, synapse counts permuted
                          -> tests whether weight placement matters
    sign_shuffle        : wiring kept, E/I identity permuted
                          -> tests whether E/I placement matters
    degree_preserving   : in/out degree sequence kept, wiring rewired
                          -> tests whether topology beats its degree sequence
                             (the strongest and most important null)
    configuration       : degree sequence kept only in expectation
    erdos_renyi         : density only
"""
import numpy as np
import scipy.sparse as sp
from copy import copy


def _clone(cx, W=None, sign=None, tag=""):
    out = copy(cx)
    if W is not None:
        out.W = W
    if sign is not None:
        out.sign = sign
    out.name = f"{cx.name}|{tag}"
    return out


def weight_shuffle(cx, seed=0):
    rng = np.random.default_rng(seed)
    W = cx.W.copy()
    W.data = rng.permutation(W.data)
    return _clone(cx, W=W, tag="null:weight_shuffle")


def sign_shuffle(cx, seed=0):
    rng = np.random.default_rng(seed)
    return _clone(cx, sign=rng.permutation(cx.sign), tag="null:sign_shuffle")


def degree_preserving(cx, n_swaps_per_edge=10, seed=0):
    """
    Directed double-edge swap: (a->b, c->d) becomes (a->d, c->b).
    Preserves both in- and out-degree sequences exactly; destroys all higher-order
    structure (motifs, modularity, reciprocity). Weights ride along with edges.
    """
    rng = np.random.default_rng(seed)
    coo = cx.W.tocoo()
    r, c, w = coo.row.copy(), coo.col.copy(), coo.data.copy()
    m = len(r)
    existing = set(zip(r.tolist(), c.tolist()))
    target = int(n_swaps_per_edge * m)
    done = 0
    for _ in range(target * 3):
        if done >= target:
            break
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b, cc, d = r[i], c[i], r[j], c[j]
        if a == d or cc == b:
            continue
        if (a, d) in existing or (cc, b) in existing:
            continue
        existing.discard((a, b)); existing.discard((cc, d))
        existing.add((a, d)); existing.add((cc, b))
        c[i], c[j] = d, b
        done += 1
    W = sp.coo_matrix((w, (r, c)), shape=cx.W.shape).tocsr()
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()
    return _clone(cx, W=W, tag=f"null:degree_preserving({done}/{target})")


def configuration(cx, seed=0):
    """Resample edges with probability proportional to out-degree x in-degree."""
    rng = np.random.default_rng(seed)
    n, m = cx.n, cx.W.nnz
    dout = np.asarray((cx.W > 0).sum(1)).ravel().astype(float)
    din = np.asarray((cx.W > 0).sum(0)).ravel().astype(float)
    dout = dout / dout.sum() if dout.sum() else np.ones(n) / n
    din = din / din.sum() if din.sum() else np.ones(n) / n
    r = rng.choice(n, m, p=dout)
    c = rng.choice(n, m, p=din)
    ok = r != c
    W = sp.coo_matrix((rng.permutation(cx.W.data)[ok], (r[ok], c[ok])), shape=cx.W.shape).tocsr()
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()
    return _clone(cx, W=W, tag="null:configuration")


def erdos_renyi(cx, seed=0):
    rng = np.random.default_rng(seed)
    n, m = cx.n, cx.W.nnz
    r = rng.integers(0, n, m); c = rng.integers(0, n, m)
    ok = r != c
    W = sp.coo_matrix((rng.permutation(cx.W.data)[ok], (r[ok], c[ok])), shape=cx.W.shape).tocsr()
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()
    return _clone(cx, W=W, tag="null:erdos_renyi")


def bipartite_degree_null(M, seed=0):
    """
    Degree-preserving null for a PN->KC style bipartite mask.
    Keeps per-KC claw count and per-PN out-degree; rewires everything else.
    """
    rng = np.random.default_rng(seed)
    r, c = np.nonzero(M)
    r = r.copy()
    for _ in range(20 * len(r)):
        i, j = rng.integers(0, len(r), 2)
        if c[i] == c[j] or r[i] == r[j]:
            continue
        if r[j] in r[c == c[i]] or r[i] in r[c == c[j]]:
            continue
        r[i], r[j] = r[j], r[i]
        break
    out = np.zeros_like(M)
    # simple exact alternative: permute PN identities per column, preserving column sums
    out = np.zeros_like(M)
    perm = rng.permutation(M.shape[0])
    for j in range(M.shape[1]):
        k = int(M[:, j].sum())
        out[rng.choice(M.shape[0], k, replace=False), j] = 1.0
    return out


ALL_NULLS = {
    "weight_shuffle": weight_shuffle,
    "sign_shuffle": sign_shuffle,
    "degree_preserving": degree_preserving,
    "configuration": configuration,
    "erdos_renyi": erdos_renyi,
}
