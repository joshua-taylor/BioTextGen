"""
Structural census: which graph properties of the connectome are (a) present and
(b) not explained by its own degree sequence.

Every statistic is reported as a z-score against an ensemble of nulls. A property
with |z| < 2 against degree_preserving is, for architecture-design purposes,
just a consequence of the degree distribution -- copy the degree distribution,
not the wiring.

Includes spectral radius and effective rank, which are the quantities that
actually decide whether a fixed random recurrent matrix is usable as a reservoir.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

from nulls import ALL_NULLS


def _binary(cx):
    B = (cx.W > 0).astype(np.int8)
    return B.tocsr()


def reciprocity(cx):
    B = _binary(cx)
    both = B.multiply(B.T).nnz
    return both / max(B.nnz, 1)


def sampled_triad_census(cx, n_samples=30000, seed=0):
    """
    Frequency of each of the 16 isomorphism classes of directed 3-node subgraph,
    over triples sampled uniformly from CONNECTED triples (two neighbours of a
    random node). Uses the standard MAN triad codes -- same classification as
    networkx.triadic_census, but sampled, since full enumeration is O(n^3).

    Read the classes as: 030T = feed-forward loop, 030C = 3-cycle,
    021D = fan-out, 021U = fan-in, 021C = cascade, 300 = fully reciprocal clique.
    Relative frequencies are only meaningful against a null with the same
    sampling scheme -- which is exactly what census_vs_nulls does.
    """
    from networkx.algorithms.triads import TRICODES, TRIAD_NAMES
    rng = np.random.default_rng(seed)
    B = _binary(cx)
    Bu = ((B + B.T) > 0).tocsr()
    indptr, indices = Bu.indptr, Bu.indices
    counts = np.zeros(16, np.int64)
    deg = np.diff(indptr)
    valid = np.flatnonzero(deg >= 2)
    if len(valid) == 0:
        return {f"triad_{t}": 0.0 for t in TRIAD_NAMES}

    # vectorised: draw all triples at once, then batch the six edge lookups
    vs = valid[rng.integers(0, len(valid), n_samples)]
    offs = rng.random((n_samples, 2))
    d = deg[vs]
    ia = (offs[:, 0] * d).astype(np.int64)
    ib = (offs[:, 1] * (d - 1)).astype(np.int64)
    ib += (ib >= ia)
    aa = indices[indptr[vs] + ia]
    bb = indices[indptr[vs] + ib]

    def edges(src, dst):
        return np.asarray(B[src, dst]).ravel() != 0

    va, av = edges(vs, aa), edges(aa, vs)
    vb, bv = edges(vs, bb), edges(bb, vs)
    ab, ba = edges(aa, bb), edges(bb, aa)

    code = (va.astype(np.int64) + 2 * av + 4 * vb + 8 * bv + 16 * ab + 32 * ba)
    tri = np.array(TRICODES, np.int64)[code] - 1
    counts = np.bincount(tri, minlength=16)
    tot = max(counts.sum(), 1)
    return {f"triad_{TRIAD_NAMES[i]}": float(counts[i] / tot) for i in range(16)}


def modularity(cx, seed=0):
    B = _binary(cx)
    G = nx.from_scipy_sparse_array(sp.triu(((B + B.T) > 0).astype(np.int8)), create_using=nx.Graph)
    comms = nx.community.louvain_communities(G, seed=seed)
    return nx.community.modularity(G, comms), len(comms)


def spectral(cx, k=1):
    """
    Spectral radius of the signed weight matrix, row-normalised.
    For reservoir use the echo state property needs rho < 1 after scaling;
    what matters architecturally is how far the raw graph sits from that, and
    how the E/I balance shapes it.
    """
    A = cx.signed().astype(np.float64)
    rs = np.abs(A).sum(1).A.ravel()
    rs[rs == 0] = 1.0
    An = sp.diags(1.0 / rs) @ A
    try:
        ev = spla.eigs(An, k=min(k + 4, cx.n - 2), return_eigenvectors=False,
                       maxiter=3000, tol=1e-4)
        return float(np.max(np.abs(ev)))
    except Exception:
        return float("nan")


def effective_rank(cx, n_probe=400, seed=0):
    """Entropy-based effective rank of a random projection of the signed matrix."""
    rng = np.random.default_rng(seed)
    A = cx.signed().astype(np.float64)
    P = rng.normal(size=(cx.n, n_probe)) / np.sqrt(n_probe)
    Y = A @ P
    s = np.linalg.svd(Y, compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def census(cx, seed=0):
    d = dict(cx.summary())
    d["reciprocity"] = reciprocity(cx)
    q, nc = modularity(cx, seed)
    d["modularity_Q"] = q
    d["n_communities"] = nc
    d["spectral_radius_rownorm"] = spectral(cx)
    d["effective_rank"] = effective_rank(cx, seed=seed)
    d.update(sampled_triad_census(cx, seed=seed))
    return d


def census_vs_nulls(cx, null_names=("degree_preserving", "configuration"),
                    n_null=5, seed=0, verbose=True):
    """
    Returns {stat: {observed, null_mean, null_std, z}} per null model.
    z = (observed - null_mean) / null_std. |z| > 2 means the property is not
    explained by whatever that null preserves.
    """
    obs = census(cx, seed=seed)
    out = {}
    for nm in null_names:
        fn = ALL_NULLS[nm]
        rows = []
        for s in range(n_null):
            if verbose:
                print(f"    null {nm} rep {s+1}/{n_null}", flush=True)
            rows.append(census(fn(cx, seed=1000 + s), seed=s))
        keys = [k for k in obs if isinstance(obs[k], (int, float))]
        res = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], float)
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            res[k] = {"observed": float(obs[k]), "null_mean": float(mu),
                      "null_std": float(sd),
                      "z": float((obs[k] - mu) / sd) if sd > 1e-12 else 0.0}
        out[nm] = res
    return obs, out
