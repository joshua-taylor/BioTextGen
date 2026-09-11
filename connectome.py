"""
Connectome container, real-data loader, and a synthetic fly-like generator.

The loader is the drop-in point for MaleCNS v1.0 / hemibrain / FlyWire exports.
Everything downstream (nulls, motifs, probes) only touches the Connectome object,
so swapping synthetic -> real is a one-line change.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp


# Neurotransmitter -> sign. Fly: ACh excitatory; GABA and Glu predominantly
# inhibitory (Glu via GluCl). Unknown -> excitatory (majority class).
NT_SIGN = {
    "acetylcholine": +1, "ach": +1, "cholinergic": +1,
    "gaba": -1, "gabaergic": -1,
    "glutamate": -1, "glu": -1, "glutamatergic": -1,
    "histamine": -1,                      # inhibitory; 7,891 neurons in MaleCNS
    "dopamine": 0, "serotonin": 0, "octopamine": 0,
    "unclear": +1, "missing": +1, "unknown": +1, "": +1,
}

# Superclasses retained when keep_central_brain=True. Drops the optic lobes
# (89,403 ol_intrinsic + 6,098 ol_sensory) and the whole VNC -- ~66% of MaleCNS
# and pure sensory/motor periphery. Counts from the MaleCNS v1.0 release.
CENTRAL_BRAIN_SUPERCLASSES = {
    "cb_intrinsic", "cb_sensory", "cb_sensory_tbc", "cb_motor", "cb_endocrine",
    "cb_efferent", "descending_neuron", "descending_neuron_tbc",
    "ascending_neuron", "sensory_ascending", "sensory_ascending_tbc",
    "sensory_descending", "visual_projection", "visual_projection_tbc",
    "visual_centrifugal", "efferent_ascending", "efferent_descending",
}


@dataclass
class Connectome:
    """Directed weighted graph. W[i, j] = synapse count from neuron i to neuron j."""
    W: sp.csr_matrix
    sign: np.ndarray                      # per-neuron presynaptic sign in {-1,0,+1}
    region: np.ndarray                    # per-neuron region / neuropil label
    celltype: np.ndarray                  # per-neuron cell type label
    name: str = "unnamed"
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.W.shape[0]

    @property
    def n_edges(self):
        return self.W.nnz

    def signed(self):
        """Signed weight matrix: rows scaled by presynaptic sign."""
        return sp.diags(self.sign.astype(np.float64)) @ self.W

    def subset(self, mask):
        """Induced subgraph on a boolean mask."""
        idx = np.flatnonzero(mask)
        return Connectome(
            W=self.W[idx][:, idx].tocsr(),
            sign=self.sign[idx], region=self.region[idx], celltype=self.celltype[idx],
            name=f"{self.name}[subset:{len(idx)}]", meta=dict(self.meta),
        )

    def summary(self):
        W = self.W
        din = np.asarray((W > 0).sum(0)).ravel()
        dout = np.asarray((W > 0).sum(1)).ravel()
        return {
            "name": self.name, "n_neurons": self.n, "n_edges": self.n_edges,
            "density": self.n_edges / max(self.n * (self.n - 1), 1),
            "mean_degree": float(dout.mean()),
            "frac_inhibitory": float((self.sign < 0).mean()),
            "median_synapses_per_edge": float(np.median(W.data)),
            "degree_cv_out": float(dout.std() / max(dout.mean(), 1e-9)),
            "degree_cv_in": float(din.std() / max(din.mean(), 1e-9)),
        }


# --------------------------------------------------------------------------
# Real data — MaleCNS v1.0 flat connectome
# --------------------------------------------------------------------------
def load_malecns_feather(root, keep_central_brain=True, min_weight=1,
                         verify=True):
    """
    Load the MaleCNS v1.0 flat connectome from the three released feather files.

        root/annotations.feather        bodyId, superclass, status, type, somaSide
        root/neurotransmitters.feather  body, consensus_nt
        root/edges.feather              body_pre, body_post, weight   (Arrow IPC, ~1.05 GB)

    Node policy follows the release convention: retain every row with a non-empty
    superclass (166,700 of 211,577 annotation rows), sorted by bodyId. Edges are
    streamed batch-by-batch from a memory map, so peak RAM stays near the size of
    the retained edge list rather than the 1 GB file.

    Schema and node policy verified against the DOOMFLY audit
    (github.com/nftechie/doomfly, MIT). Connectome data is CC-BY-4.0 — see README.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    from pathlib import Path

    root = Path(root)
    if verify:
        _verify_sources(root)

    raw = feather.read_table(root / "annotations.feather").to_pandas()
    keep = raw.superclass.notna() & raw.superclass.ne("")
    if keep_central_brain:
        keep &= raw.superclass.isin(CENTRAL_BRAIN_SUPERCLASSES)
    nodes = raw[keep].sort_values("bodyId")
    ids = nodes.bodyId.to_numpy()
    index = pd.Index(ids)

    nt = feather.read_table(root / "neurotransmitters.feather").to_pandas().set_index("body")
    consensus = nt.reindex(ids).consensus_nt.fillna("").str.lower().to_numpy()
    sign = np.array([NT_SIGN.get(str(x), +1) for x in consensus], np.int8)

    src = ipc.open_file(pa.memory_map(str(root / "edges.feather")))
    rows, cols, wts = [], [], []
    for k in range(src.num_record_batches):
        b = src.get_batch(k)
        pre, post, w = [b.column(b.schema.get_field_index(c)).to_numpy()
                        for c in ("body_pre", "body_post", "weight")]
        i, j = index.get_indexer(pre), index.get_indexer(post)
        ok = (i >= 0) & (j >= 0) & (i != j) & (w >= min_weight)
        rows.append(i[ok]); cols.append(j[ok]); wts.append(w[ok])

    r = np.concatenate(rows); c = np.concatenate(cols)
    w = np.concatenate(wts).astype(np.float64)
    n = len(ids)
    W = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()

    return Connectome(
        W=W, sign=sign,
        region=nodes.superclass.to_numpy().astype(object),
        celltype=nodes.type.fillna("").to_numpy().astype(object),
        name="malecns_v1" + ("[central_brain]" if keep_central_brain else ""),
        meta={"body_ids": ids, "min_weight": min_weight,
              "central_brain_only": keep_central_brain,
              "soma_side": nodes.somaSide.fillna("").to_numpy()},
    )


def _verify_sources(root, lock="malecns_source.lock.json"):
    """Check sizes and SHA-256 of the three source files against the release lock."""
    import hashlib, json
    from pathlib import Path
    lockfile = Path(lock)
    if not lockfile.exists():
        print("  (no lock file; skipping integrity check)")
        return
    spec = json.loads(lockfile.read_text())
    for name, info in spec.items():
        p = Path(root) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Download: {info['url']}")
        if p.stat().st_size != info["bytes"]:
            raise ValueError(f"{name}: size {p.stat().st_size} != {info['bytes']}")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for blk in iter(lambda: f.read(8 << 20), b""):
                h.update(blk)
        if h.hexdigest() != info["sha256"]:
            raise ValueError(f"{name}: sha256 mismatch")
    print("  source files verified against release lock")


def load_neuprint_csv(neurons_csv, connections_csv, min_weight=3,
                      drop_regions=("OL", "ME", "LO", "LOP", "VNC", "LA")):
    """
    Fallback loader for neuPrint CSV exports (hemibrain, FlyWire, or a
    neuprint-python query). Prefer load_malecns_feather for MaleCNS v1.0 --
    its schema is verified, this one guesses at column names.
    """
    import pandas as pd
    nrn = pd.read_csv(neurons_csv)
    con = pd.read_csv(connections_csv)

    def col(df, *names):
        for c in names:
            if c in df.columns:
                return df[c]
        raise KeyError(f"none of {names} in {list(df.columns)[:12]}")

    body = col(nrn, "bodyId", "bodyid", "id").to_numpy()
    nt = col(nrn, "predictedNt", "nt", "neurotransmitter").fillna("unknown").str.lower().to_numpy()
    reg = col(nrn, "somaNeuromere", "region", "roi").fillna("NA").to_numpy()
    typ = col(nrn, "type", "celltype").fillna("NA").to_numpy()

    keep = np.ones(len(body), bool)
    if drop_regions:
        keep &= ~np.isin([str(r).upper() for r in reg], [d.upper() for d in drop_regions])
    body, nt, reg, typ = body[keep], nt[keep], reg[keep], typ[keep]

    index = {b: i for i, b in enumerate(body)}
    pre = col(con, "bodyId_pre", "pre", "source").to_numpy()
    post = col(con, "bodyId_post", "post", "target").to_numpy()
    w = col(con, "weight", "synapses", "count").to_numpy().astype(np.float64)

    ok = np.array([p in index and q in index for p, q in zip(pre, post)]) & (w >= min_weight)
    r = np.array([index[p] for p in pre[ok]])
    c = np.array([index[q] for q in post[ok]])

    n = len(body)
    W = sp.coo_matrix((w[ok], (r, c)), shape=(n, n)).tocsr()
    W.setdiag(0); W.eliminate_zeros()
    sign = np.array([NT_SIGN.get(str(x), +1) for x in nt], np.int8)
    return Connectome(W=W, sign=sign, region=np.array(reg, object),
                      celltype=np.array(typ, object), name="neuprint",
                      meta={"min_weight": min_weight, "dropped": drop_regions})


# --------------------------------------------------------------------------
# Synthetic stand-in
# --------------------------------------------------------------------------
def synthetic_flylike(n=6000, n_modules=12, mean_degree=45, p_in=0.80,
                      frac_inhibitory=0.28, seed=0):
    """
    Synthetic graph matched to published central-brain statistics:
      - modular (neuropil-like communities), most edges within module
      - log-normal synapse counts per edge (heavy tailed, as observed)
      - log-normal out-degree hub structure
      - ~28% inhibitory presynaptic neurons (GABA + Glu share)

    IMPORTANT: this is a NULL-BY-CONSTRUCTION control. Its topology carries no
    information beyond its own statistics, so any probe that "finds structure"
    here is reporting a bug in the null models, not a discovery.
    """
    rng = np.random.default_rng(seed)
    module = rng.integers(0, n_modules, n)
    propensity = np.exp(rng.normal(0, 0.9, n))
    propensity /= propensity.mean()

    target_edges = int(n * mean_degree)
    rows, cols = [], []
    by_mod = [np.flatnonzero(module == m) for m in range(n_modules)]
    for m in range(n_modules):
        members = by_mod[m]
        k_in = int(target_edges * len(members) / n * p_in)
        p = propensity[members] / propensity[members].sum()
        rows.append(rng.choice(members, k_in, p=p))
        cols.append(rng.choice(members, k_in))
    k_out = target_edges - sum(len(r) for r in rows)
    rows.append(rng.choice(n, k_out, p=propensity / propensity.sum()))
    cols.append(rng.choice(n, k_out))

    r = np.concatenate(rows); c = np.concatenate(cols)
    ok = r != c
    r, c = r[ok], c[ok]
    w = np.round(np.exp(rng.normal(1.4, 1.0, len(r)))).clip(1, 2000)

    W = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()
    sign = np.where(rng.random(n) < frac_inhibitory, -1, 1).astype(np.int8)
    region = np.array([f"MOD{m:02d}" for m in module], object)
    return Connectome(W=W, sign=sign, region=region,
                      celltype=np.array(["syn"] * n, object),
                      name=f"synthetic_flylike(seed={seed})",
                      meta={"n_modules": n_modules, "synthetic": True})


def mushroom_body_layer(cx, pn_region=None, kc_region=None, n_pn=200, n_kc=2500,
                        claws=6, seed=0):
    """
    Extract (or synthesise) the PN -> KC bipartite projection: the sparse
    expansion motif that is the most directly transferable piece of fly
    architecture for a sequence task.

    With a real Connectome, pass pn_region/kc_region (e.g. "CA" / "MB") and the
    actual bipartite block is pulled out. Otherwise a claw-matched synthetic
    block is generated.
    Returns a dense {0,1} matrix of shape (n_pn, n_kc).
    """
    if cx is not None and pn_region is not None and kc_region is not None:
        pn = np.flatnonzero(cx.region == pn_region)
        kc = np.flatnonzero(cx.region == kc_region)
        if len(pn) and len(kc):
            M = np.asarray((cx.W[pn][:, kc] > 0).todense(), np.float64)
            return M
    rng = np.random.default_rng(seed)
    M = np.zeros((n_pn, n_kc))
    for j in range(n_kc):
        M[rng.choice(n_pn, claws, replace=False), j] = 1.0
    return M
