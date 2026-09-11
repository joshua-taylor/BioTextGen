"""
Empirical activity calibration from real MaleCNS neurons.

`malecns_probe_counts.npy` is a per-neuron activation count vector over all
166,700 retained MaleCNS v1.0 neurons, aligned to the release node ordering
(annotation rows with a non-empty superclass, sorted by bodyId). It comes from
the DOOMFLY project's sensory-driven simulation runs.

This is the one piece of REAL fly data available offline here. It does not give
us wiring, but it does give an empirical answer to a question the probe has to
guess at: how sparse is activity in an actual fly brain under sensory drive?

Use it to set the APL winner-take-all fraction from data rather than from the
textbook "~5% of Kenyon cells" figure.

Data: MaleCNS v1.0 (CC-BY-4.0), FlyEM/HHMI Janelia, Cambridge Zoology,
MRC LMB and Google Research. Derived counts via DOOMFLY (MIT).
"""
import numpy as np
from pathlib import Path

N_RETAINED = 166700


def load_probe_counts(path="malecns_probe_counts.npy"):
    v = np.load(path)
    if v.shape[0] != N_RETAINED:
        raise ValueError(f"expected {N_RETAINED} neurons, got {v.shape[0]}")
    return v


def activity_stats(v):
    order = np.argsort(v)
    top1 = order[-int(0.01 * len(v)):]
    active = v > 0
    return {
        "n_neurons": int(len(v)),
        "active_fraction": float(active.mean()),
        "mean_count": float(v.mean()),
        "p99_count": float(np.percentile(v, 99)),
        "max_count": int(v.max()),
        "top1pct_share_of_activity": float(v[top1].sum() / max(v.sum(), 1)),
        "active_mean_count": float(v[active].mean()) if active.any() else 0.0,
    }


def suggested_sparsity(v, floor=0.01, ceil=0.20):
    """APL winner-take-all fraction, taken from the observed active fraction."""
    return float(np.clip(activity_stats(v)["active_fraction"], floor, ceil))


def by_superclass(v, cx):
    """
    Split the activity vector by superclass. Requires a Connectome loaded with
    keep_central_brain=False (so indices align with the full 166,700 ordering)
    and carrying meta['body_ids'].
    """
    if cx.n != N_RETAINED:
        raise ValueError("load with keep_central_brain=False to align indices")
    out = {}
    for sc in np.unique(cx.region):
        m = cx.region == sc
        out[str(sc)] = {"n": int(m.sum()),
                        "active_fraction": float((v[m] > 0).mean()),
                        "mean_count": float(v[m].mean())}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


if __name__ == "__main__":
    import json
    p = Path("malecns_probe_counts.npy")
    if not p.exists():
        raise SystemExit("malecns_probe_counts.npy not found")
    v = load_probe_counts(p)
    print(json.dumps(activity_stats(v), indent=2))
    print(f"\nsuggested APL sparsity = {suggested_sparsity(v):.4f}")
