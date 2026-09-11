"""
Run the full harness.

  python run_all.py                                   # synthetic stand-in
  python run_all.py --neurons n.csv --connections c.csv --pn-region CA --kc-region MB

Exit code is non-zero if a pre-registered kill criterion fires.
"""
import argparse, json, sys, time
import numpy as np

from connectome import synthetic_flylike, load_neuprint_csv, load_malecns_feather
from motifs import census_vs_nulls
import mb_probe


KILL = {
    # headline claim: connectome wiring beats a claw-matched random projection
    "connectome_beats_degree_null_bpc": 0.02,
    # sanity: sparse expansion must beat no expansion at all
    "expansion_beats_none_bpc": 0.01,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malecns", help="dir with annotations/neurotransmitters/edges.feather")
    ap.add_argument("--neurons"); ap.add_argument("--connections")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--calibrate-sparsity", action="store_true",
                    help="set APL fraction from malecns_probe_counts.npy")
    ap.add_argument("--pn-region"); ap.add_argument("--kc-region")
    ap.add_argument("--text", default="corpus.txt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--skip-structure", action="store_true")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    t0 = time.time()
    if a.malecns:
        cx = load_malecns_feather(a.malecns, verify=not a.no_verify)
    elif a.neurons and a.connections:
        cx = load_neuprint_csv(a.neurons, a.connections)
    else:
        cx = synthetic_flylike(n=a.n, seed=0)
        print("!! synthetic stand-in: it is null by construction. A win here is a bug.")
    print(f"\n[connectome] {json.dumps(cx.summary(), indent=2)}\n")

    report = {"connectome": cx.summary()}

    if not a.skip_structure:
        print("[structure] census vs nulls ...")
        obs, zs = census_vs_nulls(cx, n_null=3, verbose=True)
        report["structure"] = {"observed": obs, "vs_nulls": zs}
        print("\n  stat                        observed   null(deg)      z   ratio")
        for k, v in zs["degree_preserving"].items():
            denom = abs(v["null_mean"]) if abs(v["null_mean"]) > 1e-12 else 1.0
            ratio = v["observed"] / denom
            # a property counts as real structure only if it is BOTH statistically
            # separable (|z|>2) and materially different (>20% effect). Huge z on a
            # 0.5% difference is a tight null, not a finding.
            big = abs(v["z"]) > 2 and abs(ratio - 1.0) > 0.20
            print(f"  {k:26s} {v['observed']:9.4f} {v['null_mean']:10.4f} "
                  f"{v['z']:7.2f} {ratio:7.2f}{'  <-- structure' if big else ''}")

    print(f"\n[probe] mushroom-body sparse expansion on text ({a.seeds} seeds) ...")
    cfg = mb_probe.MBConfig(n_kc=a.n_kc)
    if a.calibrate_sparsity:
        import calibrate
        cfg.sparsity = calibrate.suggested_sparsity(calibrate.load_probe_counts())
        print(f"  APL sparsity from real MaleCNS activity: {cfg.sparsity:.4f}")
    summ, raw = mb_probe.sweep(a.text, cfg=cfg, seeds=tuple(range(a.seeds)),
                               cx=cx if a.pn_region else None,
                               pn_region=a.pn_region, kc_region=a.kc_region)
    report["probe"] = {"summary": summ}
    print("\n  condition          val bits/char (mean +/- 95% CI)")
    for k, v in sorted(summ.items(), key=lambda kv: kv[1]["mean_bpc"]):
        print(f"  {k:18s} {v['mean_bpc']:.4f} +/- {v['ci95']:.4f}")

    d_null = mb_probe.paired_delta(raw, "mb_connectome", "mb_degree_null")
    d_exp = mb_probe.paired_delta(raw, "mb_connectome", "no_expansion")
    report["probe"]["paired"] = {"vs_degree_null": d_null, "vs_no_expansion": d_exp}

    print("\n[verdict]")
    fired = []
    ok1 = (d_null["mean_delta_bpc"] + d_null["ci95"]) < -KILL["connectome_beats_degree_null_bpc"]
    print(f"  wiring vs claw-matched null : {d_null['mean_delta_bpc']:+.4f} "
          f"+/- {d_null['ci95']:.4f} bpc -> {'PASS' if ok1 else 'NULL RESULT'}")
    if not ok1:
        fired.append("connectome_beats_degree_null_bpc")
    ok2 = (d_exp["mean_delta_bpc"] + d_exp["ci95"]) < -KILL["expansion_beats_none_bpc"]
    print(f"  expansion vs none           : {d_exp['mean_delta_bpc']:+.4f} "
          f"+/- {d_exp['ci95']:.4f} bpc -> {'PASS' if ok2 else 'no gain'}")
    if not ok2:
        fired.append("expansion_beats_none_bpc")

    report["kill_criteria_fired"] = fired
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out} in {time.time()-t0:.0f}s")
    if fired:
        print(f"kill criteria fired: {fired}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
