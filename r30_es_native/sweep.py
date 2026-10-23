"""run a list of e1 configs in parallel worker processes, append results to a jsonl."""
import sys, json, os, itertools
from multiprocessing import Pool
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
def job(cfg):
    import importlib; mod = importlib.import_module(cfg.pop("_mod", "e1"))
    h = mod.run(log=False, **cfg)
    if isinstance(h, dict): return dict(cfg, **h)
    return dict(cfg, hist=h)
if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); out = sys.argv[2]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
        for r in pool.imap_unordered(job, cfgs):
            with open(out, "a") as f: f.write(json.dumps(r) + "\n")
            print(r["rule"], {k: v for k, v in r.items() if k not in ("hist", "rule", "feats")}, r["hist"][-1], flush=True)
