"""train the best evolved genome (from res_e5b.jsonl) on the full training set, same budget as hard-full."""
import json, jmem
rows = [json.loads(l) for l in open("res_e5b.jsonl")]
best = min(rows, key=lambda r: r["bpc"])["genome"]
print("best genome", best, flush=True)
h = jmem.run(mixer="hard", feats=[list(f) for f in best["feats"]], H=best["H"], kw=best["kw"], steps=5000, T=1024,
             nseq=1, evals=5, log=True, ntrain=None)
open("res_full.jsonl", "a").write(json.dumps(dict(rule="evolved-hard-full", genome=best, hist=h)) + "\n")
