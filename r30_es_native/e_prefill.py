"""Corpus memory: train once, then evaluate with the lookup memory pre-filled with the model's own codes for
N chars of training text (read-only inference, no learning). Attention would need N-token context for this."""
import json, sys, time, numpy as np
import sb2

EVO = [["c", i] for i in range(1, 9)] + [["ex", 11], ["ex", 3], ["wd"], ["pw"]]

def main(out, horizon, cap, tag):
    c = dict(sb2.DEFAULT); c.update(feats=EVO, horizon=horizon, cap=cap, steps=800, evals=1)
    V, _ = sb2.sbdata.meta(); model = sb2.Model(c, V)
    # train (same loop as sb2.run)
    s = sb2.sbdata.get("trn", "s"); N = len(s); rng = np.random.default_rng(1000)
    starts = rng.integers(16, N // 2, c["S"])            # train on the first half; prefill from the end
    mems = [model.new_mems() for _ in range(c["S"])]
    for it in range(1, c["steps"] + 1):
        model.it = it
        pos = np.concatenate([np.arange(st + (it - 1) * c["T"], st + it * c["T"]) for st in starts])
        F, _ = sb2.sbdata.feature_rows(model.feats, "trn", pos, V)
        y = np.asarray(s[pos + 1]).astype(np.int64)
        model.step(F, y, mems, y, c["S"])
    for pf, pc in [(0, cap), (100_000, 512), (500_000, 1024), (2_000_000, 2048)]:
        model.c["prefill_cap"] = pc
        t = time.time(); v = sb2.evaluate(model, prefill=pf)
        r = dict(tag=tag, train_horizon=horizon, prefill=pf, cap=pc, bpc=round(v[0], 4), copy=round(v[1], 4), other=round(v[2], 4),
                 sec=round(time.time() - t, 1))
        print(json.dumps(r), flush=True); open(out, "a").write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
