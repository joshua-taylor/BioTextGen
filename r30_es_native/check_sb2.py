"""Finite-difference check of sb2's local credit (keys, lookup, 2 layers): perturb single weights, re-run the forward
pass on copied memories, compare the measured loss change with the update the model would apply."""
import copy, numpy as np, sb2
K = [[["wd"], ["pw"]], [["c", i] for i in range(1, 13)]]
EVO = [["c", i] for i in range(1, 9)] + [["ex", 11], ["ex", 3], ["wd"], ["pw"]]
r = sb2.run(steps=150, evals=1, feats=EVO, keys=K, horizon=4096, cap=256, return_model=1)
m = r["model"]; V = m.V
s = sb2.sbdata.get("trn", "s"); pos = np.arange(200000, 200512); y = np.asarray(s[pos + 1]).astype(np.int64)
F = m.rows("trn", pos)
mems = [m.new_mems()]
for st in range(200000 - 4096, 200000, 512):                       # give the memory some history
    sb2.write_only(m, mems[0], "trn", np.arange(st, st + 512))
def L():
    mm = copy.deepcopy(mems); o = m.forward(F, mm, y, 1); return m.loss(o, y).mean()
m.c["grad_only"] = 1; m.step(F, y, copy.deepcopy(mems), y, 1); g = m.last_grads; m.c["grad_only"] = 0
rng = np.random.default_rng(0); eps = 1e-3
for k in ["W1", "W2", "R2", "E", "M", "MK0", "MK1", "Wo"]:
    gk = g[k]
    if isinstance(gk, tuple):
        rows, gr = gk; cand = [(rows[i], j) for i in range(len(rows)) for j in range(gr.shape[1]) if abs(gr[i, j]) > 1e-5]
        lut = {(rows[i], j): gr[i, j] for i in range(len(rows)) for j in range(gr.shape[1])}
    else:
        nz = np.argwhere(np.abs(gk) > 1e-5); cand = [tuple(x) for x in nz]; lut = None
    pick = [cand[i] for i in rng.choice(len(cand), min(6, len(cand)), replace=False)]
    errs = []
    for idx in pick:
        P = m.p[k]; old = P[idx]
        P[idx] = old + eps; lp = L(); P[idx] = old - eps; lm = L(); P[idx] = old
        fd = (lp - lm) / (2 * eps); an = lut[idx] if lut else gk[idx]
        errs.append((round(float(fd), 7), round(float(an), 7)))
    print(k, errs)
