"""Structure evolution over the full design space of the sparse + hard-lookup model (sb2).

Genome: input features, width H, active count kw, depth L (1-2), embedding size de, lookup settings (matches
returned, horizon, rich value, second lookup), homeostasis, learning rate and per-group learning-rate multipliers.
Every candidate's weights are learned by exact forward-only credit for a short run; fitness = validation bits/char.
All candidates use the same seed and data order (common random numbers), so differences are less noisy.
(mu + lambda) selection; mutations change 1-2 genes; a log line per evaluation.
"""
import numpy as np, json, sys, os, copy
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

EX = [2, 3, 4, 5, 6, 8, 11, 12, 16]
CHOICES = dict(H=[512, 1024, 2048], kw=[16, 24, 32, 48, 64], L=[1, 2], de=[32, 64, 96], topm=[1, 2, 4],
               horizon=[1024, 8192, 32768, 131072], lk2=[0, 1],
               lr=[1.5e-3, 3e-3, 5e-3], lmE=[0.5, 1.0, 2.0], lmW=[0.5, 1.0, 2.0], lmO=[0.5, 1.0, 2.0],
               mem_kind=["unit", "pair"], P=[6, 8, 12], bcap=[32, 64], write_thr=[2.0, 0.95, 0.9], boost=[0.0, 0.3])

def rand_feat(rng):
    t = rng.choice(["c", "ng", "sk", "wd", "pw", "ex"], p=[0.25, 0.2, 0.15, 0.05, 0.05, 0.3])
    if t == "c": return ["c", int(rng.integers(1, 13))]
    if t == "ng": return ["ng", int(rng.integers(2, 9))]
    if t == "sk":
        a, b = sorted(map(int, rng.choice(np.arange(1, 13), 2, replace=False))); return ["sk", a, b]
    if t == "ex": return ["ex", int(rng.choice(EX))]
    return [t]

KEY_TEMPLATES = [[["wd"], ["pw"]], [["c", i] for i in range(1, 13)], [["ng", 4], ["ng", 6]], [["pw"]], [["c", i] for i in range(1, 7)]]

def mutate_keys(g, rng):
    ks = g.setdefault("keys", [])
    q = rng.random()
    if q < 0.4 and len(ks) < 2:
        t = copy.deepcopy(KEY_TEMPLATES[int(rng.integers(len(KEY_TEMPLATES)))])
        if t not in ks: ks.append(t)
    elif q < 0.6 and ks:
        ks.pop(int(rng.integers(len(ks))))
    elif ks:
        k = ks[int(rng.integers(len(ks)))]
        if rng.random() < 0.5 and len(k) < 14:
            f = rand_feat(rng)
            if f[0] != "ex" and f not in k: k.append(f)
        elif len(k) > 1:
            k.pop(int(rng.integers(len(k))))

def mutate(g, rng):
    g = copy.deepcopy(g)
    for _ in range(int(rng.integers(1, 3))):
        r = rng.random()
        if r < 0.15:
            mutate_keys(g, rng); continue
        if r < 0.5:   # feature edit
            q = rng.random()
            if q < 0.45 and len(g["feats"]) < 20:
                f = rand_feat(rng)
                if f not in g["feats"]: g["feats"].append(f)
            elif q < 0.75 and len(g["feats"]) > 3:
                g["feats"].pop(int(rng.integers(len(g["feats"]))))
            else:
                i = int(rng.integers(len(g["feats"]))); f = g["feats"][i]
                if f[0] in ("c", "ng"): f[1] = int(np.clip(f[1] + rng.choice([-1, 1]), 1 if f[0] == "c" else 2, 12 if f[0] == "c" else 8))
                elif f[0] == "ex": f[1] = int(rng.choice(EX))
                elif f[0] == "sk": g["feats"][i] = rand_feat(rng) if rng.random() < 0.5 else ["sk", f[1], min(12, f[2] + 1)]
        else:
            k = str(rng.choice(list(CHOICES))); g[k] = CHOICES[k][int(rng.integers(len(CHOICES[k])))]
    # de-duplicate features
    seen = []; [seen.append(f) for f in g["feats"] if f not in seen]; g["feats"] = seen
    return g

def to_cfg(g, steps):
    c = {k: v for k, v in g.items() if k not in ("lmE", "lmW", "lmO")}
    c["lm"] = {"E": g["lmE"], "W": g["lmW"], "O": g["lmO"]}
    c.update(steps=steps, evals=1, seed=0)
    if c["L"] == 1: c["rv"] = 0; c["lk2"] = 0
    c["cap"] = int(min(1024, max(64, 2 * c["horizon"] * c["kw"] / c["H"])))   # index every remembered position
    if c["mem_kind"] == "pair": c["topm"] = 1
    return c

def key(g): return json.dumps({k: (sorted(map(str, v)) if k in ("feats", "keys") else v) for k, v in sorted(g.items())})

STEPS = int(os.environ.get("EVO_STEPS", "300"))

def fitness(g):
    import sb2
    try:
        r = sb2.run(**to_cfg(g, STEPS)); h = r["hist"][-1]
        return dict(bpc=h[3], copy=h[4], other=h[5], sec=h[2], eff=r["eff_units"])
    except Exception as e:
        return dict(bpc=9.9, err=str(e)[:200])

if __name__ == "__main__":
    out = sys.argv[1]; gens = int(sys.argv[2]); mu = int(sys.argv[3]); lam = int(sys.argv[4])
    rng = np.random.default_rng(int(os.environ.get("EVO_SEED", "0")))
    EVO = [["c", i] for i in range(1, 9)] + [["ex", 11], ["ex", 3], ["wd"], ["pw"]]
    seed_g = dict(feats=EVO, H=1024, kw=32, L=2, de=64, topm=1, horizon=1024, rv=0, lk2=0, homeo=0.0, lr=3e-3,
                  lmE=1.0, lmW=1.0, lmO=1.0, mem_kind="unit", P=8, bcap=64, write_thr=2.0, boost=0.0, keys=[])
    if len(sys.argv) > 5: seed_g.update(json.loads(sys.argv[5]))
    seen = {}
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if "genome" in r: seen[key(r["genome"])] = r["fit"]["bpc"]
    def ev(gs):
        todo = [g for g in gs if key(g) not in seen]
        with Pool(2) as pool:
            for g, f in zip(todo, pool.map(fitness, todo)):
                seen[key(g)] = f["bpc"]
                with open(out, "a") as fh: fh.write(json.dumps(dict(genome=g, fit=f)) + "\n")
                print(round(f["bpc"], 4), f, json.dumps({k: v for k, v in g.items() if k != "feats"}), g["feats"], flush=True)
    pop = [seed_g] + [mutate(seed_g, rng) for _ in range(mu - 1)]
    ev(pop)
    for gen in range(gens):
        pop = sorted(pop, key=lambda g: seen[key(g)])[:mu]
        kids = []
        tries = 0
        while len(kids) < lam and tries < 200:
            tries += 1
            par = pop[int(rng.integers(len(pop)))]
            if rng.random() < 0.25 and len(pop) > 1:   # crossover of feature sets and settings
                other = pop[int(rng.integers(len(pop)))]
                c = copy.deepcopy(par)
                for k in list(CHOICES) + ["keys"]:
                    if rng.random() < 0.5: c[k] = copy.deepcopy(other.get(k, c.get(k)))
                fs = par["feats"] + [f for f in other["feats"] if f not in par["feats"]]
                c["feats"] = [f for f in fs if rng.random() < 0.7] or par["feats"]
                c = mutate(c, rng) if rng.random() < 0.5 else c
            else:
                c = mutate(par, rng)
            if key(c) not in seen and all(key(c) != key(k) for k in kids): kids.append(c)
        ev(kids); pop = pop + kids
        best = min(pop, key=lambda g: seen[key(g)])
        print("GEN", gen, "best", round(seen[key(best)], 4), json.dumps(best), flush=True)
