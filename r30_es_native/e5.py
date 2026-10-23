"""E5: let evolution choose the STRUCTURE; exact forward-only credit fills the WEIGHTS.

Genome = which context features feed the sparse layer (each feature indexes its own block of table rows),
plus layer width H and active count kw.  Every candidate is trained briefly (sparse k-WTA net, L=1, exact
per-unit credit) and scored on validation bits/char. (mu + lambda) evolution with add/remove/tweak mutations.
"""
import numpy as np, json, sys, os, time
from common import load, xent, logsoftmax, Adam, LN2
from fast import Sparse1

R = 1 << 15
P1, P2 = 1000003, 2654435761

def block_size(f, V):
    t = f[0]
    if t == "c": return V
    if t == "ex": return V + 1
    return R

def feat_col(f, seq, T, V, cache):
    """row ids (T,) within the feature's own block, for targets seq[1..T] given seq[..t]."""
    t = f[0]
    pad = 16
    s = np.concatenate([np.zeros(pad, np.int64), seq])        # s[pad + i] = seq[i]
    pos = np.arange(T) + pad                                   # index of seq[t] (last known char) in s
    if t == "c":
        return s[pos - (f[1] - 1)]
    if t == "ng":
        h = np.zeros(T, np.int64)
        for o in range(f[1]): h = (h * P1 + s[pos - o] + 1) % (1 << 61)
        return (h * P2 + 7 * f[1]) % R
    if t == "sk":
        a, b = f[1], f[2]
        return ((s[pos - (a - 1)] * 131 + s[pos - (b - 1)] + 977 * (a * 17 + b)) * P2) % R
    if t in ("wd", "pw"):
        key = ("words",)
        if key not in cache:
            sp = cache["space"]
            wd = np.zeros(T, np.int64); pw = np.zeros(T, np.int64); cur = 0; prev = 0
            for i in range(T):
                c = seq[i]
                if c == sp: prev, cur = cur, 0
                else: cur = (cur * 1000003 + int(c) + 1) % 2305843009213693951
                wd[i] = cur; pw[i] = prev
            cache[key] = (wd, pw)
        wd, pw = cache[key]
        return ((wd if t == "wd" else pw) * P2 + (11 if t == "wd" else 13)) % R
    if t == "ex":
        n = f[1]; last = {}; r = np.full(T, V, np.int64)
        for i in range(T):
            if i + 1 >= n:
                k = seq[i + 1 - n:i + 1].tobytes(); pv = last.get(k)
                if pv is not None: r[i] = seq[pv + 1]
                last[k] = i
        return r
    raise ValueError(f)

def build(genome, stream, starts, T, V, space):
    feats = genome["feats"]; offs = np.cumsum([0] + [block_size(f, V) for f in feats])
    Fs, ys = [], []
    for st in starts:
        seq = stream[st:st + T + 1]; cache = {"space": space}
        Fs.append(np.stack([offs[j] + feat_col(f, seq, T, V, cache) for j, f in enumerate(feats)], 1)); ys.append(seq[1:])
    return np.concatenate(Fs), np.concatenate(ys), int(offs[-1])

def train_eval(genome, steps=1500, T=256, nseq=4, lr=3e-3, seed=0, ntrain=2_000_000, nval=60_000):
    trn, val, itos = load(ntrain); V = len(itos); space = itos.index(' ')
    rng = np.random.default_rng(seed); H, kw = genome["H"], genome["kw"]
    nrows = sum(block_size(f, V) for f in genome["feats"])
    net = Sparse1(nrows, H, kw, V, lr=lr, seed=seed)
    t0 = time.time()
    for it in range(steps):
        F, y, _ = build(genome, trn, rng.integers(0, len(trn) - T - 2, nseq), T, V, space)
        net.step(F, y)
    tot = cnt = 0.0
    for s0 in range(0, nval - T - 1, T):
        F, y, _ = build(genome, val, [s0], T, V, space)
        tot += net.loss(F, y).sum(); cnt += len(y)
    return tot / cnt / LN2, time.time() - t0

FEAT_POOL = lambda rng: [
    ("c", int(rng.integers(1, 13))),
    ("ng", int(rng.integers(2, 9))),
    ("sk", *sorted(map(int, rng.choice(np.arange(1, 13), 2, replace=False)))),
    ("wd",), ("pw",),
    ("ex", int(rng.choice([3, 4, 5, 6, 8, 10, 12, 16]))),
]

def mutate(g, rng):
    g = {"feats": [tuple(f) for f in g["feats"]], "H": g["H"], "kw": g["kw"]}
    r = rng.random()
    if r < 0.4 and len(g["feats"]) < 16:
        pool = FEAT_POOL(rng); f = pool[rng.integers(len(pool))]
        if f not in g["feats"]: g["feats"].append(f)
    elif r < 0.6 and len(g["feats"]) > 2:
        g["feats"].pop(int(rng.integers(len(g["feats"]))))
    elif r < 0.85:
        i = int(rng.integers(len(g["feats"]))); f = list(g["feats"][i])
        if len(f) > 1:
            j = int(rng.integers(1, len(f))); f[j] = int(max(1, min(16, f[j] + rng.choice([-2, -1, 1, 2]))))
            if f[0] == "sk" and f[1] == f[2]: f[2] += 1
            g["feats"][i] = tuple(f)
    else:
        if rng.random() < 0.5: g["H"] = int(rng.choice([512, 1024, 2048]))
        else: g["kw"] = int(rng.choice([8, 16, 32]))
    return g

def key(g): return json.dumps([sorted(map(list, g["feats"])), g["H"], g["kw"]])

if __name__ == "__main__":
    from multiprocessing import Pool
    os.environ["OMP_NUM_THREADS"] = "1"
    out = sys.argv[1] if len(sys.argv) > 1 else "res_e5.jsonl"
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rng = np.random.default_rng(0)
    seed_g = {"feats": [("c", i) for i in range(1, 9)], "H": 1024, "kw": 16}   # = the hand-made e2 input
    seen = {}
    def ev(gs):
        todo = [g for g in gs if key(g) not in seen]
        with Pool(2) as pool:
            for g, (v, sec) in zip(todo, pool.map(train_eval, todo)):
                seen[key(g)] = v
                with open(out, "a") as f: f.write(json.dumps(dict(genome=g, bpc=v, sec=round(sec, 1))) + "\n")
                print(round(v, 4), round(sec), g, flush=True)
    pop = [seed_g] + [mutate(mutate(seed_g, rng), rng) for _ in range(3)]
    ev(pop)
    for gen in range(gens):
        pop = sorted(pop, key=lambda g: seen[key(g)])[:4]
        kids = []
        while len(kids) < 4:
            par = pop[int(rng.integers(len(pop)))]; c = mutate(par, rng)
            if rng.random() < 0.5: c = mutate(c, rng)
            if key(c) not in seen and all(key(c) != key(k) for k in kids): kids.append(c)
        ev(kids); pop = pop + kids
        best = min(pop, key=lambda g: seen[key(g)])
        print("GEN", gen, "best", round(seen[key(best)], 4), best, flush=True)
