"""E5b: evolve the STRUCTURE of the winning design (sparse net + hard lookup); weights learn by exact
forward-only-equivalent credit. Genome = input feature list, H, kw. Fitness = val bpc after a short run."""
import numpy as np, json, sys, os
from e5 import mutate, key
import jmem

def train_eval(g, steps=1500):
    h = jmem.run(mixer="hard", feats=[list(f) for f in g["feats"]], H=g["H"], kw=g["kw"], steps=steps, T=1024, nseq=1,
                 evals=1, log=False)
    return h[-1][3], h[-1][2]

if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1]; gens = int(sys.argv[2])
    rng = np.random.default_rng(1)
    seed_g = {"feats": [("c", i) for i in range(1, 9)], "H": 1024, "kw": 32}
    seen = {}
    def ev(gs):
        todo = [g for g in gs if key(g) not in seen]
        with Pool(2) as pool:
            for g, (v, sec) in zip(todo, pool.map(train_eval, todo)):
                seen[key(g)] = v
                with open(out, "a") as f: f.write(json.dumps(dict(genome=g, bpc=v, sec=sec)) + "\n")
                print(round(v, 4), round(sec), g, flush=True)
    pop = [seed_g] + [mutate(mutate(seed_g, rng), rng) for _ in range(3)]
    ev(pop)
    for gen in range(gens):
        pop = sorted(pop, key=lambda g: seen[key(g)])[:4]
        kids = []
        while len(kids) < 4:
            c = mutate(pop[int(rng.integers(len(pop)))], rng)
            if rng.random() < 0.5: c = mutate(c, rng)
            if key(c) not in seen and all(key(c) != key(k) for k in kids): kids.append(c)
        ev(kids); pop = pop + kids
        best = min(pop, key=lambda g: seen[key(g)])
        print("GEN", gen, "best", round(seen[key(best)], 4), best, flush=True)
