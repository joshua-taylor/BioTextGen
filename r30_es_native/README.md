# Round 30 — designing text models for evolution (forward-only) training

Char-level WikiText-2 experiments behind `claude/round30-es-native-architecture.md` in the Bio brain text gen project.
CPU-only, numpy + JAX.

## Setup
```bash
pip install numpy scipy jax optax
mkdir -p data && for f in train valid test; do
  curl -sSfL -o data/wt2_$f.txt https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$f.txt; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## What each file is
| File | Experiment |
|---|---|
| `e1.py` | Dense MLP LM: backprop vs per-token node perturbation vs classic (EGGROLL-style rank-1) ES vs readout-only; guided-subspace noise (`gnp`) |
| `e2.py`, `e2b.py` | Sparse k-WTA net in numpy; `e2b` adds K-draw perturbation and exact one-unit-at-a-time forward credit (`fdo`) |
| `fast.py`, `e2f.py` | Fast one-layer sparse net with lazy Adam; the perturbation-price sweep |
| `jsparse.py` | 2-layer sparse net in JAX (fixed-selection gradient = forward-only credit) |
| `jfdo.py` | Forward-only exact credit for `jsparse` (forward-mode probes over each token's cone); `python jfdo.py check` |
| `jmem.py` | Context mixers in the sparse backbone: none / hard own-code lookup / random-code lookup / soft attention / counterfactual re-ranking |
| `jfdo_mem.py` | Forward-only training of the hard-lookup models; `python -c "import jfdo_mem as J; J.check('hard')"` |
| `tfm.py` | Backprop transformer baseline |
| `e5.py`, `e5b.py` | Evolution of structure (input features, width, active units) with exact inner learning |
| `sweep.py` | Runs a JSON list of configs on 2 worker processes |
| `price.py` | Data price of random perturbation from the price sweep |
| `notes.md` | Working notes with every number |

`results/` holds the raw `res_*.jsonl` and logs.

## Round 31 additions (fast bench)
| File | What |
|---|---|
| `sbdata.py` | precomputed feature pool (run once: `python sbdata.py`; long validation stream built in round 31 notes) |
| `sb.py`, `sb2.py` | fast numpy+numba sparse + hard-lookup model with exact local credit; options for swap trials, neurogenesis, homeostasis, boosting, exploration, Hebbian learning, novelty-gated memory, keyed LSH memories, pair index, corpus/long-document evaluation |
| `pairmem.py` | pair-of-top-units index for long memories |
| `evo2.py` | structure evolution over the full design space |
| `e_prefill.py`, `e_long_fixed.py` | corpus-memory and fixed-window long-document memory tests |
| `check_sb2.py` | finite-difference check of every update path |
| `round31.md` | write-up |

Design D (recommended): `sb2.run(feats=EVO+[["ng",4]], keys=[[["wd"],["pw"]], [["c",i] for i in range(1,13)]], topm=2, lm={"O":2.0}, horizon=32768, cap=512)`.
