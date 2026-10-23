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
