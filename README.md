# flybrain — a connectome-to-text-model test harness

Tests whether fly connectome structure is useful for next-token prediction, and
separates "useful" from "explained by its own statistics".

**The connectome itself could not be downloaded here** — it lives on a Google
Cloud bucket the sandbox proxy blocks (`host_not_allowed`), and the edge file is
1.05 GB. What *was* recovered: the verified file schema, the release node policy,
source URLs with SHA-256 digests, and a real per-neuron activity vector over all
166,700 MaleCNS neurons. The graph code is validated against a synthetic
fly-like stand-in; the real-data path is `--malecns <dir>`.

---

## The question, stated so it can fail

Not "can a connectome-constrained network learn text?" — it can, any sparse
random graph can. The question is:

> Does the **specific wiring** beat a random graph with the **same statistics**?

If it doesn't, the transferable finding is the statistics (degree distribution,
E/I ratio, sparsity, expansion ratio), and you should generate architectures
from those rather than copying 166,000 neurons.

## Pre-registered kill criteria

| # | Claim | Criterion | Fires if |
|---|-------|-----------|----------|
| 1 | Wiring carries information beyond claw counts | `mb_connectome` beats `mb_degree_null` by ≥0.02 bits/char, paired, 95% CI excluding 0, ≥5 seeds | not met → **wiring is not the asset; statistics are** |
| 2 | Sparse expansion helps at all | `mb_connectome` beats `no_expansion` by ≥0.01 bits/char | not met → the whole MB motif is dead for this task |

Criterion 2 is **expansion-ratio dependent** — measured, not assumed. At
`n_kc=2000` (5% active → 100 effective dims vs 200 PNs) expansion wins by 0.125
bits/char. At `n_kc=600` (30 effective dims) it *loses* by 0.108. Sweep `n_kc`
before reading anything into a single setting.

A structural property counts as real only if `|z| > 2` **and** the effect is
>20% of the null mean. A z of 300 on a 0.5% difference is a tight null, not a
finding — `run_all.py` enforces both.

---

## Quick start

```bash
pip install -r requirements.txt
python make_corpus.py              # builds corpus.txt (WikiText if reachable)
python run_all.py --seeds 5        # synthetic stand-in; validates the harness
```

## Layout

| file | what it does |
|---|---|
| `connectome.py` | `Connectome` container, neuPrint CSV loader, synthetic fly-like generator, MB bipartite extractor |
| `nulls.py` | five matched-statistics nulls, each stripping one property |
| `motifs.py` | structural census: reciprocity, modularity, 16-class directed triad census, spectral radius, effective rank — all as z-scores vs nulls |
| `mb_probe.py` | mushroom-body sparse-expansion probe on char-level text, closed-form ridge readout, no backprop |
| `make_corpus.py` | builds `corpus.txt` from WikiText / Kaggle inputs / local fallback |
| `run_all.py` | orchestration + verdict |

## Nulls — what each one isolates

| null | preserves | a win against it means |
|---|---|---|
| `weight_shuffle` | topology | synapse counts are placed informatively |
| `sign_shuffle` | wiring | E/I identity is placed informatively |
| `degree_preserving` | exact in/out degree sequence | **topology beats its own degree sequence** (the one that matters) |
| `configuration` | degrees in expectation | weaker version of the above |
| `erdos_renyi` | density only | anything at all is structured |

## The probe

```
context (8 chars, one-hot)
  → AL   fixed random projection to 200 "PNs" + divisive normalisation
  → KC   sparse mask, ~6 claws per KC, 2000 KCs
  → APL  winner-take-all, top 5% active
  → MBON ridge regression to next-char logits   (closed form, no gradients)
```

Conditions, each differing from the reference by exactly one thing:
`mb_connectome`, `mb_degree_null`, `mb_dense_proj`, `mb_no_wta`, `no_expansion`.

Readout is ridge + a 1-D temperature calibration on validation, so bits/char is
a real probability score rather than a regression residual.

---

## Loading MaleCNS v1.0

The connectome is **not** vendored in any repo — it is three files on a Google
Cloud bucket totalling ~1.1 GB. URLs and SHA-256 digests are in
`malecns_source.lock.json`. On a machine with network:

```bash
mkdir -p malecns_v1 && cd malecns_v1
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -O $B/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -O $B/body-neurotransmitters-male-cns-v1.0.feather
curl -O $B/connectome-weights-male-cns-v1.0-minconf-0.5.feather
# rename to annotations.feather / neurotransmitters.feather / edges.feather
```

Then:

```bash
python run_all.py --malecns malecns_v1 --seeds 5
```

`load_malecns_feather` verifies sizes and SHA-256 against the lock before
reading, streams the 1.05 GB edge file batch-by-batch from a memory map, and
filters to the central brain by **superclass** — the release's own annotation —
rather than by guessing at neuropil strings.

### Schema (verified, not guessed)

| file | columns used |
|---|---|
| `annotations.feather` | `bodyId`, `superclass`, `status`, `type`, `somaSide` |
| `neurotransmitters.feather` | `body`, `consensus_nt` |
| `edges.feather` | `body_pre`, `body_post`, `weight` (Arrow IPC record batches) |

Node policy: retain every row with a non-empty `superclass` — 166,700 of 211,577
annotation rows — sorted by `bodyId`. `keep_central_brain=True` then drops the
optic lobes (89,403 `ol_intrinsic` + 6,098 `ol_sensory`) and the entire VNC,
leaving roughly 50k neurons.

**Sign convention.** GABA, glutamate **and histamine** are inhibitory; ACh and
everything else excitatory. Histamine matters — 7,891 MaleCNS neurons are
histaminergic and an earlier version of this loader had them as excitatory.
Modulators (dopamine 392, octopamine 101, serotonin 48) are signed 0 here; they
should be gated rather than summed once you move to a three-factor rule.

Release NT counts, for sanity-checking your load: acetylcholine 103,720 ·
glutamate 29,302 · gaba 22,069 · histamine 7,891 · unclear 2,999 ·
dopamine 392 · missing 178 · octopamine 101 · serotonin 48.

## Empirical activity calibration

`malecns_probe_counts.npy` is a per-neuron activation count over all 166,700
retained neurons, aligned to the release node ordering. It is the only real fly
data usable offline here, and it settles a parameter the probe would otherwise
guess:

```
active fraction            5.15%
top 1% of neurons carry    37% of all activity
mean count (active only)   303
```

5.15% is strikingly close to the textbook ~5% Kenyon cell sparsity, so the APL
winner-take-all setting in `mb_probe.py` is empirically anchored rather than
assumed. `python calibrate.py` prints the stats; `calibrate.by_superclass`
splits them once you have the annotations loaded.

The heavy tail is the more interesting number. A third of all activity sits in
1% of neurons — that is a hub structure your synthetic architectures should
reproduce, and it is testable against the degree-preserving null.

## Scaling notes

Developed on 1 CPU / 3 GB. `degree_preserving` is a Python swap loop at
~10 swaps/edge; on the real central brain (several million edges) drop to
`n_swaps_per_edge=3` or move it to `igraph.rewire`. The probe is dominated by a
`n_kc × n_kc` Gram solve — fine to 8k KCs on CPU, above that use your T4.

## What to run next, in order

1. Structural census on the real central brain. Read off which properties
   survive the degree-preserving null with a >20% effect. Those are your
   architectural priors.
2. Probe with the real CA→MB block. Expect kill criterion 1 to fire. That is a
   publishable negative if it's clean and multi-seed.
3. If 1 fires, regenerate synthetic graphs *matched to the surviving properties
   only* and check they recover the same probe performance — that confirms the
   statistics are sufficient and you can stop copying wiring.
4. Only then move to dynamics: replace ridge with a dopamine-gated three-factor
   local rule and see whether the topology matters more once learning is local.

---

## Attribution

- **MaleCNS v1.0 connectome** — CC-BY-4.0. The MaleCNS collaboration: FlyEM at
  HHMI Janelia, University of Cambridge Department of Zoology, MRC Laboratory of
  Molecular Biology, and Google Research.
  Release: https://male-cns.janelia.org/download/ ·
  Paper: https://doi.org/10.1016/j.cell.2026.08.015
- **Schema verification and derived probe counts** — adapted from DOOMFLY
  (github.com/nftechie/doomfly), MIT licence. Their `doom/audit_data.py` is the
  source of the node policy and column schema used here; the CC-BY terms on the
  underlying connectome data are unaffected by that MIT licence.
