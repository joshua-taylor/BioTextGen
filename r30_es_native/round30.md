# Round 30: designing for evolution — the cone law, per-token populations, sparse nets with hard lookup

Continues `claude/experiment-register.md` (R01–R29) and the shortlist doc "BioTextGen — Beyond Attention" (idea 5, the wildcard: architectures designed for evolution strategies). Date: 23 Oct 2026. All runs on 2 CPU cores, char-level WikiText-2 (96-symbol vocabulary), validation bits/char on the first 60k validation chars.

**Exam question (Josh).**
1. If we no longer need backprop, can we design something better optimised for text generation than attention?
2. How do we control the search space and optimise for EGGROLL-style evolution, or find a more effective evolution approach?

**Answer.**
- **Q2 — controlling the search space.** The cost of perturbation learning is set by each prediction's *dependency cone*: the number of perturbed quantities a single loss term depends on, not the number of parameters.
  - EGGROLL perturbs whole networks, so its cone is every parameter. It learned nothing beyond a random-feature baseline here.
  - Perturbing each token separately makes the batch the population. Making layers sparse shrinks the cone to the switched-on units.
  - Once the cone is small, enumerate it: perturb each switched-on unit of each token, one at a time. That gives the exact gradient from forward passes alone, at about 2× backprop's compute.
  - Evolution in the strict sense (trial and selection) is then only needed for discrete choices: which structure to use, and which past position to retrieve.
- **Q1 — better than attention.** In this setting, yes. A hard lookup that retrieves the single earlier position whose sparse code best matches the current one has no learned parameters and no differentiable path. It beat soft attention trained by backprop in the same network, by 0.10 bits/char overall and 0.46 on copyable text at 1,024-char context. The full model also beat backprop transformers at equal data (best tuned transformer 2.055 bpc vs 1.825 for the hand design (1.4M params) and 1.689 for the evolved design (5.6M params, mostly hashed word rows), full training set, 5.1M chars seen).
- **Caveat that decides what comes next:** everything here is tiny (≤1.4M dense-equivalent parameters, 5.1M training chars seen, CPU). Transformers are known to need more data than local models, so the transformer gap is the least trustworthy number. The next step is a GPU scale test.

### The evolution ladder measured this round (Q2 in one table)

| Method | What gets perturbed | Population | Data price vs backprop (measured) |
|---|---|---|---|
| EGGROLL-style: one rank-1 weight perturbation per member, shared by the batch | all weights | N = 32 members | learned nothing beyond random features at equal compute |
| Per-token rank-1, input-aligned (= node perturbation) | each token's units | the batch itself | ~60–100× (dense, cone 250–1,150) |
| + sparse k-of-H layers | active units only | the batch | ~5× (cone 16), ~12× (cone 64) |
| + K draws per token | same | batch × K | ~2.4× at K = 4 (cone 16) |
| + enumerate the cone | each active unit, one at a time | batch × cone | **1× (exact)**, ~2× compute |
| Evolution proper: (4+4) selection over discrete structure | the design itself | 4 designs | −0.17 bits in 36 trials |

**On EGGROLL's low-rank trick.** A rank-1 perturbation a·bᵀ of a weight matrix changes a token's units by (x·b)·a. Only the component of b along that token's own input x has any effect, so a random b wastes variance. Choosing b = x is exactly node perturbation. Perturbing per token instead of per member turns the batch into the population at no extra forward passes.

## 1. The maths: the cone law

Loss L = Σ_b l_b, one term per predicted token. Perturb quantities ξ_j (weights or unit activations), antithetically, and estimate the credit of coordinate i using only the loss terms that depend on it:

```
signal          = Σ_{b ∈ C(i)} ∂l_b/∂ξ_i
noise variance  ≈ Σ_{b ∈ C(i)} Σ_{j ∈ D(b), j ≠ i} (∂l_b/∂ξ_j)²
```

C(i) = the loss terms coordinate i touches. D(b) = everything perturbed that loss term b depends on: its **cone**. The data price of perturbation learning relative to backprop therefore grows with |D(b)|. Parameters that are not perturbed for that token cost nothing.

| Scheme | Cone per prediction | Expected price |
|---|---|---|
| Classic ES / EGGROLL: one weight perturbation shared by all tokens | every perturbed parameter | ~ parameter count |
| Per-token population (batch = population), node perturbation | units on that token's computation | ~ width × depth |
| Attention | units at every earlier position | ~ width × depth × context (worst case) |
| Sparse k-of-H layers | active units only | ~ k × depth |
| Hard retrieval of m positions | m positions | ~ m |
| Enumerate the cone (one coordinate at a time) | — | exact credit; compute ~ cone × (cost to recompute downstream) |

Enumeration is cheap when the cone is small (sparse layers) or the downstream is linear (the readout).

## 2. Evidence for the cone law

### E1 — dense networks (MLP on the last 8 chars, embeddings 16/char, 5.1M chars seen)

| Width / depth | Backprop | Per-token node perturbation | Hidden layer never trained | Classic ES, N=32, equal compute |
|---|---|---|---|---|
| H32 L1 | 2.675 | 3.194 | 4.114 | 3.451 |
| H128 L1 | 2.381 | 3.176 | 3.466 | 3.482 |
| H128 L2 | 2.241 | 3.342 | — | — |
| H512 L1 | 2.317 | 3.128 | 3.457 | — |
| H512 L2 | 2.313 | 3.379 | — | — |

- Classic ES learns nothing beyond the readout at equal compute.
- Per-token perturbation learns, but at ~60–100× the data. Each prediction's cone is ~250–1,150 units.
- **Estimator check.** The per-token estimate converges to the true gradient: cosine 0.98 after 4,000 draws. A single draw at batch 256 has cosine 0.09.
- **Guided noise (negative).** Drawing the noise from a learned low-rank subspace, PCA of past credit, did not help: 3.33 vs 3.29 at 2,000 steps. Here the credit is not low-rank (vocabulary ≈ width).

### E2 — sparse cones (one-layer net, one-hot (position, char) inputs → H units, keep top kw)

Data price of random per-token perturbation, K draws per token, against exact credit on the same network. The price is the data multiple, from a power-law fit of the exact curve.

| Network | Cone | Exact | K=1 | K=4 | Price, K=1 / K=4 |
|---|---|---|---|---|---|
| H1024, 16 active | 16 | 2.103 | 2.446 | 2.272 | ~5× / ~2.4× |
| H1024, 64 active | 64 | 2.014 | 2.497 | — | ~12× / — |
| H128 dense | 128 | 2.323 | 2.761 | 2.628 | — / ~5× |

- **Price grows with the cone**, roughly cone/5 per draw, and K draws divide it by about K.
- **Exact enumeration matches backprop.** Perturbing each active unit of each token once gave 3.0952 bpc after 300 steps; backprop gave 3.0952. Wall-clock: 9.5 s vs 6.4 s.

### Forward-only exact credit on the full designs

- **Cone per token.** 64 input dims, plus 64 layer-2 input dims (embedding plus retrieved value), plus 32 + 32 active units. Each is probed with a forward-mode derivative of that token's own loss.
- **Weight updates are local.** Each update is (credit at a unit) × (that unit's input). The output error, softmax − one-hot, is local to the readout.
- **The retrieval scorer** (for the re-ranking variant) learns from counterfactual forward passes only.
- **Check against the backprop gradient:** cosine **1.000000**, norm ratio 1.0000, on every weight block, for:
  - the 2-layer net,
  - the net with hard lookup,
  - the net with the counterfactual re-ranker.
- **Training:** 200 steps gave 2.7129 forward-only vs 2.7159 backprop, at 2.3× the wall-clock (unoptimised).
- **So the architecture numbers below hold for backprop-free training.** For speed they were run with a gradient call, which on these fixed-selection networks is numerically identical.

## 3. Architecture results

### E6 — sparse vs dense backbone (5.1M chars; training stream = first 2M chars, so 2.5 epochs)

| Inputs | Sparse: 2 layers, 1,024 units, 32 active | Dense: 2 layers, 256 units |
|---|---|---|
| Last 8 chars | **1.982** | 2.122 |
| + hashed n-grams and word hashes | 1.909 (min) | — |
| + exact-match copy rows | 1.872 (min) | 1.861 (min) |

Sparse wins with plain inputs and ties once inputs are rich. Both overfit late with large hash tables on the 2M-char slice.

### E7/E8 — what replaces attention (same sparse backbone; value read = the char that followed the retrieved position)

| Mixer | Trained by | 256 context | 1,024 context: overall | 1,024: copyable | 1,024: other |
|---|---|---|---|---|---|
| None | — | 1.984 | 1.990 | 2.183 | 1.970 |
| **Hard: argmax cosine of own sparse code** | nothing in the lookup | **1.940** | **1.870** | **1.666** | **1.890** |
| Hard, fixed random sparse projection | nothing | 1.938 | — | — | — |
| Soft attention, 1 head | backprop | 1.978 | 1.973 | 2.126 | 1.958 |
| Soft attention, 4 heads | backprop | 1.971 | 1.981 | 2.189 | 1.961 |
| Hard + learned re-rank of top-4, counterfactual teaching | trial and error | — | 1.877 | 1.681 | 1.896 |
| Hard top-1 by learned q·k, counterfactual teaching, no confidence signal | trial and error | 1.990 | — | — | — |

"Copyable" = positions whose previous 8 chars already appeared earlier in the window.

- **Hard lookup wins.** It is the only mixer that learns to copy within 5M chars. Soft attention does not, with 1 or 4 heads.
- **Learned choice adds nothing.** Re-ranking by trial and error does not beat the plain nearest code. A learned scorer without a match-strength signal is worse than no mixer.
- **Full training set.** The hard-lookup model trained on all 10.8M training chars instead of 2M reached 1.825 overall (copyable 1.647, other 1.843) at the same 5.1M chars seen — no overfitting, still falling.

### Backprop transformers (raw chars, AdamW, warmup + cosine; same 5.1M chars, same validation chars)

| Model | Params | Batch × context | lr | Training stream | Val bpc |
|---|---|---|---|---|---|
| Transformer d64, 2 layers | 121k | 4 × 256 | 3e-3 | 2M | 2.365 |
| Transformer d128, 4 layers | 833k | 4 × 256 | 5e-4 / 1e-3 / 2e-3 / 4e-3 | 2M | 2.311 / **2.075** / 2.122 / 2.270 |
| Transformer d128, 4 layers | 833k | 8 × 256 | 2e-3 | 2M | 2.143 |
| Transformer d128, 4 layers | 931k | 1 × 1,024 | 2e-3 | 2M | 3.290 (batch 1 unstable; not usable) |
| **Transformer d128, 4 layers (best lr)** | 833k | 4 × 256 | 1e-3 | full 10.8M | **2.055** |
| Sparse + hard lookup, hand design | ~1.4M | 1 × 1,024 | 3e-3 | 2M | 1.870 |
| Sparse + hard lookup, hand design | ~1.4M | 1 × 1,024 | 3e-3 | full 10.8M | 1.825 |
| **Sparse + hard lookup, evolved design** | ~5.6M (4.2M of them hashed word rows, 2 read per char) | 1 × 1,024 | 3e-3 | full 10.8M | **1.689** |

At equal data, the hand-designed sparse + hard-lookup model (~1.4M params, ~1.7× the transformer) leads the tuned transformer by 0.21 bits (2M stream) to 0.23 (full stream). The evolved design leads by 0.37, but carries 4.2M hashed word-embedding rows: per-character compute is similar, parameter count is not, so the 0.37 is not a like-for-like parameter comparison.

## 4. Evolution at the structure level (E5b)

Evolution where it belongs: a (4+4) population evolved the **structure** of the sparse + hard-lookup model. Each candidate was trained for 1,500 steps of 1,024 chars (1.5M chars) with exact credit, then scored on validation bpc.

- **Genome:**
  - which context features feed the first layer: chars at offsets, hashed n-grams, skip-pairs, current-word and previous-word hashes, exact-match copy lookups of length n;
  - the width H;
  - the active count kw.
- **Mutations:** add, remove or tweak a feature; change H or kw. 8 generations, 36 evaluations, ~1 CPU-hour.

| Generation | Best val bpc (1.5M chars) | Change |
|---|---|---|
| 0 (hand design: chars 1–8) | 2.030 | — |
| 1 | 2.026 | + exact-match length 12, previous-word hash |
| 3 | 1.939 | + current-word hash |
| 4 | 1.902 | + previous-word hash |
| 7 | **1.864** | + exact-match lengths 11 and 3 |

Search alone gained 0.17 bits. It found linguistic structure (word identity, copying) that the hand design lacked. Mutations that removed the last char were heavily punished (3.2+), so the search also learned what is essential. Trained on the full training set for the same 5.1M chars, the evolved design reached **1.689** bpc overall (copyable 1.493, other 1.709) (hand design: 1.825).

## 5. Critique against the objectives

| Objective | Status |
|---|---|
| Advantage over attention | Yes at this scale: better bpc, O(1) lookups possible with an inverted index, no lookup parameters. Not shown at scale. |
| Backprop-free | Yes: forward-only exact credit, verified equal to the gradient. Cost ~2× backprop (unoptimised). |
| Evolution / EGGROLL | Whole-network ES is the wrong granularity (cone = all parameters). Per-token, cone-restricted perturbation is the effective form. Trial and error helps only where choices are discrete; here it didn't beat plain nearest-code lookup. |
| Route to LLM scale | Sparse layers are MoE-like (cost ∝ active units). A hard lookup over an inverted index is sublinear in context. Credit cost grows with depth × active units, so deep stacks need small kw. |
| Bio-inspired | k-WTA expansion (mushroom body / dentate gyrus), retrieval by code overlap (CA3), local three-factor updates. |
| Novelty | Node perturbation, forward gradients and kNN-LM are all known. New here: the cone law as a design principle, exact cone enumeration on sparse nets, and own-code hard lookup beating soft attention inside the same backbone. |

**Risks.**
- **Transformer bars are small and CPU-bound.** Induction heads need more data than 5M chars.
- **The lookup was computed as a full similarity matrix over the window here.** The inverted-index version (sublinear) is the obvious implementation but not yet built or timed.
- **Related work in the repo.** `main` of `joshua-taylor/BioTextGen` holds a GPU benchmark whose headline mixer is a random sparse expansion + winner-take-all + inverted-index memory (PSM). It is close to the random-code hard lookup here (1.938 bpc), which is a natural bridge to the GPU scale test.
- **Sparse layers in JAX are dense under the hood.** So there is no wall-clock claim yet.
- **Hard lookup is one step.** Multi-hop retrieval is untested.

## 6. Next steps (pre-registered)

1. **GPU scale test on Kaggle.**
   - Setup: sparse + hard lookup vs a tuned GPT at 20M, 100M and 500M chars of WikiText-103, char or byte level.
   - Kill: the GPT matches or beats it at 100M chars.
2. **A real sparse kernel**, then the same comparison at matched wall-clock.
3. **Depth.**
   - Stack 4 sparse layers with a hard lookup at layers 2 and 4.
   - Check that the cost of forward-only credit stays under 3× backprop.

Files: `es/` (`e1.py`, `e2.py`, `e2b.py`, `fast.py`, `e2f.py`, `jsparse.py`, `jfdo.py`, `jmem.py`, `jfdo_mem.py`, `e5.py`, `e5b.py`, `tfm.py`, `sweep.py`, `price.py`, `notes.md`), results `res_*.jsonl`, logs `log_*.txt`.
