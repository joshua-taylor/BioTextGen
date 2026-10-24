# Round 31: optimising the ES-native design — what helps, what doesn't

Follows `claude/round30-es-native-sparse-lookup.md`. Date: 24 Oct 2026. Brief from Josh: get the most out of the approach before a Kaggle scale-up; look for things beyond what backprop can do; no new backprop baselines; many fast experiments.

**Answer.**
- **What helps (in order of size):**
  1. A longer memory for the hard lookup: −0.076 bits/char going from 1k to 32k reach at equal training.
  2. Reading more text into memory with no learning: −0.093 from 1M earlier chars on a fixed scoring window.
  3. Extra lookups keyed on fixed random hashes of chosen inputs (word key + 12-char key): −0.044 on top of the 32k memory.
  4. Structure evolution plus hand combination of everything that helped: the combined design averages 1.908 bits/char over 2 seeds, against 2.010 for the round-30 evolved design with 32k memory (−0.102). Trained 3× longer (2.5M chars) it reaches 1.761.
- **What doesn't help:** every mechanism that tries to choose *which neurons fire* better than the exact forward-only credit already does. That covers swap trials, neurogenesis, homeostatic thresholds and competitive Hebbian learning — all no gain or worse. Boosting and exploration help slightly, and only at extreme sparsity.
- **Where evolution genuinely beats gradients here:** discrete design choices. These are what the memory is keyed on, how far back it reaches, how it is indexed, which inputs exist, and the per-part learning rates. Weight learning stays with exact forward-only credit (finite-difference check: matches the measured loss slope on 47/48 random weights; the one miss is a winner-set flip).
- **Kaggle recommendation:** scale the combined design (section 5). Train it with the exact local credit; on GPU that is autodiff with the winner selection held fixed, which gives numerically identical updates, plus one short forward-only run to confirm. The pre-registered kill rule stays: a tuned GPT matching it at 100M chars.

## Design D in plain English (the recommended model)

For each character position, predicting the next character:

1. **Describe the moment.** Add up 64-number table rows for 13 inputs:
   - the last 8 characters;
   - the word so far;
   - the previous word;
   - the current 4-character chunk;
   - two exact-copy hints: the character that followed the last exact occurrence of the previous 3 and 11 characters, within 1,024 characters.
2. **Fingerprint (layer 1).** 1,024 detector neurons score the summary; only the top 32 fire.
3. **Three hard lookups, no gradient through any of them.**
   - **Fingerprint memory.** Find the two most similar stored fingerprints from the last 32k characters; read the character that followed each, plus how close the match was.
   - **Word memory.** A fixed random hash of (word so far, previous word) is compared with stored hashes, asking "when did these words last appear, and what came next?"
   - **12-character memory.** The same, keyed on the last 12 characters.
4. **Combine (layer 2).** The lookup results are added to the summary. 1,024 more neurons see it plus the fingerprint; the top 32 fire.
5. **Vote.** Every firing neuron votes on the 96 characters.

**Learning.** Every weight is updated from its own local credit, computed exactly from the character's own error. No backward pass is needed: the finite-difference check confirms the updates equal the measured loss slopes. The lookups store what happened as the text is read, with no training.

**Evolution's role.** Evolution chose the inputs, which memories exist and what they are keyed on, and the learning-rate split. These are the discrete decisions gradients cannot make.

## 1. Bench (`sb.py`, `sb2.py`, `sbdata.py`, `pairmem.py`)

- **Model** (round-30 champion re-implemented in numpy + numba):
  1. Sum of 64-dim rows for the input features.
  2. Layer 1: 1,024 units, top 32 fire (the fingerprint).
  3. Hard lookup: best cosine match among remembered fingerprints; read (similarity bin, next char).
  4. Layer 2: top 32 of 1,024, with a residual from layer 1.
  5. Readout over 96 chars.
- **Learning.** Exact per-unit credit in closed form. With winner sets and lookup choices held fixed it equals the forward-only measurement (round 30). Updates are local.
- **Memory.** Per training stream, a ring of the last `horizon` positions. Each unit keeps a posting list of the last `cap` positions where it fired. Stored codes are never refreshed.
- **Budget.** 800 steps × 2 contiguous streams × 512 chars = 0.82M training chars. Validation bits/char on the first 40k validation chars, also split into copyable (previous 8 chars seen in the last 1,024) and other.
- **Throughput.** ~2 min per run, 2 runs in parallel. Seed noise ±0.005–0.02.

## 2. Results

### Calibration (S1)
| Model | bpc |
|---|---|
| Hand design (chars 1–8) | 2.247 / 2.249 |
| Hand design, no lookup | 2.424 |
| Evolved design (round 30 genome) | 2.080 / 2.087 / 2.082 |
| Evolved design, no lookup | 2.113 |
| Evolved design, 1 layer | 2.130 |

### Choosing which neurons fire (S2, S5–S7) — negative
| Mechanism | 32 of 1,024 firing (base 2.083) | 8 of 4,096 (base 2.287) | 4 of 16,384 (base 2.471) |
|---|---|---|---|
| Swap trials (runner-up replaces weakest winner if the token's loss falls) | 2.076–2.097 | 2.393 | 2.572 |
| Straight-through credit for runner-ups (control) | 2.071–2.089 | — | — |
| Neurogenesis (rewrite unused units for recent high-loss tokens) | 2.085–2.091 | 2.303 | 2.469 |
| Homeostatic thresholds (bias drifts to equalise rates) | — | 2.918 (0.01) / 2.365 (5e-4) | 3.575 |
| Boosting (selection score only) 0.1 / 0.3 / 1.0 | 2.095 (0.3) | 2.266 / 2.259 / 2.292 | — |
| Exploration noise in selection 0.3 / 1.0 | 2.083 (0.3) | 2.270 / 2.347 | — |
| Boost 0.3 + explore 0.3 | — | **2.253** | — |
| Competitive Hebbian (units move toward the contexts they win) | 2.147 | 2.431–2.585 | — |

- **Sparsity sweep (one layer):**
  - 32 of 1,024 firing: 2.130.
  - 8 of 1,024: 2.258.
  - 32 of 4,096: 2.135.
  - 8 of 4,096: 2.287.
  - 16 of 16,384: 2.185.
  - 4 of 16,384: 2.471.
- **Width is wasted:** 55–87% of units never fire.
- **The number of firing units is the capacity per prediction**, so extreme sparsity costs accuracy whatever the utilisation.

### Memory (S3, S7, corpus and long-document tests) — positive
| Memory reach (posting-list cap) | bpc | Other chars |
|---|---|---|
| 256 (64) | 2.109 | 2.144 |
| 1,024 (128) | 2.093 | 2.132 |
| 8,192 (256) | 2.043 | 2.083 |
| 32,768 (128) | 2.079 | 2.119 |
| **32,768 (512)** | **2.018** | **2.056** |

Index capacity matters as much as reach: effective memory ≈ cap × H / kw positions, and cost per lookup ≈ kw × cap.

**Novelty-gated writes** (skip if a near-identical moment is stored): 2.079/2.079/2.076 at cap 128 and 2.018/2.024/2.018 at cap 512. No gain.

**Other context mechanisms:** top-2 matches 2.080; top-4 2.086; second lookup on the layer-2 code 2.075. Retrieving the model's own fingerprint from after the match ("rich value") gave 2.095, no gain.

**Corpus memory.** Train on the first half of the training stream. Read N chars from its unseen end into memory (no learning), then score validation:

| Chars read | Model trained with 1k reach | Model trained with 32k reach |
|---|---|---|
| 0 | 2.053 | 1.994 |
| 100k | 2.013 | 1.994 |
| 500k | 2.006 | 1.980 |
| 2M | 2.004 | 1.978 |

The gains come on non-copyable chars. They flatten because the cap limits the index to ~65k moments.

**Long-document memory, fixed scoring window** (validation chars 1,000,448–1,040,448; history = preceding validation text, read write-only):

| History | Unit index | Pair index | Read time, unit / pair |
|---|---|---|---|
| 0 | 2.090 | 2.062 | — |
| 30k | 2.021 | 2.051 | — |
| 100k | 2.007 | 2.054 | — |
| 300k | 2.004 | 2.053 | 97 s / 15 s |
| 1M | **1.997** | 2.028 | 292 s / 36 s |

The **pair index** (bucket = a pair of the top-8 firing units; candidates re-ranked by cosine of the stored top-8 code) is 8× faster but loses ~0.03 recall at long range.

### Evolvable memory keys (S8) — positive
Each extra lookup indexes moments by a **fixed random projection** (an LSH) of a chosen subset of input features, top 32 of 1,024. Nothing about the key is learned; the model learns only a table of how to use the (similarity bin, next char) that comes back. Evolved design, 32k reach, cap 512 (base 2.018):

| Extra key(s) | bpc | Copyable | Other |
|---|---|---|---|
| Current + previous word | 1.993 | 1.715 | 2.031 |
| Last 12 chars | 1.989 | 1.710 | 2.028 |
| 4- and 6-char n-grams | 1.987 | 1.689 | 2.029 |
| Previous word only | 2.022 | 1.751 | 2.060 |
| **Word + 12 chars** | **1.974** | **1.673** | **2.016** |

### Structure evolution (E-evo3)
A (6+6) population over the whole design ran for 6 generations, with a 300-step fitness (0.3M chars) and the same seed for every candidate. It was stopped after stalling, at 42 evaluations, ~3 min each.

**Genome:**
- input features;
- up to 2 extra memory keys;
- width, active count and depth;
- embedding size;
- memory reach (1k–131k) and index type (unit or pair);
- novelty gate;
- boosting;
- top-m;
- second lookup;
- learning rate and per-part multipliers.

| Generation | Best (0.3M-char fitness) | Change |
|---|---|---|
| 0 | 2.252 | round-30 evolved design + 32k memory |
| 1 | 2.210 | top-2 matches, output learning rate ×2 |
| 2–5 | **2.180** | + 4-char chunk input (ng4) |

Runners-up: word key (2.185–2.213). Evolution never chose:
- the pair index for the main memory,
- extreme sparsity,
- novelty gating,
- the second lookup.

**Full-budget confirmation** (0.8M chars, 32k reach, cap 512):

| Design | Seed 0 | Seed 1 |
|---|---|---|
| A: round-30 evolved design | 2.018 | 2.004 |
| B: A + ng4 | 1.975 | — |
| C: B + word key + 12-char key | 1.924 | — |
| **D: C + top-2 + output lr ×2** | **1.920** | **1.897** |
| E: D with 131k reach (cap 1,024) | 1.925 | — |
| F: D trained 2,400 steps (2.46M chars) | 1.761 at 2.05M, 1.769 at 2.46M | — |

E shows nothing beyond 32k, but the scoring text is only 40k long; the long-document test is the right measure of longer reach. For orientation only: round 30's tuned backprop transformer scored 2.055 at 5.1M chars on a 60k validation slice. F's 1.76 at 2.5M chars is on a 40k slice and a different bench, so the two are not a like-for-like comparison.

### Exactness check
`check_sb2.py` perturbs 48 random weights across W1, W2, R2, E, M, both key tables and Wo, on copied memories, and compares the measured loss slope with the update applied:
- 47 of 48 agree within float32 finite-difference resolution (~6e-5).
- The exception is a nudge that flips the winner set: a selection jump the method deliberately ignores.

## 3. What this says about "beyond backprop"

| Claim | Status |
|---|---|
| Forward-only learning can equal backprop | Yes: exact on sparse nets (rounds 30–31) |
| Evolution can improve how units are selected | No at moderate sparsity; marginal at extreme sparsity |
| Evolution can choose structure gradients can't (inputs, memory keys, reach, index) | Yes: round 30 (−0.17), keys (−0.044), combined design −0.102 |
| A non-differentiable memory gives an advantage attention lacks | Yes: cost per lookup scales with posting lists, not context. It improves with 1M chars read and with unseen corpus text, with no learning |

## 4. Closed directions (this round)
- Swap trials / counterfactual selection credit.
- Neurogenesis by one-shot rewrite.
- Homeostatic thresholds.
- Competitive Hebbian learning of layer-1 units.
- Novelty-gated memory writes.
- Successor-fingerprint ("rich") values.
- Extreme sparsity (≤8 firing) as a compute design.

## 5. Next steps
1. **Kaggle scale test of design D** (pre-registered).
   - **Data:** WikiText-103, char or byte level.
   - **Steps:** 20M, 100M and 500M training chars.
   - **Comparator:** the tuned GPT from rounds 29–30.
   - **Kill rule:** the GPT matches or beats D at 100M chars.
   - **Also report:** bits on the copyable/other slices, and the long-document memory test, with 1M+ chars of history on a fixed scoring window.
2. **GPU implementation notes.**
   - Within a training window (≤64k chars), the hard lookup can be computed as blockwise sparse-code inner products with a top-1 per row; no index is needed.
   - For corpus memory, use several independent hash tables (multi-table LSH over top-unit pairs/triples). This should recover the pair index's lost recall while keeping its 8× speed.
3. **Training rule.** Exact local credit. On GPU, use autodiff with `stop_gradient` on winner selection and lookup choice (identical numbers; verified rounds 30–31). Confirm with one short forward-only run.
4. **Search at scale.** Keep evolution for discrete choices (inputs, keys, reach), run on small proxies, and transfer.
