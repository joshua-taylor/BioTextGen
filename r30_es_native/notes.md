# Round 30 working notes — ES-native text generation

## Exam question
1. Without backprop, can we design something better optimised for text gen than attention?
2. How do we control the search space so EGGROLL-style ES (or a better evolution method) is efficient?

## Facts about EGGROLL (arXiv 2511.16652)
- Perturbation E = (1/sqrt r) A B^T per member, r = 1 typical; update M += (a/N) sum_i f_i E_i (rank up to N r).
- Pretrained EGG: int8 6L-256D GRU, char-level minipile, 3.40 bits/byte vs backprop transformer 3.58 at the same tiny batch (16 seqs),
  but ~180x the GPU-hours. Throughput ~ inference; sample efficiency is NOT claimed.

## Variance law (the "dependency cone")
Loss L = sum_b l_b (one term per predicted token). Perturbations xi_j (on weights or unit activations), antithetic.
Credit estimate for coordinate i using only the loss terms that depend on it:
  g_i ~= sum_{b in C(i)} (dl_b) xi_i / (2 sigma)
  signal = sum_{b in C(i)} dl_b/dxi_i
  noise variance ~= sum_{b in C(i)} sum_{j in D(b), j != i} (dl_b/dxi_j)^2
C(i) = loss terms touched by i; D(b) = perturbed quantities that loss term b depends on (its "cone").
=> Cost relative to backprop ~ |D(b)|. Parameters cost nothing; only perturbed units in a prediction's cone do.
Consequences:
- classic ES (one weight perturbation shared by every token): cone = ALL perturbed params -> cost ~ d.
- per-token node perturbation (batch = population): cone = units in that token's computation -> cost ~ width x depth (x positions mixed).
- attention: each prediction depends on every earlier position -> cone ~ width x depth x context. Worst case for ES.
- huge sparse tables with narrow units: capacity free, cone small. Hard retrieval of k positions: cone ~ k, not context.
- if the true credit lies in an r-dim subspace and the noise is drawn there, cone -> r (guided / learned-subspace noise).

## E1 results (char WikiText-2, k=8 context, B=256, 20k steps = 5.1M chars; val bpc on 50k chars)
| model | backprop | per-token node perturbation | readout only (hidden random) | classic ES (N=32, equal compute) |
|---|---|---|---|---|
| H32 L1 | | | | |
| H128 L1 | 2.381 | 3.176 | 3.466 | 3.482 |
| H128 L2 | 2.241 | 3.342 | | |
| H512 L1 | 2.317 | 3.128 | 3.457 | |
| H512 L2 | 2.313 | 3.379 | | |
- estimator check: node-perturbation estimate converges to the true gradient (cos 0.98 after 4000 draws); single draw cos 0.09 at B=256.
- np at 20k steps ~ bp at ~300 steps: price ~ 60-100x in data, consistent with cone ~256-1152 (input layer 128 + hidden).
- classic ES learns nothing beyond the readout at equal compute.

## Guided (learned-subspace) node perturbation — negative (quick test)
H128 L1, 2000 steps: np 3.29; guided r=16 3.33; r=4 3.34; r=16 alpha=0 3.33. No gain: hidden credit isn't low-rank here (V=96 ~ H).

## Exact forward-only credit on sparse nets ("fdo")
Perturb each ACTIVE unit of every token one at a time (antithetic), read each token's own loss change.
- sparse H1024 kw16 L1, 300 steps: bp 3.0952, fdo 3.0952 (identical), wall-clock 6.4 s vs 9.5 s.
- L2 (H512 kw8), sigma 1e-3: 4.269 vs 4.275 after 100 steps (float32 finite-difference noise).
Cost of forward-only exact credit = sum over layers of (units in the cone) x (cost to recompute downstream).
Cheap when the cone is small (sparse) or the downstream is linear (last layer). Dense deep nets: ~H x forward.

## Forward-only exact credit for the 2-layer sparse net (jfdo.py)
Cone per token = 64 embedding dims + 32 active layer-1 units + 32 active layer-2 units (128 forward-mode probes).
Check vs backprop gradient on every weight block: cosine 1.000000, norm ratio 1.0000.
200 steps: bp 2.7159 bpc (28.5 s), forward-only 2.7129 bpc (64.4 s) -> 2.3x wall-clock, unoptimised.

## E6 (5.1M chars, 5000 steps of 4x256)
sparse-base (emb 64 -> 2 k-WTA layers H1024 kw32, residual): 1.982 bpc
E6 full (val bpc at 1/5..5/5 of 5000 steps; NB training stream = first 2M chars, so 5.1M = 2.5 epochs -> late overfit with big hash tables)
| model | curve | final |
| sparse-base (H1024 kw32 L2) | 2.229 2.094 2.031 2.001 1.982 | 1.982 |
| dense-base (H256 dense L2)  | 2.375 2.249 2.161 2.149 2.122 | 2.122 |
| sparse-rich | 1.968 1.920 1.909 1.937 1.936 | |
| sparse-richmem | 1.923 1.872 1.876 1.873 1.902 | min 1.872 |
| dense-richmem | 1.894 1.861 1.871 1.885 1.909 | min 1.861 |
Sparse beats dense with raw char inputs (-0.14); equal once inputs are rich. JAX sparse impl is dense under the hood (11x slower), no compute claim.

## Price of random per-token perturbation (fast.py, one-hot inputs, L1, 5.1M chars; exact credit = K0)
| config | cone | K0 (exact) final | K1 final | K4 final | data price K1 / K4 (power-law fit of exact curve) |
| H1024 kw16 | 16 | 2.103 | 2.446 | 2.272 | ~5x / ~2.4x |
| H1024 kw64 | 64 | 2.014 | pending | | |
| H128 dense | 128 | 2.323 | 2.761 | pending | >=8x and rising (fit unreliable) |
Sparse also wins on this model: kw16 2.10 / kw64 2.01 vs dense H128 2.32.
Final price table (end of run, power-law inversion of the exact curve):
| cone | K1 price | K4 price |
| 16 (H1024 kw16) | ~5x | ~2.4x |
| 64 (H1024 kw64) | ~12x | |
| 128 (dense H128) | (final 2.761 vs exact 2.323) | ~5x |
=> price grows with the cone, roughly cone/5 per draw; K draws divide it by ~K.

## E7: context mixer inside the sparse backbone (T=256 windows, 5.1M chars, ntrain 2M)
| mixer | trained by | val bpc |
| none | - | 1.984 |
| hard: argmax cosine of own layer-1 sparse code, value=(cos bin, next char) | nothing in lookup | 1.940 |
| hardq: same with a fixed random sparse projection code | nothing | 1.938 |
| soft 1 head (q,k from code; value = next-char emb) | backprop | 1.978 |
| soft 4 heads | backprop | 1.971 |
| hardc: learned q.k top-1, scorer taught by counterfactual top-4 losses; NO confidence bin | trial and error | 1.990 |
Transformers (B=20, only 1000 steps - undertrained): d64L2 2.662, d128L4 2.282. Tuned (B=4, 5000 steps) queued in E8.
Forward-only training of hard and hardc verified equal to gradient on every block (cos 1.000000).

## E5b structure evolution (sparse + hard lookup, T1024, 1500 steps/eval, (4+4), 8 gens, 36 evals)
gen0 2.030 (chars 1-8) -> gen3 1.939 (+current word hash) -> gen4 1.902 (+prev word) -> gen7 1.864 (+exact-match 11 and 3).
Best genome on full train data, 5000 steps T1024: 1.689 overall, copyable 1.493, other 1.709 (hand design same setting: 1.825).

## E9 transformer tuning (d128 L4, B4 x 256, 5000 steps = 5.1M chars)
lr 5e-4 2.311 | 1e-3 2.075 | 2e-3 2.122 | 4e-3 2.270 ; B8 lr2e-3 2.143 ; best lr on full train stream: 2.055
