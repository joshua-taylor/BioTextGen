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

# Round 31 (24 Oct): fast bench sb.py / sb2.py (numpy+numba, exact closed-form credit = forward-only; 0.8M chars, 2 streams x 512)
S1 calibration: base 2.2474/2.2490 (2 seeds); evo 2.0798/2.0872/2.0822 (3 seeds; mean 2.083); base-nolookup 2.424; evo-nolookup 2.113; evo-L1 2.130
  -> lookup worth 0.18 (base) / 0.03 (evo, which already has exact-copy features); seed noise ~0.005-0.02
S2 selection credit at kw32/H1024 (eff units ~430-700): swap4 2.0957/2.0756; swap8 2.0965; swap4 lr0.3 2.0904; ste8 2.0886/2.0707;
   regrow thr1 2.0905; regrow thr5 2.0849  -> all within noise: NO GAIN at moderate sparsity. swap 'good' fraction ~50% (noise-like)
S3 context mechanisms (evo feats, sb2): evo 2.0934 | rv (retrieve successor fingerprint) 2.0951 | top2 2.0801 | top4 2.0861 | lk2 (2nd lookup on layer-2 code) 2.0753
  horizon (cap): 256(64) 2.1087 | 1024(128) 2.0934 | 8192(256) 2.0432 | 32768(512) 2.0175  <- MEMORY LENGTH IS THE LEVER (-0.076 from 1k to 32k)
  'other' (not copyable within 1024): 2.132 -> 2.056 ; val stream is only 40k chars, so 32k ~ all of it.
  effective indexed memory ~ cap * H / kw ; cost per query ~ kw * cap  =>  cost for N positions ~ N kw^2 / H : extreme sparsity makes long memory cheap.
S4 sparsity (L=1, evo feats): H1024 k32 2.130 (eff 328) | H1024 k8 2.258 (eff 76, dead 35%) | H4096 k32 2.135 (dead 25%) | H4096 k8 2.287 (dead 55%)
  | H4096 k4 2.453 | H16384 k16 2.185 (dead 61%) | H16384 k8 2.269 (dead 74%) | H16384 k4 2.471
  -> active count kw dominates; extra width is wasted (units die). Utilisation is the bottleneck at high sparsity.
S5 utilisation fixes at L1 H4096 k8 (base 2.287, dead 55%, eff 78): swap4 2.393 (worse) | regrow 2.303 | homeo(bias,0.01) 2.918 (much worse: shifts activations) | homeo+regrow 4.05
  -> rule 14: never pkill/pgrep -f a pattern that appears in your own command line (killed the shell twice; waiter loops matched themselves).
S6 gentler utilisation fixes at L1 H4096 k8 (base 2.287): boost0.1 2.266 | boost0.3 2.259 (eff 78->191) | boost1 2.292 (eff 1696!) | explore0.3 2.270 | explore1 2.347
  | boost0.3+explore0.3 2.253 (best, -0.034) | homeo5e-4 2.365 ; mainline L2 k32: boost0.3 2.095, explore0.3 2.083 (no effect)
  -> utilisation fixable but small gains; kw (units per prediction) is the capacity limit. Extreme sparsity costs accuracy.
Corpus memory (e_prefill; trained on first half of train, prefill from unseen end of train, cap limits index to ~65k moments):
  h1k-trained: 0 2.053 | 100k 2.013 | 500k 2.006 | 2M 2.004 ; h32k-trained: 1.994 | 1.994 | 1.980 | 1.978  (gains on 'other' chars)
Pair index (pairmem.py): bucket = pair of top-8 units, 2^18 buckets x 64; ~same quality as unit index at 100k history, 2x faster.
S7: hebbian competitive learning HURTS: L1 H4096 k8 hebb0.01 2.431, 0.05 2.585; mainline hebb0.01 2.147 (vs 2.083)
    novelty-gated writes: cap128 h32k 2.0786/2.0786/2.0759 (wt 2/0.95/0.85); cap512 2.0175/2.0243/2.0175 -> no gain
    index capacity matters: h32k cap128 2.079 vs cap512 2.018
S8 keyed lookups (fixed random LSH key over a chosen feature subset; evo feats, h32k cap512; base 2.0175):
    word key (wd,pw) 1.9925 | 12-char key 1.9886 | n-gram key (ng4,ng6) 1.9870 | prev-word only 2.0221 | word + 12-char 1.9737 (-0.044)
    -> what the memory is keyed on is a discrete design choice; two extra keys stack.
Long-document memory, FIXED scoring window (val chars 1,000,448..1,040,448), history = preceding val text read write-only (evo h32k-trained):
  unit index: 0 2.0895 | 30k 2.0210 | 100k 2.0068 | 300k 2.0039 | 1M 1.9969 (292 s)
  pair index: 0 2.0621 | 30k 2.0513 | 100k 2.0535 | 300k 2.0534 | 1M 2.0280 (36 s)
  -> -0.093 bits from reading 1M chars of earlier text (no learning). Pair index 8x faster, loses ~0.03 recall at long range.
  (earlier e_long rows used different windows per history -> discarded)
FD check of sb2 local credit (keys+lookup+2 layers, check_sb2.py): 47/48 random weights match the measured loss slope within float32
  resolution (6e-5); the one mismatch is a nudge that flips the winner set (selection jump), as expected.
evo3 (structure evolution, 300-step fitness, 42 evals, stopped after stall): 2.252 (start, evo h32k) -> gen1 2.210 (top2 + output lr x2) -> gen2 2.180 (+ng4 input); runners-up: word key 2.185-2.213
S9 combination at full budget (0.8M chars, h32k cap512):
  A evo-h32k 2.0175 / 2.0035 (2 seeds) | B +ng4 1.9754 | C +ng4+2 keys(word,12c) 1.9239 | D C+top2+lmO2 1.9198 / 1.8968 (mean 1.908, -0.102 vs A)
  E D with 131k reach (cap1024) 1.9247 (val window only 40k -> can't show >32k here)
F D trained 2400 steps (2.46M chars): 2.040 1.930 1.828 1.791 1.761 1.769 (copy 1.629 other 1.789)
