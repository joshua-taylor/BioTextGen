# BioTextGen — bio-inspired sequence mixer benchmark

Compares bio-inspired sequence mixers against standard baselines on a
WikiText-103 subset. Scores **train perplexity**, **validation perplexity**
and **tokens/sec**.

The headline candidate is a **pattern-separated episodic memory**: addresses
from a frozen random sparse expansion plus winner-take-all, writes by a
cerebellar delta rule touching only the active rows, reads by inverted-index
lookup, and four compartments competing so the longest matching context wins.

```bash
pip install -r requirements.txt
python tests.py                       # correctness — run this first
python bench.py --arms sliding_only,full_attention,gla,deltanet,psm
```

---

## The one thing this benchmark is designed to get right

It is very easy to build a harness that compares *implementation quality* and
call it a comparison of *mechanisms*. Attention gets a fused CUDA kernel; your
new idea gets a Python loop; your idea loses; you learn nothing. Six controls
guard against that.

**1. One trunk, one slot.** Every arm is the same 12-block model with the same
embedding, RMSNorms, SwiGLU FFNs, depth, width, tied unembedding and init
seed. Blocks 3, 6, 9 and 12 are the *memory slot*; only the mixer in those
four blocks changes. The other eight blocks are always sliding-window
attention (window 64). The `sliding_only` arm fills the memory slot with
another sliding-window block, making it a genuine "no long-range mechanism"
control rather than a different architecture.

**2. Shared implementations.** The two attention arms are one class differing
only in the mask; both use `scaled_dot_product_attention`. GLA and DeltaNet
are *literally the same function* with one boolean flag — neither can be
advantaged by better code. `tests.py` verifies both against a naive
token-by-token reference (max abs error ~1e-6).

**3. Parameter parity.** Non-embedding parameter counts are held within 5%
across arms and printed for every run; a wider gap is flagged loudly in the
results table. Current spread is 2.8%.

**4. Identical data and schedule.** Same token stream, same batch order (same
`Batcher` seed), same optimiser, LR, warmup, cosine schedule, gradient
clipping, precision and step count. Evaluation uses deterministic
non-overlapping sequential windows, not random samples.

**5. Timing hygiene.** Throughput excludes the first 20 steps (allocator and
autotuner noise) and synchronises CUDA around every measured region. Training
and inference throughput are reported separately.

**6. Thermal-drift control.** Arm execution order is shuffled per seed, so a
GPU that slows down as it heats cannot systematically favour whichever arm
happened to run first.

### What is still not a fair comparison

Tokens/sec is genuinely implementation-dependent and always will be. Attention
runs on a fused kernel that has had years of engineering; the memory mixer
runs on `index_select`/`index_add_`. **Read perplexity as the mechanism
comparison and throughput as a statement about today's kernels.** A mechanism
that wins on perplexity and loses 40% on throughput is a kernel problem, not a
research failure. A mechanism that loses on perplexity is a research failure
regardless of throughput.

---

## Arms

| Arm | Memory slot contains | Role |
|---|---|---|
| `sliding_only` | sliding-window attention (64) | Floor: no long-range mechanism |
| `full_attention` | full causal attention | Ceiling |
| `gla` | gated linear attention | Linear-attention rung |
| `deltanet` | gated delta rule | **The rung that matters.** Any new mixer must beat this, not vanilla linear attention |
| `psm_single` | pattern-separated memory, one compartment | Isolates addressing from compartments |
| `psm` | pattern-separated memory, compartments (1,2,4,8) | The candidate |
| `psm_depression` | `psm` + short-term synaptic depression | Ablation |
| `psm_modulators` | `psm` + neuromodulatory channels | Ablation |
| `psm_full` | `psm` + both | Combined |

---

## The candidate mechanism

Per compartment *n*, at each token:

1. **Address.** Project the last *n* hidden states (causal depthwise conv, then
   linear) to `d_a=256`. Expand through two **frozen** random half-codebooks in
   product form and take the top `k=32` of the Cartesian product. Nothing in
   the expansion is learned — that is the pattern-separation claim, and it is
   why no auxiliary anti-collapse loss is needed.
2. **Read.** Gather the `k` addressed rows, weight by address values, divide by
   accumulated row usage (divisive normalisation).
3. **Write.** Delta rule: `err = v - read`, scattered back into the same `k`
   rows, scaled by a write rate gated on surprise (previous-token NLL).
4. **Compete.** Compartments are mixed by softmax over match strength plus a
   learned priority bias — a differentiable longest-prefix match.

Memory operates at chunk granularity (64 tokens): reads use the state as of
the chunk start, writes land at the chunk end. **This is a design choice, not
a shortcut.** Intra-chunk mixing under 64 tokens is the sliding window's job
in the hybrid layout; letting the memory duplicate it contradicts the
labour-division finding that retrieval and local mixing should not compete.

Why it should beat DeltaNet, stated against the measured deficit:

| Deficit component | Mechanism |
|---|---|
| Weighting function (~0.53 nats) | Sparse code overlap is a steep similarity by construction — no softmax, no temperature. `tests.py` measures the falloff: overlap 1.00 → 0.88 → 0.50 → 0.19 → 0.00 as input distance grows. |
| Delta rule (~0.36 nats) | Present, and costs `k` rows instead of a `d×d` state. |
| Range (~0.07 nats) | Per-compartment row decay plus usage normalisation. |
| Raw values | With M=8281 rows, k=32 active, T=512, expected writes per row ≈ 2, and the delta rule overwrites rather than sums — a read returns approximately one stored raw value, not a blend. |

---

## Reading the output

```
arm                 train ppl    val ppl  val nats   tok/s tr  tok/s inf  non-emb par
```

Plus a `vs deltanet rung` table in nats. That difference is the number to care
about. The pre-registered claim from the design document: **`psm` must beat
`deltanet` by ≥0.25 nats** with the gap between `deltanet` and `full_attention`
as the available headroom. Less than that and the design is an incremental
variant, and should be reported as a clean negative.

`psm_single` vs `psm` isolates the contribution of compartments specifically.
If `psm_single` already captures most of the gain, the longest-match
machinery is not earning its complexity.

---

## Files

| File | Contents |
|---|---|
| `data.py` | WikiText loading, BPE training, deterministic batching |
| `mixers.py` | All sequence mixers; shared chunked scaffold |
| `model.py` | Shared trunk, block layout, arm construction |
| `bench.py` | Training, evaluation, throughput, results table |
| `tests.py` | Correctness: chunked math, causality, separation, parity |

## Config

Defaults: 12 blocks, `d_model=384`, 6 heads, seq 512, batch 16, BPE vocab
8192, AdamW, lr 1e-3, cosine, 2000 steps. ~24M parameters. Tuned to fit a
Kaggle T4; `--precision` defaults to bf16 where supported and fp16 otherwise.

Memory state is `M × d_v × C` per memory block per sequence = 8281 × 128 × 4
floats. Reduce `d_v` first if you hit VRAM limits, `m_half` second.

## Data sources

`data.py` tries, in order: HuggingFace `datasets` (needs internet), any
wikitext files under `/kaggle/input`, then a file at `$BENCH_TEXT`. If all
three fail it exits rather than silently benchmarking on something else.
