"""
Benchmark runner.

Scores every arm on train perplexity, validation perplexity and tokens/sec.

Fairness controls, all enforced here rather than trusted:
  - identical trunk, depth, width, init seed and parameter init
  - identical token stream and batch order (same Batcher seed per arm)
  - identical optimiser, schedule, warmup, clipping, precision, steps
  - identical eval batches (deterministic sequential windows)
  - parameter counts reported per arm; a >5% non-embedding gap is flagged
  - throughput measured after warmup steps, excluding compile/alloc noise
  - the arm order is randomised per seed so GPU thermal drift cannot
    systematically favour whichever arm runs first
"""
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch

from data import get_data, Batcher
from model import Model


SIZES = {
    # name:   d_model n_layers n_heads  d_a d_v m_half  k   vocab
    "tiny":  dict(d_model=192, n_layers=6,  n_heads=3, d_a=128, d_v=48,
                  m_half=45, k=16, vocab=2048),
    "small": dict(d_model=256, n_layers=8,  n_heads=4, d_a=192, d_v=64,
                  m_half=64, k=16, vocab=4096),
    "base":  dict(d_model=384, n_layers=12, n_heads=6, d_a=256, d_v=128,
                  m_half=91, k=32, vocab=8192),
}

DEFAULT_CFG = dict(
    window=64, chunk=64, orders=(1, 2, 4, 8), surprise_gate=True,
)

ARMS = {
    # name              memory-slot mixer   depression  modulators
    "sliding_only":     ("sliding",         False, False),
    "full_attention":   ("attention",       False, False),
    "gla":              ("gla",             False, False),
    "deltanet":         ("deltanet",        False, False),
    "psm_single":       ("psm_single",      False, False),
    "psm":              ("psm",             False, False),
    "psm_depression":   ("psm",             True,  False),
    "psm_modulators":   ("psm",             False, True),
    "psm_full":         ("psm",             True,  True),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, batcher, n_batches, device, amp_dtype):
    model.eval()
    tot, ntok = 0.0, 0
    for x, y in batcher.sequential(n_batches):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            _, loss = model(x, y)
        tot += loss.item() * y.numel(); ntok += y.numel()
    model.train()
    return tot / max(ntok, 1)


def run_arm(name, spec, train_ids, val_ids, vocab, args, device, amp_dtype):
    mixer, depression, modulators = spec
    set_seed(args.seed)
    cfg = args.model_cfg
    model = Model(vocab, memory_mixer=mixer, cfg=cfg,
                  use_depression=depression, use_modulators=modulators).to(device)
    pr = model.param_report()
    print(f"\n=== {name} === params total {pr['total']:,} "
          f"non-emb {pr['non_embedding']:,} memory-slot {pr['memory_slot_mixers']:,}",
          flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup,
        0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    tb = Batcher(train_ids, args.batch_size, args.seq_len, seed=args.seed)
    vb = Batcher(val_ids, args.batch_size, args.seq_len, seed=args.seed)
    eb = Batcher(train_ids, args.batch_size, args.seq_len, seed=args.seed + 777)

    hist, t_train, tok_train = [], 0.0, 0
    prev_nll = None
    for step in range(args.steps):
        x, y = tb.batch(); x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits, loss = model(x, y, prev_nll=prev_nll)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if step >= args.warmup_timing:                 # exclude warmup from tok/s
            t_train += dt; tok_train += y.numel()

        with torch.no_grad():                          # surprise for next step
            # cross_entropy(reduction='none') avoids materialising a full
            # (B,T,V) fp32 log_softmax -- that was 268 MB per step at base size
            nll = torch.nn.functional.cross_entropy(
                logits.detach().view(-1, logits.size(-1)).float(),
                y.reshape(-1), reduction="none").view_as(y)
            z = (nll - nll.mean()) / (nll.std() + 1e-6)
            prev_nll = torch.sigmoid(z)

        hist.append(loss.item())
        if (step + 1) % args.log_every == 0 or step == 0:
            msg = (f"  step {step+1:5d}/{args.steps}  loss "
                   f"{np.mean(hist[-args.log_every:]):.4f}")
            if t_train > 0:
                tps = tok_train / t_train
                eta = (args.steps - step - 1) * y.numel() / max(tps, 1e-9)
                msg += f"  {tps:,.0f} tok/s  eta {eta/60:.1f} min"
            print(msg, flush=True)

    train_loss = evaluate(model, eb, args.eval_batches, device, amp_dtype)
    val_loss = evaluate(model, vb, args.eval_batches, device, amp_dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); n = 0
    with torch.no_grad():
        for _ in range(args.throughput_batches):
            x, y = tb.batch(); x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                model(x, y)
            n += y.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_tps = n / (time.perf_counter() - t0)

    peak = (torch.cuda.max_memory_allocated() / 1e9
            if device.type == "cuda" else 0.0)
    res = {
        "arm": name, "peak_mem_gb": peak, "mixer": mixer, "depression": depression,
        "modulators": modulators, "params": pr,
        "train_loss": train_loss, "val_loss": val_loss,
        "train_ppl": math.exp(min(train_loss, 20)),
        "val_ppl": math.exp(min(val_loss, 20)),
        "train_tokens_per_sec": tok_train / max(t_train, 1e-9),
        "infer_tokens_per_sec": infer_tps,
        "final_train_batch_loss": float(np.mean(hist[-50:])),
    }
    print(f"  -> train ppl {res['train_ppl']:.2f}  val ppl {res['val_ppl']:.2f}"
          f"  {res['train_tokens_per_sec']:,.0f} tok/s train"
          f"  peak {peak:.2f} GB", flush=True)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="sliding_only,full_attention,gla,deltanet,psm")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--warmup-timing", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--throughput-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-chars", type=int, default=30_000_000)
    p.add_argument("--val-chars", type=int, default=1_000_000)
    p.add_argument("--size", default="small", choices=list(SIZES))
    p.add_argument("--vocab-size", type=int, default=0,
                   help="0 = use the size preset's vocab")
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--out", default="results.json")
    p.add_argument("--allow-cpu", action="store_true",
                   help="run without a GPU (roughly 50x slower; for debugging only)")
    args = p.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2 or P100, "
            "then re-run. A full sweep on CPU takes days. Pass --allow-cpu to override.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "auto":
        if device.type != "cuda":
            amp = None
        elif torch.cuda.is_bf16_supported():
            amp = torch.bfloat16
        else:
            amp = torch.float16
    else:
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16,
               "fp32": None}[args.precision]
    print(f"device {device}  precision {amp}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    preset = dict(SIZES[args.size])
    vocab_size = args.vocab_size or preset.pop("vocab")
    preset.pop("vocab", None)
    args.model_cfg = dict(DEFAULT_CFG); args.model_cfg.update(preset)
    print(f"\n[size] {args.size}: " +
          "  ".join(f"{k}={v}" for k, v in preset.items()), flush=True)

    print("\n[data]", flush=True)
    train_ids, val_ids, vocab = get_data(args.train_chars, args.val_chars,
                                         vocab_size)

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n}; choose from {list(ARMS)}")
    order = list(names)
    random.Random(args.seed).shuffle(order)             # thermal-drift control
    print(f"\n[arms] {len(order)} arms, run order {order}", flush=True)

    results = []
    for n in order:
        results.append(run_arm(n, ARMS[n], train_ids, val_ids, vocab,
                               args, device, amp))
        Path(args.out).write_text(json.dumps(
            {"config": {k: v for k, v in vars(args).items() if k != "model_cfg"},
             "model_cfg": {k: list(v) if isinstance(v, tuple) else v
                           for k, v in args.model_cfg.items()},
             "vocab": vocab, "results": results}, indent=2))

    results.sort(key=lambda r: r["val_ppl"])
    ref = next((r for r in results if r["arm"] == "deltanet"), None)
    print("\n" + "=" * 91)
    print(f"{'arm':18s} {'train ppl':>10s} {'val ppl':>10s} {'val nats':>9s} "
          f"{'tok/s tr':>10s} {'tok/s inf':>10s} {'peak GB':>8s} {'non-emb par':>12s}")
    print("-" * 91)
    for r in results:
        print(f"{r['arm']:18s} {r['train_ppl']:10.2f} {r['val_ppl']:10.2f} "
              f"{r['val_loss']:9.4f} {r['train_tokens_per_sec']:10,.0f} "
              f"{r['infer_tokens_per_sec']:10,.0f} {r.get('peak_mem_gb',0):8.2f} "
              f"{r['params']['non_embedding']:12,}")
    print("=" * 91)

    base = max(r["params"]["non_embedding"] for r in results)
    for r in results:
        gap = abs(r["params"]["non_embedding"] - base) / base
        if gap > 0.05:
            print(f"!! {r['arm']}: non-embedding params {gap*100:.1f}% below the "
                  f"largest arm -- adjust d_v/d_a before drawing conclusions")
    if ref:
        print(f"\nvs deltanet rung (val nats, negative = better):")
        for r in results:
            if r["arm"] != "deltanet":
                print(f"  {r['arm']:18s} {r['val_loss'] - ref['val_loss']:+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
