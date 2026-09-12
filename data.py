"""
WikiText-103 subset loading and tokenisation.

Every arm in the benchmark sees exactly the same token stream in exactly the
same order, so data is never a confound. The tokenizer is trained once on the
training subset and cached; all arms reuse the cache.
"""
import os, json, hashlib
from pathlib import Path
import numpy as np

CACHE = Path(os.environ.get("BENCH_CACHE", "./cache"))


# ---------------------------------------------------------------- raw text
def _from_hf(n_train_chars, n_val_chars):
    from datasets import load_dataset
    tr = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
    va = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    buf, tot = [], 0
    for row in tr:
        t = row["text"]
        if not t.strip():
            continue
        buf.append(t); tot += len(t)
        if tot >= n_train_chars:
            break
    train = "".join(buf)[:n_train_chars]
    val = "".join(r["text"] for r in va if r["text"].strip())[:n_val_chars]
    return train, val


def _from_kaggle_input(n_train_chars, n_val_chars):
    import glob
    files = sorted(glob.glob("/kaggle/input/**/wiki*.*", recursive=True) +
                   glob.glob("/kaggle/input/**/*.raw", recursive=True) +
                   glob.glob("/kaggle/input/**/*.txt", recursive=True),
                   key=os.path.getsize, reverse=True)
    if not files:
        raise FileNotFoundError("no wikitext under /kaggle/input")
    tr = [f for f in files if "train" in f.lower()] or files[:1]
    va = [f for f in files if "valid" in f.lower()] or files[:1]
    train = open(tr[0], encoding="utf8", errors="ignore").read()[:n_train_chars]
    val = open(va[0], encoding="utf8", errors="ignore").read()
    if va[0] == tr[0]:                       # same file: take a disjoint tail
        val = val[n_train_chars: n_train_chars + n_val_chars]
    else:
        val = val[:n_val_chars]
    return train, val


def _from_local(n_train_chars, n_val_chars):
    p = os.environ.get("BENCH_TEXT")
    if not p or not os.path.exists(p):
        raise FileNotFoundError("BENCH_TEXT not set or missing")
    s = open(p, encoding="utf8", errors="ignore").read()
    return s[:n_train_chars], s[n_train_chars:n_train_chars + n_val_chars]


def raw_text(n_train_chars, n_val_chars):
    for name, fn in [("huggingface", _from_hf), ("/kaggle/input", _from_kaggle_input),
                     ("BENCH_TEXT", _from_local)]:
        try:
            tr, va = fn(n_train_chars, n_val_chars)
            print(f"  corpus from {name}: {len(tr):,} train chars, {len(va):,} val chars")
            return tr, va
        except Exception as e:
            print(f"  {name}: unavailable ({type(e).__name__}: {e})")
    raise SystemExit(
        "No WikiText source. Enable internet, or attach a wikitext dataset "
        "to the Kaggle notebook."
    )


# ---------------------------------------------------------------- tokenise
def _train_bpe(text, vocab_size, path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"])
    tok.train_from_iterator([text[i:i + 100000]
                             for i in range(0, len(text), 100000)], trainer)
    tok.save(str(path))
    return tok


def get_data(n_train_chars=30_000_000, n_val_chars=1_000_000, vocab_size=8192):
    """Returns (train_ids, val_ids, vocab_size). int32 numpy arrays."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{n_train_chars}-{n_val_chars}-{vocab_size}".encode()).hexdigest()[:12]
    tr_p, va_p, meta_p = (CACHE / f"train-{key}.npy", CACHE / f"val-{key}.npy",
                          CACHE / f"meta-{key}.json")
    if tr_p.exists() and va_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        print(f"  cache hit: {meta['n_train']:,} train tokens, "
              f"{meta['n_val']:,} val tokens, vocab {meta['vocab_size']}")
        return np.load(tr_p), np.load(va_p), meta["vocab_size"]

    train_txt, val_txt = raw_text(n_train_chars, n_val_chars)

    try:
        tok_p = CACHE / f"bpe-{key}.json"
        tok = _train_bpe(train_txt, vocab_size, tok_p)
        enc = lambda s: np.array(tok.encode(s).ids, np.int32)
        V = tok.get_vocab_size()
        print(f"  trained byte-level BPE, vocab {V}")
    except Exception as e:
        print(f"  tokenizers unavailable ({type(e).__name__}); falling back to bytes")
        enc = lambda s: np.frombuffer(s.encode("utf8"), np.uint8).astype(np.int32)
        V = 256

    # encode in slices to bound memory
    def encode_all(s):
        return np.concatenate([enc(s[i:i + 1_000_000])
                               for i in range(0, len(s), 1_000_000)])

    tr, va = encode_all(train_txt), encode_all(val_txt)
    np.save(tr_p, tr); np.save(va_p, va)
    meta_p.write_text(json.dumps(
        {"n_train": int(len(tr)), "n_val": int(len(va)), "vocab_size": int(V)}))
    print(f"  tokenised: {len(tr):,} train, {len(va):,} val, vocab {V}")
    return tr, va, V


class Batcher:
    """
    Deterministic batch sampler. Given a seed, produces an identical sequence
    of batches for every arm -- this is a fairness control, not a convenience.
    """

    def __init__(self, ids, batch_size, seq_len, seed=0):
        self.ids = ids; self.B = batch_size; self.T = seq_len
        self.rng = np.random.default_rng(seed)
        self.n = len(ids) - seq_len - 1

    def batch(self):
        import torch
        i = self.rng.integers(0, self.n, self.B)
        x = np.stack([self.ids[j:j + self.T] for j in i]).astype(np.int64)
        y = np.stack([self.ids[j + 1:j + 1 + self.T] for j in i]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def sequential(self, n_batches):
        """Non-overlapping sequential windows, for deterministic evaluation."""
        import torch
        step = self.B * self.T
        for b in range(n_batches):
            o = b * step
            if o + step + 1 > len(self.ids):
                return
            chunk = self.ids[o:o + step + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1].reshape(self.B, self.T).copy())
            y = torch.from_numpy(chunk[1:].reshape(self.B, self.T).copy())
            yield x, y
