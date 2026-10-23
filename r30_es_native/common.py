"""Shared data + utilities for the ES-native text-gen bench (char-level WikiText-2)."""
import numpy as np, os, time

DATA = os.path.join(os.path.dirname(__file__), "data")
_cache = {}

def load(n_train=None, n_val=200_000, vocab=96):
    key = (n_train, n_val, vocab)
    if key in _cache: return _cache[key]
    tr = open(os.path.join(DATA, "wt2_train.txt"), encoding="utf8").read()
    va = open(os.path.join(DATA, "wt2_valid.txt"), encoding="utf8").read()
    from collections import Counter
    cnt = Counter(tr)
    chars = [c for c, _ in cnt.most_common(vocab - 1)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}   # 0 = rare/unknown char
    enc = lambda s: np.frombuffer(bytes(0 for _ in range(0)), dtype=np.uint8) if not s else \
        np.array([stoi.get(c, 0) for c in s], dtype=np.int64)
    trn = enc(tr if n_train is None else tr[:n_train])
    val = enc(va[:n_val])
    itos = ['?'] + chars
    _cache[key] = (trn, val, itos)
    return _cache[key]

def contexts(stream, idx, k):
    """(B,k) previous-k token ids and (B,) targets for positions idx (idx >= k)."""
    offs = np.arange(-k, 0)
    return stream[idx[:, None] + offs], stream[idx]

def logsoftmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

def xent(logits, y):
    """per-row cross entropy in nats."""
    ls = logsoftmax(logits)
    return -ls[np.arange(len(y)), y]

class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps; self.wd = wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            if self.wd: g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)

LN2 = np.log(2.0)
class Timer:
    def __init__(self): self.t0 = time.time()
    def __call__(self): return time.time() - self.t0
