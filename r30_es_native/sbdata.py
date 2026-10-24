"""Precomputed feature pool for the fast sparse bench (sb.py).

Stream convention: position p holds the last known char s[p]; the target is s[p+1].
Features (row id inside the feature's own block):
  c_o   : s[p-o+1]                                  (V rows)
  ng_n  : hash of s[p-n+1..p]                       (R rows)
  sk_a_b: hash of (s[p-a+1], s[p-b+1])              (R rows)
  wd    : hash of the current word so far           (R rows)
  pw    : hash of the previous whole word           (R rows)
  ex_n  : char after the most recent earlier occurrence (within 1024 chars) of s[p-n+1..p], else V (V+1 rows)
"""
import numpy as np, os
from numba import njit
from common import load

R = 1 << 14
DIR = os.path.join(os.path.dirname(__file__), "sbcache")
EX_ORDERS = (2, 3, 4, 5, 6, 8, 11, 12, 16)

@njit(cache=True)
def _words(s, space):
    n = len(s); wd = np.zeros(n, np.int64); pw = np.zeros(n, np.int64); cur = 0; prev = 0
    for i in range(n):
        c = s[i]
        if c == space:
            prev = cur; cur = 0
        else:
            cur = (cur * 1000003 + c + 1) % 2305843009213693951
        wd[i] = cur; pw[i] = prev
    return wd, pw

@njit(cache=True)
def _exact(s, n, V, horizon):
    m = len(s); out = np.full(m, V, np.uint8)
    d = dict()
    d[np.int64(0)] = np.int64(0)
    for p in range(n - 1, m - 1):
        h = np.int64(0)
        for o in range(n):
            h = (h * 131 + s[p - o] + 1) % 2305843009213693951
        if h in d:
            q = d[h]
            if p - q <= horizon:
                out[p] = s[q + 1]
        d[h] = p
    return out

def build(ntrain=4_000_000, nval=60_000):
    os.makedirs(DIR, exist_ok=True)
    trn, val, itos = load(None)
    V = len(itos); space = itos.index(' ')
    for name, s in (("trn", trn[:ntrain]), ("val", val[:nval])):
        s = np.ascontiguousarray(s.astype(np.int64))
        np.save(f"{DIR}/{name}_s.npy", s.astype(np.uint8))
        wd, pw = _words(s, space)
        np.save(f"{DIR}/{name}_wd.npy", (wd * 2654435761 % R).astype(np.int32))
        np.save(f"{DIR}/{name}_pw.npy", (pw * 2654435761 % R).astype(np.int32))
        for n in EX_ORDERS:
            np.save(f"{DIR}/{name}_ex{n}.npy", _exact(s, n, V, 1024))
    np.save(f"{DIR}/meta.npy", np.array([V, space]))

_mm = {}
def get(name, key):
    k = (name, key)
    if k not in _mm: _mm[k] = np.load(f"{DIR}/{name}_{key}.npy", mmap_mode="r")
    return _mm[k]

def meta():
    V, space = np.load(f"{DIR}/meta.npy"); return int(V), int(space)

def block_size(f, V):
    t = f[0]
    if t == "c": return V
    if t == "ex": return V + 1
    return R

def feature_rows(feats, split, pos, V):
    """pos: (B,) int positions (>= 16). returns (B, nf) global row ids and total rows."""
    s = get(split, "s")
    cols = []; off = 0
    for f in feats:
        t = f[0]
        if t == "c":
            col = s[pos - (f[1] - 1)].astype(np.int64)
        elif t == "ng":
            h = np.zeros(len(pos), np.int64)
            for o in range(f[1]): h = (h * 1000003 + s[pos - o].astype(np.int64) + 1) % (1 << 61)
            col = (h * 2654435761 + 7 * f[1]) % R
        elif t == "sk":
            a, b = f[1], f[2]
            col = ((s[pos - (a - 1)].astype(np.int64) * 131 + s[pos - (b - 1)] + 977 * (a * 17 + b)) * 2654435761) % R
        elif t in ("wd", "pw"):
            col = get(split, t)[pos].astype(np.int64)
        elif t == "ex":
            col = get(split, f"ex{f[1]}")[pos].astype(np.int64)
        else:
            raise ValueError(f)
        cols.append(off + col); off += block_size(f, V)
    return np.stack(cols, 1), off

if __name__ == "__main__":
    import time; t = time.time(); build(); print("built in", round(time.time() - t, 1), "s")
