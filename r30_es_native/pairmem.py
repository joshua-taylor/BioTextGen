"""Pair index for the hard lookup: bucket = a PAIR of the query's top-p firing units.

Single-unit posting lists grow with memory size (each unit fires in ~kw/H of all moments), so their cost per
lookup grows with the memory. A pair of top units is shared by far fewer moments, so a fixed-size bucket covers a
much longer history at constant cost per lookup. Candidates found through shared pairs are re-ranked by the cosine
of their stored top-p codes. Interface matches sb2.Memory.query_write.
"""
import numpy as np
from numba import njit

NB = 4

@njit(cache=True)
def _pair_qw(act, val, nxt, gid0, P, bucket, bhead, bcnt, ent_gid, ent_next, ent_i, ent_v, ent_norm, score, touched,
             horizon, V, out_r, out_cos, do_query):
    T, k = act.shape; NBK, bcap = bucket.shape; M = ent_gid.shape[0]
    ti = np.empty(P, np.int64); tv = np.empty(P, np.float64)
    for t in range(T):
        g = gid0 + t
        # top-P units of this code
        for q in range(P):
            ti[q] = -1; tv[q] = 0.0
        for u in range(k):
            v = val[t, u]
            if v <= tv[P - 1]: continue
            m = P - 1
            while m > 0 and v > tv[m - 1]:
                tv[m] = tv[m - 1]; ti[m] = ti[m - 1]; m -= 1
            tv[m] = v; ti[m] = act[t, u]
        nrm = 0.0
        for q in range(P):
            nrm += tv[q] * tv[q]
        nrm = np.sqrt(nrm) + 1e-6
        best = -1; bestc = 0.0
        if do_query:
            nt = 0
            for a in range(P):
                if ti[a] < 0: continue
                for b in range(a + 1, P):
                    if ti[b] < 0: continue
                    x = ti[a]; y = ti[b]
                    if x > y: x, y = y, x
                    h = ((x * 1000003) ^ (y * 2654435761)) & (NBK - 1)
                    for q in range(bcnt[h]):
                        e = bucket[h, q]
                        slot = e % M
                        if ent_gid[slot] != e or g - e > horizon: continue
                        if score[slot] == 0.0:
                            touched[nt] = slot; nt += 1
                        score[slot] += 1.0
            for i in range(nt):
                slot = touched[i]; score[slot] = 0.0
                d = 0.0
                for a in range(P):
                    if ti[a] < 0: continue
                    for b in range(P):
                        if ent_i[slot, b] == ti[a]: d += tv[a] * ent_v[slot, b]
                cs = d / (nrm * ent_norm[slot])
                if cs > bestc: bestc = cs; best = slot
        if best < 0 or bestc <= 0:
            out_r[t, 0] = NB * (V + 1); out_cos[t, 0] = 0.0
        else:
            bb = int(bestc * NB)
            if bb > NB - 1: bb = NB - 1
            out_r[t, 0] = bb * (V + 1) + ent_next[best]; out_cos[t, 0] = bestc
        # write
        slot = g % M
        ent_gid[slot] = g; ent_next[slot] = nxt[t]; ent_norm[slot] = nrm
        for q in range(P):
            ent_i[slot, q] = ti[q]; ent_v[slot, q] = tv[q]
        for a in range(P):
            if ti[a] < 0: continue
            for b in range(a + 1, P):
                if ti[b] < 0: continue
                x = ti[a]; y = ti[b]
                if x > y: x, y = y, x
                h = ((x * 1000003) ^ (y * 2654435761)) & (NBK - 1)
                if bcnt[h] < bcap:
                    bucket[h, bcnt[h]] = g; bcnt[h] += 1
                else:
                    bucket[h, bhead[h]] = g; bhead[h] = (bhead[h] + 1) % bcap

class PairMemory:
    def __init__(self, horizon, P=8, nbk_log2=18, bcap=32):
        M = 1
        while M < horizon + 2: M *= 2
        self.P = P; self.horizon = horizon
        self.bucket = np.zeros((1 << nbk_log2, bcap), np.int32); self.bhead = np.zeros(1 << nbk_log2, np.int64)
        self.bcnt = np.zeros(1 << nbk_log2, np.int64)
        self.gid = np.full(M, -1, np.int64); self.next = np.zeros(M, np.int64); self.norm = np.ones(M)
        self.ei = np.full((M, P), -1, np.int64); self.ev = np.zeros((M, P)); self.score = np.zeros(M)
        self.touched = np.zeros(M, np.int64); self.g = 0
    def query_write(self, act, val, nxt, V, topm, do_query=True):
        T, k = act.shape; r = np.empty((T, 1), np.int64); c = np.empty((T, 1))
        _pair_qw(act.astype(np.int64), val.astype(np.float64), nxt.astype(np.int64), self.g, self.P, self.bucket,
                 self.bhead, self.bcnt, self.gid, self.next, self.ei, self.ev, self.norm, self.score, self.touched,
                 self.horizon, V, r, c, do_query)
        self.g += T
        return r, c, np.zeros((T, k), np.int64), np.zeros((T, k))
