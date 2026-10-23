"""Fast one-layer sparse (k-WTA) net with row-sparse ('lazy') Adam, for many quick experiments.

Credit for the active units is the exact per-unit finite-difference credit (e2b rule 'fdo'), computed here in
closed form for speed: dz_j = Wo[j] . (softmax - onehot). Use fdo() to verify it with forward passes only.
"""
import numpy as np
from common import logsoftmax, xent, LN2

class LazyAdam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape, np.float32); self.v = np.zeros(shape, np.float32)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
    def rows(self, W, rows, g):
        """g: (len(rows), C) gradient for the given unique rows."""
        self.t += 1; b1, b2 = self.b1, self.b2
        m = self.m[rows] * b1 + (1 - b1) * g; v = self.v[rows] * b2 + (1 - b2) * g * g
        self.m[rows] = m; self.v[rows] = v
        W[rows] -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + self.eps)

class Sparse1:
    def __init__(self, nrows, H, kw, V, lr=3e-3, seed=0, tscale=0.1):
        rng = np.random.default_rng(seed)
        self.T = (rng.normal(0, 1, (nrows, H)) * tscale).astype(np.float32)
        self.b0 = np.zeros((1, H), np.float32)
        self.Wo = np.zeros((H, V), np.float32); self.bo = np.zeros((1, V), np.float32)
        self.H, self.kw, self.V = H, kw, V
        self.oT, self.ob0, self.oWo, self.obo = (LazyAdam(x.shape, lr) for x in (self.T, self.b0, self.Wo, self.bo))
    def forward(self, F):
        z = self.T[F].sum(1) + self.b0[0]
        idx = np.argpartition(z, -self.kw, axis=1)[:, -self.kw:]
        a = np.maximum(np.take_along_axis(z, idx, 1), 0)
        logits = np.einsum("bk,bkv->bv", a, self.Wo[idx]) + self.bo[0]
        return idx, a, logits
    def loss(self, F, y):
        return xent(self.forward(F)[2], y)
    def credit(self, idx, a, logits, y):
        B = len(y); D = np.exp(logsoftmax(logits)); D[np.arange(B), y] -= 1; D /= B
        dz = np.einsum("bv,bkv->bk", D, self.Wo[idx]) * (a > 0)
        return D, dz
    def fdo(self, idx, a, y, sigma=0.05):
        """forward-only exact credit: perturb each active unit of every token, one at a time."""
        B = len(y); dz = np.zeros_like(a); base = np.einsum("bk,bkv->bv", a, self.Wo[idx]) + self.bo[0]
        for j in range(self.kw):
            w = self.Wo[idx[:, j]]
            lp = xent(base + sigma * w, y); lm = xent(base - sigma * w, y)
            dz[:, j] = (lp - lm) / (2 * sigma) / B * (a[:, j] > 0)
        return dz
    def np_credit(self, idx, a, y, K, sigma=0.05, rng=None):
        """per-token random node perturbation (K antithetic draws per token): the evolution-style estimate."""
        B = len(y); dz = np.zeros_like(a); base = np.einsum("bk,bkv->bv", a, self.Wo[idx]) + self.bo[0]
        for _ in range(K):
            xi = rng.standard_normal(a.shape).astype(np.float32) * (a > 0)
            dl = np.einsum("bk,bkv->bv", xi, self.Wo[idx]) * sigma
            c = (xent(base + dl, y) - xent(base - dl, y)) / (2 * sigma)
            dz += c[:, None] * xi / (B * K)
        return dz
    def step(self, F, y, exact_fd=False, K=0, rng=None):
        idx, a, logits = self.forward(F)
        D, dz = self.credit(idx, a, logits, y)
        if exact_fd: dz = self.fdo(idx, a, y)
        if K: dz = self.np_credit(idx, a, y, K, rng=rng)
        H, V = self.H, self.V; B, nf = F.shape
        # readout rows
        u, inv = np.unique(idx, return_inverse=True); inv = inv.reshape(idx.shape)
        gWo = np.zeros((len(u), V), np.float32)
        np.add.at(gWo, inv.reshape(-1), (a[:, :, None] * D[:, None, :]).reshape(-1, V))
        self.oWo.rows(self.Wo, u, gWo)
        self.obo.rows(self.bo, np.array([0]), D.sum(0, keepdims=True))
        # table rows (only the active columns are non-zero)
        ur, invr = np.unique(F, return_inverse=True); invr = invr.reshape(F.shape)
        flat = (invr[:, :, None] * H + idx[:, None, :]).reshape(-1)
        gT = np.bincount(flat, np.repeat(dz[:, None, :], nf, 1).reshape(-1), minlength=len(ur) * H).reshape(len(ur), H)
        self.oT.rows(self.T, ur, gT.astype(np.float32))
        gb = np.bincount(idx.reshape(-1), dz.reshape(-1), minlength=H)[None].astype(np.float32)
        self.ob0.rows(self.b0, np.array([0]), gb)
        return xent(logits, y)
