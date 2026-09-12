"""
Sequence mixers.

FAIRNESS NOTE -- read before changing anything here.

The point of this file is that the arms differ in *mechanism*, not in
implementation effort. Two rules are enforced:

 1. The two attention mixers share one implementation and differ only in the
    mask. Both use torch.nn.functional.scaled_dot_product_attention.
 2. The three recurrent mixers (GLA, DeltaNet, pattern-separated memory) share
    one chunked scaffold: the same chunk length, the same fp32 recurrence, the
    same intra-chunk parallel form. GLA and DeltaNet are literally the same
    function with one flag, so neither can be advantaged by better code.

Where the mechanisms genuinely differ in cost, that difference is real and
belongs in the tokens/sec number. Where they differ only because one got a
fused kernel, that is a confound -- hence rule 1 and 2.

The pattern-separated memory operates at chunk granularity: reads use the
state as of the start of the chunk, writes land at the end. That is a design
choice, not a shortcut -- intra-chunk (< 64 token) mixing is the sliding
window's job in the hybrid layout, and letting the memory duplicate it
contradicts the labour-division finding.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ helpers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(T, hd, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


# ------------------------------------------------------------------ attention
class Attention(nn.Module):
    """Causal multi-head attention. window=None -> full; window=w -> sliding."""

    def __init__(self, d_model, n_heads, window=None):
        super().__init__()
        self.h, self.hd, self.window = n_heads, d_model // n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self._mask_cache = {}

    def _mask(self, T, device):
        key = (T, device)
        if key not in self._mask_cache:
            i = torch.arange(T, device=device)
            m = i[:, None] >= i[None, :]
            if self.window is not None:
                m &= (i[:, None] - i[None, :]) < self.window
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v)]
        cos, sin = rope_cache(T, self.hd, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device))
        return self.o(y.transpose(1, 2).reshape(B, T, D))


# ------------------------------------------------------ shared chunked scaffold
def _chunked_linear_recurrence(q, k, v, log_alpha, beta, chunk, delta_rule):
    """
    One scaffold, two mechanisms.

      delta_rule=False -> gated linear attention
          S_i = a_i S_{i-1} + k_i v_i^T
      delta_rule=True  -> gated delta rule (DeltaNet)
          S_i = a_i S_{i-1} (I - b_i k_i k_i^T) + b_i k_i v_i^T

    Shapes: q,k (B,H,T,dk); v (B,H,T,dv); log_alpha,beta (B,H,T).
    Decay is folded out by rescaling v, which reduces the gated recursion to
    the ungated one exactly (see README). Run in fp32 for stability.
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    assert T % chunk == 0, "sequence length must be divisible by chunk"
    N = T // chunk
    q, k, v = q.float(), k.float(), v.float()
    log_alpha, beta = log_alpha.float(), beta.float()

    resh = lambda t, last: t.view(B, H, N, chunk, last)
    q, k, v = resh(q, dk), resh(k, dk), resh(v, dv)
    la = log_alpha.view(B, H, N, chunk)
    bt = beta.view(B, H, N, chunk)

    cum = la.cumsum(-1)                      # log A_i within chunk
    A = cum.exp()                            # (B,H,N,chunk)
    A_end = A[..., -1]                       # (B,H,N)
    v_hat = v / A.unsqueeze(-1).clamp_min(1e-6)

    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
    tri_strict = torch.ones(chunk, chunk, device=q.device, dtype=torch.bool).tril(-1)
    tri_incl = torch.ones(chunk, chunk, device=q.device, dtype=torch.bool).tril(0)
    out = torch.empty(B, H, N, chunk, dv, device=q.device, dtype=torch.float32)

    for n in range(N):
        qn, kn, vn = q[:, :, n], k[:, :, n], v_hat[:, :, n]
        An, bn = A[:, :, n], bt[:, :, n]

        if delta_rule:
            KK = (kn @ kn.transpose(-1, -2)) * tri_strict          # (B,H,C,C)
            M = torch.eye(chunk, device=q.device)[None, None] + bn.unsqueeze(-1) * KK
            rhs = bn.unsqueeze(-1) * (vn - kn @ S)
            U = torch.linalg.solve_triangular(M, rhs, upper=False, unitriangular=True)
        else:
            U = vn

        QK = (qn @ kn.transpose(-1, -2)) * tri_incl
        o = qn @ S + QK @ U                                        # (B,H,C,dv)
        out[:, :, n] = o * An.unsqueeze(-1)

        S = A_end[:, :, n, None, None] * (S + kn.transpose(-1, -2) @ U)

    return out.view(B, H, T, dv)


class LinearRecurrentMixer(nn.Module):
    """Gated linear attention (delta_rule=False) or gated DeltaNet (True)."""

    def __init__(self, d_model, n_heads, chunk=64, delta_rule=False,
                 max_log_decay=0.05):
        super().__init__()
        self.h, self.hd, self.chunk = n_heads, d_model // n_heads, chunk
        self.delta_rule, self.max_log_decay = delta_rule, max_log_decay
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.a = nn.Linear(d_model, n_heads, bias=True)
        self.b = nn.Linear(d_model, n_heads, bias=True)
        self.g = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(self.hd)
        nn.init.constant_(self.a.bias, 2.0)          # decay starts near 1
        nn.init.constant_(self.b.bias, 0.0)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        sh = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        log_alpha = -F.softplus(-self.a(x)).clamp(max=self.max_log_decay)
        log_alpha = log_alpha.transpose(1, 2)                       # (B,H,T)
        beta = torch.sigmoid(self.b(x)).transpose(1, 2)
        y = _chunked_linear_recurrence(q, k, v, log_alpha, beta,
                                       self.chunk, self.delta_rule)
        y = self.norm(y.to(x.dtype)).transpose(1, 2).reshape(B, T, D)
        return self.o(y * torch.sigmoid(self.g(x)))


# ---------------------------------------------- pattern-separated memory (3A)
class ProductAddress(nn.Module):
    """
    Fixed random sparse expansion + k-winner-take-all, in product form.

    Two frozen random half-codebooks E1, E2 of size (d_a/2, m). Candidate
    scores are z1_i + z2_j over the Cartesian product of the top-r of each
    half; the global top-k of those candidates is the address. Nothing here
    is learned -- that is the pattern-separation claim. Only the projection
    feeding it (W_a, outside this module) is learned.
    """

    def __init__(self, d_a, m_half, k, r=8, seed=0):
        super().__init__()
        assert d_a % 2 == 0
        g = torch.Generator().manual_seed(seed)
        half = d_a // 2
        e1 = torch.randn(half, m_half, generator=g) / math.sqrt(half)
        e2 = torch.randn(half, m_half, generator=g) / math.sqrt(half)
        self.register_buffer("E1", e1); self.register_buffer("E2", e2)
        self.m_half, self.k, self.r = m_half, k, max(r, int(math.ceil(k ** 0.5)))
        self.M = m_half * m_half

    def forward(self, a):
        B, T, d_a = a.shape
        a1, a2 = a.chunk(2, -1)
        z1, z2 = a1 @ self.E1, a2 @ self.E2                  # (B,T,m_half)
        v1, i1 = z1.topk(self.r, -1)
        v2, i2 = z2.topk(self.r, -1)
        cand = v1.unsqueeze(-1) + v2.unsqueeze(-2)           # (B,T,r,r)
        idx = i1.unsqueeze(-1) * self.m_half + i2.unsqueeze(-2)
        cand, idx = cand.flatten(-2), idx.flatten(-2)        # (B,T,r*r)
        val, sel = cand.topk(self.k, -1)
        index = idx.gather(-1, sel)                          # (B,T,k)
        w = F.relu(val)
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        return index, w


class PatternSeparatedMemory(nn.Module):
    """
    Compartmented episodic memory with a cerebellar delta-rule write.

    Per compartment n: address on the last n hidden states, read the k
    addressed rows with divisive normalisation by row usage, write the
    prediction error back into the same k rows. Compartments compete by match
    strength plus a learned priority bias, so the longest matching context
    wins -- a differentiable longest-prefix match.
    """

    def __init__(self, d_model, d_a=256, d_v=128, m_half=91, k=32,
                 orders=(1, 2, 4, 8), chunk=64, seed=0, surprise_gate=True):
        super().__init__()
        self.orders, self.C, self.chunk = tuple(orders), len(orders), chunk
        self.d_v, self.k, self.surprise_gate = d_v, k, surprise_gate

        self.ctx = nn.ModuleList([                     # causal depthwise conv
            nn.Conv1d(d_model, d_model, n, groups=d_model, padding=n - 1, bias=False)
            if n > 1 else nn.Identity() for n in self.orders])
        self.addr_proj = nn.ModuleList(
            [nn.Linear(d_model, d_a, bias=False) for _ in self.orders])
        self.address = ProductAddress(d_a, m_half, k, seed=seed)
        self.M = self.address.M

        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.read_gate = nn.Linear(d_model, self.C, bias=True)
        self.out = nn.Linear(self.C * d_v, d_model, bias=False)
        self.norm = RMSNorm(d_v)

        self.beta = nn.Parameter(torch.zeros(1))                       # write rate
        self.log_decay = nn.Parameter(torch.full((self.C,), -6.9))     # ~0.999
        self.priority = nn.Parameter(torch.arange(self.C).float() * 0.5)
        self.match_scale = nn.Parameter(torch.ones(1))

    def _context(self, x, i):
        if isinstance(self.ctx[i], nn.Identity):
            return x
        n = self.orders[i]
        y = self.ctx[i](x.transpose(1, 2))[..., :-(n - 1)]
        return y.transpose(1, 2)

    def forward(self, x, surprise=None, modul=None):
        B, T, D = x.shape
        dev, dt = x.device, x.dtype
        assert T % self.chunk == 0
        N, Ck = T // self.chunk, self.chunk

        idxs, ws = [], []
        for i in range(self.C):
            a = self.addr_proj[i](self._context(x, i))
            if modul is not None:                 # acetylcholine: code sharpness
                a = a * modul[..., 1:2]
            index, w = self.address(a)
            idxs.append(index); ws.append(w)

        v = self.v_proj(x).float()                                   # (B,T,d_v)
        gamma = torch.sigmoid(self.beta).expand(B, T).clone()
        if self.surprise_gate and surprise is not None:
            gamma = gamma * surprise
        if modul is not None:                     # dopamine: write rate
            gamma = gamma * modul[..., 0]
        gamma = gamma.float()

        V = [torch.zeros(B, self.M, self.d_v, device=dev, dtype=torch.float32)
             for _ in range(self.C)]
        U = [torch.zeros(B, self.M, device=dev, dtype=torch.float32)
             for _ in range(self.C)]
        decay = torch.sigmoid(self.log_decay).float()
        if modul is not None:                     # noradrenaline: flush rate
            pass                                  # applied per chunk below

        reads = [torch.empty(B, T, self.d_v, device=dev, dtype=torch.float32)
                 for _ in range(self.C)]
        match = torch.empty(B, T, self.C, device=dev, dtype=torch.float32)
        boff = torch.arange(B, device=dev)[:, None, None] * self.M

        for n in range(N):
            sl = slice(n * Ck, (n + 1) * Ck)
            for i in range(self.C):
                index, w = idxs[i][:, sl], ws[i][:, sl].float()      # (B,Ck,k)
                flat = (index + boff).reshape(-1)                    # (B*Ck*k,)
                Vf = V[i].reshape(B * self.M, self.d_v)
                Uf = U[i].reshape(B * self.M)

                rows = Vf.index_select(0, flat).view(B, Ck, self.k, self.d_v)
                use = Uf.index_select(0, flat).view(B, Ck, self.k)
                m = (w * use).sum(-1)                                # match strength
                r = (w.unsqueeze(-1) * rows).sum(-2) / (m.unsqueeze(-1) + 1e-4)
                reads[i][:, sl] = r
                match[:, sl, i] = m

                # ---- delta-rule write, against the chunk-start state ----
                err = v[:, sl] - r                                   # (B,Ck,d_v)
                g = gamma[:, sl].unsqueeze(-1)                       # (B,Ck,1)
                upd = (g * w).unsqueeze(-1) * err.unsqueeze(-2)      # (B,Ck,k,d_v)
                Vf.index_add_(0, flat, upd.reshape(-1, self.d_v))
                Uf.index_add_(0, flat, (g * w).reshape(-1))

                lam = decay[i] ** Ck
                if modul is not None:
                    lam = lam * modul[:, sl, 2].mean(1).view(B, 1, 1)
                    V[i] = V[i] * lam; U[i] = U[i] * lam.squeeze(-1)
                else:
                    V[i] = V[i] * lam; U[i] = U[i] * lam

        w_cmp = torch.softmax(self.match_scale * match + self.priority, -1)
        gate = torch.sigmoid(self.read_gate(x)).float()
        parts = [self.norm(reads[i].to(dt)).float() *
                 (w_cmp[..., i] * gate[..., i]).unsqueeze(-1) for i in range(self.C)]
        y = self.out(torch.cat(parts, -1).to(dt))
        if modul is not None:                     # global gain
            y = y * modul[..., 3:4].to(dt)
        return y


# ----------------------------------------------------- short-term depression (3B)
class ShortTermDepression(nn.Module):
    """
    Tsodyks-Markram resource dynamics as a content-dependent gain.

        x_t = x_{t-1} + (1 - x_{t-1})/tau - U_t * act_{t-1} * x_{t-1}

    Applied to output logits with act = previous-step probability. Suppresses
    exactly what was just emitted, in proportion to how confidently it was
    emitted, and can learn to disengage where repetition is correct.
    """

    def __init__(self, d_model, tau_init=8.0, u_init=0.2):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.logit_u = nn.Parameter(torch.tensor(math.log(u_init / (1 - u_init))))
        self.w_u = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.w_u.weight); nn.init.zeros_(self.w_u.bias)

    def forward(self, logits, h):
        B, T, Vv = logits.shape
        tau = self.log_tau.exp().clamp(min=1.0)
        U = torch.sigmoid(self.logit_u) * torch.sigmoid(self.w_u(h))   # (B,T,1)
        p = torch.softmax(logits.float(), -1)
        # teacher-forced: act_{t-1} is the previous position's distribution
        act = torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], 1)
        res = torch.ones(B, Vv, device=logits.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            res = res + (1.0 - res) / tau - U[:, t] * act[:, t] * res
            res = res.clamp(0.05, 1.0)
            outs.append(res)
        x = torch.stack(outs, 1)
        return logits + torch.log(x.to(logits.dtype) + 1e-6)


# ------------------------------------------------------- neuromodulators (3C)
class Neuromodulators(nn.Module):
    """Four global scalars per token, broadcast to every memory layer."""

    def __init__(self, d_model, ema=0.98):
        super().__init__()
        self.ema = ema
        self.proj = nn.Linear(d_model + 1, 4)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, h, nll_prev):
        B, T, D = h.shape
        a = torch.arange(T, device=h.device).float()
        w = self.ema ** (a[:, None] - a[None, :])
        w = torch.tril(w) * (1 - self.ema)
        s = w @ h.float()                                  # causal EMA summary
        z = torch.cat([s, nll_prev.unsqueeze(-1).float()], -1)
        return 2.0 * torch.sigmoid(self.proj(z.to(h.dtype))).float()
