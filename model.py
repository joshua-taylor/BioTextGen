"""
Shared trunk. Every arm uses the SAME embedding, norms, FFN, depth, width,
initialisation and tied unembedding. The only thing that changes between arms
is which mixer occupies the four "memory slot" blocks.

Layout (12 blocks):
    blocks 1,2,4,5,7,8,10,11  sliding-window attention (64) + SwiGLU
    blocks 3,6,9,12           MEMORY SLOT + SwiGLU

The 'sliding' arm fills the memory slots with another sliding-window block,
so it is the true "no long-range mechanism" control rather than a different
architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixers import (RMSNorm, Attention, LinearRecurrentMixer,
                    PatternSeparatedMemory, ShortTermDepression, Neuromodulators)

def memory_slots(n_layers, n_slots=4):
    """Evenly spaced memory slots, last block always one. Adapts to depth."""
    step = max(n_layers // n_slots, 1)
    return tuple(sorted({min(n_layers - 1, (i + 1) * step - 1)
                         for i in range(n_slots)}))


MEMORY_SLOTS = memory_slots(12)       # module-level default; tests override


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        hidden = int(2 * mult * d / 3 / 64 + 0.5) * 64
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, mixer, is_memory=False):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mixer, self.ffn, self.is_memory = mixer, SwiGLU(d), is_memory

    def forward(self, x, surprise=None, modul=None):
        if self.is_memory:
            x = x + self.mixer(self.n1(x), surprise=surprise, modul=modul)
        else:
            x = x + self.mixer(self.n1(x))
        return x + self.ffn(self.n2(x))


def build_mixer(kind, cfg):
    d, h = cfg["d_model"], cfg["n_heads"]
    if kind == "sliding":
        return Attention(d, h, window=cfg["window"]), False
    if kind == "attention":
        return Attention(d, h, window=None), False
    if kind == "gla":
        return LinearRecurrentMixer(d, h, cfg["chunk"], delta_rule=False), False
    if kind == "deltanet":
        return LinearRecurrentMixer(d, h, cfg["chunk"], delta_rule=True), False
    if kind.startswith("psm"):
        orders = (1,) if kind == "psm_single" else cfg["orders"]
        return PatternSeparatedMemory(
            d, d_a=cfg["d_a"], d_v=cfg["d_v"], m_half=cfg["m_half"],
            k=cfg["k"], orders=orders, chunk=cfg["chunk"],
            surprise_gate=cfg["surprise_gate"]), True
    raise ValueError(f"unknown mixer {kind}")


class Model(nn.Module):
    def __init__(self, vocab, memory_mixer="sliding", cfg=None,
                 use_depression=False, use_modulators=False):
        super().__init__()
        self.cfg = cfg
        d, L = cfg["d_model"], cfg["n_layers"]
        self.slots = memory_slots(L) if MEMORY_SLOTS == memory_slots(12) else MEMORY_SLOTS
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList()
        self.any_memory = False
        for i in range(L):
            if i in self.slots:
                mixer, is_mem = build_mixer(memory_mixer, cfg)
            else:
                mixer, is_mem = build_mixer("sliding", cfg)
            self.any_memory |= is_mem
            self.blocks.append(Block(d, mixer, is_mem))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight               # tied
        self.depression = ShortTermDepression(d) if use_depression else None
        self.modulators = Neuromodulators(d) if use_modulators else None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.abs().sum() == 0:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, y=None, prev_nll=None):
        h = self.emb(x)
        surprise = prev_nll if (prev_nll is not None and self.any_memory) else None
        modul = None
        if self.modulators is not None:
            z = prev_nll if prev_nll is not None else torch.zeros_like(h[..., 0])
            modul = self.modulators(h, z)
        for b in self.blocks:
            h = b(h, surprise=surprise, modul=modul)
        h = self.norm(h)
        logits = self.head(h)
        if self.depression is not None:
            logits = self.depression(logits, h)
        if y is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               y.reshape(-1))
        return logits, loss

    def param_report(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.emb.weight.numel()
        mixer = sum(p.numel() for i in self.slots
                    for p in self.blocks[i].mixer.parameters())
        return {"total": total, "non_embedding": total - emb,
                "memory_slot_mixers": mixer}
