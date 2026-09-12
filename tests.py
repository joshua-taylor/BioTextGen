"""
Correctness tests. Run these before trusting any benchmark number.

The benchmark claims to compare mechanisms, not implementations. That claim
only holds if every mixer is (a) mathematically what it says it is and (b)
strictly causal. These tests check both.

    python tests.py
"""
import torch
import torch.nn.functional as F

from bench import DEFAULT_CFG, SIZES
import model as M
from mixers import _chunked_linear_recurrence


def test_chunked_matches_sequential():
    """The chunked parallel form must equal the naive token-by-token recurrence."""
    torch.manual_seed(0)
    B, H, T, dk, dv, C = 2, 3, 24, 8, 8, 4
    q = torch.randn(B, H, T, dk)
    k = F.normalize(torch.randn(B, H, T, dk), dim=-1)
    v = torch.randn(B, H, T, dv)
    la = -F.softplus(torch.randn(B, H, T)).clamp(max=0.05)
    be = torch.sigmoid(torch.randn(B, H, T))

    def reference(delta):
        out = torch.zeros(B, H, T, dv)
        S = torch.zeros(B, H, dk, dv)
        for t in range(T):
            a = la[:, :, t].exp()[:, :, None, None]
            kt, vt = k[:, :, t], v[:, :, t]
            bt = be[:, :, t][:, :, None]
            if delta:
                S = a * (S - bt.unsqueeze(-1) * kt.unsqueeze(-1)
                         * (kt.unsqueeze(-2) @ S)) \
                    + bt.unsqueeze(-1) * kt.unsqueeze(-1) * vt.unsqueeze(-2)
            else:
                S = a * S + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            out[:, :, t] = (q[:, :, t].unsqueeze(-2) @ S).squeeze(-2)
        return out

    for delta in (False, True):
        got = _chunked_linear_recurrence(q, k, v, la, be, C, delta)
        err = (got - reference(delta)).abs().max().item()
        name = "deltanet" if delta else "gla"
        assert err < 1e-4, f"{name} chunked != sequential ({err})"
        print(f"  {name:10s} chunked == sequential   max abs err {err:.2e}")


def test_causality():
    """
    Perturbing token t must not change any output at position < t.
    Catches leakage in masks, convolution padding, and memory read/write order.
    """
    cfg = dict(DEFAULT_CFG)
    cfg.update(d_model=96, n_layers=4, n_heads=3, m_half=17, k=8,
               d_a=32, d_v=32, chunk=8)
    M.MEMORY_SLOTS = (1, 3)
    T, t_pert, V = 32, 20, 200
    x = torch.randint(0, V, (1, T))

    for mix, dep, mod in [("sliding", False, False), ("attention", False, False),
                          ("gla", False, False), ("deltanet", False, False),
                          ("psm_single", False, False), ("psm", False, False),
                          ("psm", True, True)]:
        torch.manual_seed(0)
        m = M.Model(V, memory_mixer=mix, cfg=cfg,
                    use_depression=dep, use_modulators=mod).eval()
        prev = torch.rand(1, T)
        with torch.no_grad():
            a = m(x, prev_nll=prev)
            x2 = x.clone(); x2[0, t_pert] = (x[0, t_pert] + 7) % V
            b = m(x2, prev_nll=prev)
        leak = (a[:, :t_pert] - b[:, :t_pert]).abs().max().item()
        tag = f"{mix}{'+dep' if dep else ''}{'+mod' if mod else ''}"
        assert leak < 1e-4, f"{tag} leaks future information ({leak})"
        print(f"  {tag:16s} causal                 max leak {leak:.2e}")


def test_address_separation():
    """
    Pattern separation sanity: similar inputs should share more active address
    units than dissimilar ones, and overlap should fall off steeply.
    """
    from mixers import ProductAddress
    torch.manual_seed(0)
    addr = ProductAddress(d_a=64, m_half=31, k=16)
    base = torch.randn(1, 1, 64)
    i0, _ = addr(base)
    s0 = set(i0.flatten().tolist())
    print("  address overlap vs input distance:")
    prev = None
    for noise in (0.0, 0.1, 0.5, 1.0, 2.0):
        pert = base + noise * torch.randn_like(base)
        i1, _ = addr(pert)
        ov = len(s0 & set(i1.flatten().tolist())) / addr.k
        print(f"    noise {noise:4.1f}  overlap {ov:.2f}")
        if prev is not None:
            assert ov <= prev + 0.15, "overlap should not increase with distance"
        prev = ov


def test_decay_equivalence():
    """
    PSM folds uniform row decay into the write scale instead of rescaling the
    state, which stops autograd saving a copy of V per chunk. The two forms
    must be numerically identical.
    """
    torch.manual_seed(0)
    B, Mrows, dv, k, Ck, N = 2, 64, 8, 4, 4, 6
    lam = torch.tensor(0.93)
    idx = torch.randint(0, Mrows, (N, B, Ck, k))
    w = torch.rand(N, B, Ck, k); w = w / w.sum(-1, keepdim=True)
    upd = torch.randn(N, B, Ck, k, dv)

    def run(fold):
        V = torch.zeros(B, Mrows, dv); U = torch.zeros(B, Mrows)
        ws = torch.ones(()); reads = []
        off = torch.arange(B)[:, None, None] * Mrows
        for n in range(N):
            flat = (idx[n] + off).reshape(-1)
            Vf, Uf = V.reshape(B * Mrows, dv), U.reshape(B * Mrows)
            rows = Vf.index_select(0, flat).view(B, Ck, k, dv)
            use = Uf.index_select(0, flat).view(B, Ck, k)
            m_raw = (w[n] * use).sum(-1)
            reads.append((w[n].unsqueeze(-1) * rows).sum(-2)
                         / (m_raw.unsqueeze(-1) + 1e-4 * (ws if fold else 1.0)))
            g = ws if fold else 1.0
            Vf.index_add_(0, flat, (g * (w[n].unsqueeze(-1) * upd[n])).reshape(-1, dv))
            Uf.index_add_(0, flat, (g * w[n]).reshape(-1))
            if fold:
                ws = ws / lam
            else:
                V = V * lam; U = U * lam
        return torch.stack(reads)

    err = (run(False) - run(True)).abs().max().item()
    assert err < 1e-4, f"write-scale decay != state decay ({err})"
    print(f"  write-scale decay == state decay   max abs err {err:.2e}")


def test_param_parity():
    """Non-embedding parameter counts must be within 5% across arms."""
    for size in ("small", "base"):
        preset = dict(SIZES[size]); vocab = preset.pop("vocab")
        cfg = dict(DEFAULT_CFG); cfg.update(preset)
        M.MEMORY_SLOTS = M.memory_slots(cfg["n_layers"])
        counts = {}
        for mix in ("sliding", "attention", "gla", "deltanet", "psm"):
            counts[mix] = M.Model(vocab, memory_mixer=mix,
                                  cfg=cfg).param_report()["non_embedding"]
        _check_parity(size, counts)


def _check_parity(size, counts):
    base = max(counts.values())
    for mix, c in counts.items():
        gap = abs(c - base) / base
        print(f"  [{size}] {mix:10s} non-emb {c:,}  ({gap*100:+.2f}% vs largest)")
        assert gap < 0.05, f"{size}/{mix} is {gap*100:.1f}% off parity"


if __name__ == "__main__":
    print("\n[1] chunked recurrence correctness")
    test_chunked_matches_sequential()
    print("\n[2] causality")
    test_causality()
    print("\n[3] address pattern separation")
    test_address_separation()
    print("\n[4] decay refactor equivalence")
    test_decay_equivalence()
    print("\n[5] parameter parity")
    test_param_parity()
    print("\nall tests passed")
