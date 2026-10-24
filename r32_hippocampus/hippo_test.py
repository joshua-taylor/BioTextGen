#!/usr/bin/env python3
"""Round 32 merged test: does a hippocampus-style memory beat text retrieval read twice?

Every arm fine-tunes the same small LM (SmolLM2-135M) on a stream of freshly generated
fictional "fact worlds" (people, countries, cities, companies, ...). Facts are written once
(or stored once) per world; the model must answer forward, backward (reverse) and two-hop
questions. Held-out worlds and MQuAKE-derived real-name tests measure what each arm learned
to do, not what it memorised.

Arms
  ctx_gold       the needed facts (+2 distractors) pasted into the prompt as text  (ceiling)
  mem_gold       the needed facts (+2 distractors) given as compact memory vectors (can 4 vectors carry a fact?)
  rag1 / rag2    dense retrieval of fact sentences as text, read once / read twice
  mem1way        compact memory: each fact -> 4 vectors, filed under its subject only, read once
  mem2way        ... filed under subject AND object (two-way filing), read once
  mem2way_twice  ... two-way, read twice (second read keyed from what the first read returned)
  memfull        ... two-way + integrative linking at write time + read twice
  memlink_once   ... two-way + linking, read once (ablation)

Kaggle (T4):  python hippo_test.py --arms all --steps 3000 --out r32_results.json
CPU smoke:    python hippo_test.py --debug --arms all --steps 6
"""
import argparse, json, math, os, random, re, time, subprocess, copy
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------------------------------------------------------- names
ON = ['b', 'c', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'z', 'br', 'dr', 'gr', 'kr', 'tr', 'st', 'th', 'sh']
VW = ['a', 'e', 'i', 'o', 'u', 'ai', 'ou', 'ia', 'eo']
CO = ['', '', 'n', 'r', 'l', 's', 'm', 'th', 'x']


def word(r, nsyl):
    return (''.join(r.choice(ON) + r.choice(VW) for _ in range(nsyl)) + r.choice(CO)).capitalize()


NAME_FN = {
    'person': lambda r: word(r, r.choice([1, 2])) + ' ' + word(r, r.choice([2, 3])),
    'country': lambda r: word(r, 2) + r.choice(['ia', 'land', 'ar', 'istan', 'ora']),
    'city': lambda r: word(r, r.choice([1, 2])) + r.choice(['burg', 'ton', 'ova', 'grad', 'port', 'ville', '']),
    'company': lambda r: word(r, 2) + ' ' + r.choice(['Corp', 'Labs', 'Group', 'Industries', 'Systems', 'Works']),
    'university': lambda r: word(r, 2) + ' ' + r.choice(['University', 'Institute', 'College']),
    'work': lambda r: 'The ' + word(r, 2) + ' ' + r.choice(['Chronicles', 'Song', 'Garden', 'Voyage', 'Letters', 'Machine']),
    'language': lambda r: word(r, 2) + r.choice(['ish', 'ese', 'ic', 'ian']),
    'sport': lambda r: word(r, 2).lower() + r.choice(['ball', 'polo', ' racing', ' wrestling']),
    'religion': lambda r: word(r, 2) + 'ism',
    'field': lambda r: word(r, 2).lower() + r.choice(['ology', 'ics', 'ometry']),
    'continent': lambda r: word(r, 2) + 'a',
}


class Namer:
    def __init__(self, r):
        self.r, self.seen = r, set()

    def fresh(self, t):
        for _ in range(10000):
            n = NAME_FN[t](self.r)
            if n.lower() not in self.seen:
                self.seen.add(n.lower())
                return n
        raise RuntimeError('name space exhausted')


# ----------------------------------------------------------------------------- relations
# statement / question templates follow MQuAKE's own (prompts/templates); 'np' is the noun phrase used to
# compose two-hop questions; 'rq' is the reverse question for one-to-one relations ([Y] = the object).
REL = {
    'P27': dict(s='person', o='country', st='[X] is a citizen of __', q='What is the country of citizenship of [X]?', np='the country of citizenship of [X]'),
    'P19': dict(s='person', o='city', st='[X] was born in the city of __', q='Which city was [X] born in?', np='the city where [X] was born'),
    'P69': dict(s='person', o='university', st='The university where [X] was educated is __', q='Which university was [X] educated at?', np='the university where [X] was educated'),
    'P108': dict(s='person', o='company', st='[X] is employed by __', q='Who is the employer of [X]?', np='the employer of [X]'),
    'P641': dict(s='person', o='sport', st='[X] is associated with the sport of __', q='Which sport is [X] associated with?', np='the sport associated with [X]'),
    'P140': dict(s='person', o='religion', st='[X] is affiliated with the religion of __', q='Which religion is [X] affiliated with?', np='the religion of [X]'),
    'P1412': dict(s='person', o='language', st='[X] speaks the language of __', q='What language does [X] speak?', np='the language spoken by [X]'),
    'P106': dict(s='person', o='field', st='[X] works in the field of __', q='What kind of work does [X] do?', np='the field of work of [X]'),
    'P26': dict(s='person', o='person', st='[X] is married to __', q='Who is [X] married to?', np='the spouse of [X]', rq='Who is [Y] married to?'),
    'P36': dict(s='country', o='city', st='The capital of [X] is __', q='What is the capital of [X]?', np='the capital of [X]', rq='[Y] is the capital of which country?'),
    'P35': dict(s='country', o='person', st='The name of the current head of state in [X] is __', q='What is the name of the current head of state in [X]?', np='the head of state of [X]', rq='[Y] is the head of state of which country?'),
    'P6': dict(s='country', o='person', st='The name of the current head of the [X] government is __', q='What is the name of the current head of the [X] government?', np='the head of government of [X]', rq='[Y] is the head of government of which country?'),
    'P37': dict(s='country', o='language', st='The official language of [X] is __', q='What is the official language of [X]?', np='the official language of [X]'),
    'P30': dict(s='country', o='continent', st='[X] is located in the continent of __', q='Which continent is [X] located in?', np='the continent where [X] is located'),
    'P17': dict(s='city', o='country', st='[X] is located in the country of __', q='Which country is [X] located in?', np='the country where [X] is located'),
    'P112': dict(s='company', o='person', st='[X] was founded by __', q='Who founded [X]?', np='the founder of [X]', rq='Which company was founded by [Y]?'),
    'P169': dict(s='company', o='person', st='The chief executive officer of [X] is __', q='Who is the chief executive officer of [X]?', np='the chief executive officer of [X]', rq='Which company is [Y] the chief executive officer of?'),
    'P159': dict(s='company', o='city', st='The headquarters of [X] is located in the city of __', q='Which city is the headquarter of [X] located in?', np='the city where the headquarters of [X] is located'),
    'P50': dict(s='work', o='person', st='The author of [X] is __', q='Who is the author of [X]?', np='the author of [X]'),
    'P495': dict(s='work', o='country', st='[X] was created in the country of __', q='Which country was [X] created in?', np='the country where [X] was created'),
    'P131': dict(s='university', o='city', st='[X] is located in the city of __', q='Where is [X] located?', np='the location of [X]'),
}
PERSON_RELS = ['P27', 'P19', 'P69', 'P108', 'P641', 'P140', 'P1412', 'P106']


def make_world(r, namer, size=1.0):
    """A small consistent fictional world. Returns list of facts (s, rel, o)."""
    E = lambda t, n: [namer.fresh(t) for _ in range(n)]
    nC, nCo, nU, nW = max(2, round(4 * size)), max(2, round(6 * size)), max(2, round(3 * size)), max(2, round(6 * size))
    nP = max(round(24 * size), 2 * nC + 2 * nCo + 4)
    conts, langs, countries = E('continent', 2), E('language', nC), E('country', nC)
    facts, cities = [], []
    for i, c in enumerate(countries):
        cs = E('city', 3)
        cities += cs
        facts += [(c, 'P30', r.choice(conts)), (c, 'P37', langs[i]), (c, 'P36', cs[0])] + [(x, 'P17', c) for x in cs]
    people, unis, comps, works = E('person', nP), E('university', nU), E('company', nCo), E('work', nW)
    sports, rels, fields = E('sport', 4), E('religion', 3), E('field', 4)
    pool = people[:]
    r.shuffle(pool)
    it = iter(pool)
    for c in countries:
        facts += [(c, 'P35', next(it)), (c, 'P6', next(it))]
    for co in comps:
        facts += [(co, 'P112', next(it)), (co, 'P169', next(it)), (co, 'P159', r.choice(cities))]
    for u in unis:
        facts.append((u, 'P131', r.choice(cities)))
    for w in works:
        facts += [(w, 'P50', r.choice(people)), (w, 'P495', r.choice(countries))]
    obj = {'country': countries, 'city': cities, 'university': unis, 'company': comps, 'sport': sports,
           'religion': rels, 'language': langs, 'field': fields}
    for p in people:
        for rel in PERSON_RELS:
            if r.random() < 0.5:
                facts.append((p, rel, r.choice(obj[REL[rel]['o']])))
    married = people[:]
    r.shuffle(married)
    for i in range(0, max(2, nP // 3) - 1, 2):
        facts.append((married[i], 'P26', married[i + 1]))
    r.shuffle(facts)
    return facts


def make_questions(facts):
    """fwd / rev / two-hop questions with gold fact indices."""
    by_s = {}
    for i, (s, rel, o) in enumerate(facts):
        by_s.setdefault(s, []).append(i)
    qs = []
    for i, (s, rel, o) in enumerate(facts):
        R = REL[rel]
        qs.append(dict(type='fwd', q=R['q'].replace('[X]', s), a=o, g1=[i], g2=[]))
        if 'rq' in R:
            qs.append(dict(type='rev', q=R['rq'].replace('[Y]', o), a=s, g1=[i], g2=[]))
        for j in by_s.get(o, []):
            s2, rel2, o2 = facts[j]
            if (rel, rel2) == ('P36', 'P17') or o2 == s:
                continue
            q = REL[rel2]['q'].replace('[X]', REL[rel]['np'].replace('[X]', s))
            qs.append(dict(type='two', q=q, a=o2, g1=[i], g2=[j], link=(i, j)))
    return qs


# ----------------------------------------------------------------------------- tokenizer wrappers
class CharTok:
    """Debug stand-in for the HF tokenizer (no network here)."""
    vocab_size = 128

    def encode(self, s):
        return [min(ord(c), 127) for c in s]


class Tok:
    def __init__(self, hf):
        self.hf, self.cache = hf, {}

    def encode(self, s):
        if s not in self.cache:
            self.cache[s] = self.hf.encode(s, add_special_tokens=False)
        return self.cache[s]


def stmt_tokens(tok, template, s, o):
    """Tokenise a statement from segments so subject / object spans are exact."""
    parts = re.split(r'(\[X\]|__)', template)
    ids, spans, pend_space, text = [], {}, '', ''
    for p in parts:
        if p in ('[X]', '__'):
            name = s if p == '[X]' else o
            seg = pend_space + name
            st = len(ids)
            ids += tok.encode(seg)
            spans['s' if p == '[X]' else 'o'] = (st, len(ids))
            text += seg
            pend_space = ''
        elif p:
            core = p.rstrip(' ')
            pend_space = ' ' if p.endswith(' ') else ''
            if core:
                ids += tok.encode(core)
                text += core
    ids += tok.encode('.')
    text += '.'
    return ids, spans, text


# ----------------------------------------------------------------------------- model pieces
class MLP(nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))

    def forward(self, x):
        return F.normalize(self.f(x), dim=-1)


class AttnPool(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n, d) / math.sqrt(d))
        self.k = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):  # x [B,T,d], mask [B,T] bool
        a = torch.einsum('nd,btd->bnt', self.q, self.k(x)) / math.sqrt(x.shape[-1])
        a = a.masked_fill(~mask[:, None, :], -1e4).softmax(-1)
        return torch.einsum('bnt,btd->bnd', a, x)


class Heads(nn.Module):
    def __init__(self, d, dk, m, emb_scale):
        super().__init__()
        # compact memory
        f = 2 * d  # retrieval features = [token embedding ; middle-layer hidden]
        self.ent, self.relf, self.relr = MLP(f, dk), MLP(f, dk), MLP(f, dk)
        self.linkf, self.linkr = MLP(2 * f, dk), MLP(2 * f, dk)
        self.qpool, self.qe, self.qr = AttnPool(f, 2), MLP(f, dk), MLP(f, dk)
        self.q2pool, self.q2e, self.q2r = AttnPool(f, 2), MLP(f, dk), MLP(f, dk)
        self.slotpool = AttnPool(d, m)
        self.slotproj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.slot_scale = nn.Parameter(torch.tensor(float(emb_scale)))
        # dense text retrieval
        self.kd, self.qd, self.q2d = MLP(f, dk), MLP(f, dk), MLP(f, dk)
        self.temp = nn.Parameter(torch.tensor(math.log(10.0)))


# ----------------------------------------------------------------------------- the system
class System:
    def __init__(self, args, arm, dev):
        self.args, self.arm, self.dev = args, arm, dev
        self.kind = 'ctx' if arm == 'ctx_gold' else ('rag' if arm.startswith('rag') else 'mem')
        self.gold = arm in ('ctx_gold', 'mem_gold')
        self.twice = arm in ('rag2', 'mem2way_twice', 'memfull')
        self.two_way = arm in ('mem2way', 'mem2way_twice', 'memfull', 'memlink_once')
        self.link = arm in ('memfull', 'memlink_once')
        if args.debug:
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                              num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
            torch.manual_seed(0)
            self.lm = LlamaForCausalLM(cfg)
            self.tok = CharTok()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
            self.tok = Tok(AutoTokenizer.from_pretrained(args.model))
        self.lm.to(dev)
        self.emb = self.lm.get_input_embeddings()
        d = self.emb.weight.shape[1]
        self.L = self.lm.config.num_hidden_layers
        self.lk = max(1, round(self.L * args.lk_frac))
        scale = self.emb.weight.detach().norm(dim=-1).mean().item()
        self.h = Heads(d, args.dk, args.m, scale).to(dev)
        self.d = d
        self.opt = torch.optim.AdamW([{'params': self.lm.parameters(), 'lr': args.lr_lm},
                                      {'params': self.h.parameters(), 'lr': args.lr_heads}], weight_decay=0.0)
        self.use_amp = dev.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.nl = self.tok.encode('\n')
        self.q_pre, self.a_pre = self.tok.encode('Question:'), self.tok.encode('\nAnswer:')

    # ---- helpers
    def ac(self):
        return torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp)

    def pad(self, seqs, val=0):
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), val, dtype=torch.long)
        m = torch.zeros((len(seqs), T), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            m[i, :len(s)] = True
        return x.to(self.dev), m.to(self.dev)

    @torch.no_grad()
    def hidden(self, seqs, bs=64):
        """Run the LM on token lists; return per-sequence (layer-lk hidden, last hidden) [T,d] float32."""
        out = []
        for i in range(0, len(seqs), bs):
            x, m = self.pad(seqs[i:i + bs])
            with self.ac():
                o = self.lm(input_ids=x, attention_mask=m.long(), output_hidden_states=True)
            hk = torch.cat([o.hidden_states[0], o.hidden_states[self.lk]], -1).float()  # token identity + context
            hv = o.hidden_states[-1].float()
            for b in range(x.shape[0]):
                n = int(m[b].sum())
                out.append((hk[b, :n], hv[b, :n]))
        return out

    def qtok(self, q):
        return self.q_pre + self.tok.encode(' ' + q) + self.a_pre

    def atok(self, a):
        return self.tok.encode(' ' + a) + self.nl

    # ---- writing a world
    def write(self, facts, gold_links=None):
        """Store a world. Text arms keep sentences; memory arms build items (+links) with keys."""
        tok = self.tok
        W = dict(facts=facts, items=[])
        for i, (s, rel, o) in enumerate(facts):
            ids, sp, text = stmt_tokens(tok, REL[rel]['st'], s, o)
            W['items'].append(dict(ids=ids, sp=sp, text=text, s=s, o=o, facts=(i,), kind='fact'))
        if self.kind == 'ctx':
            return W
        H = self.hidden([it['ids'] for it in W['items']])
        for it, (hk, hv) in zip(W['items'], H):
            it['hv'] = hv
            it['mean'] = hk.mean(0)
            a, b = it['sp']['s']
            c, e = it['sp']['o']
            relpos = [p for p in range(len(it['ids'])) if not (a <= p < b or c <= p < e)]
            it['fs'], it['fo'], it['fr'] = hk[a:b].mean(0), hk[c:e].mean(0), hk[relpos].mean(0)
        if self.kind == 'mem' and self.link:
            self.add_links(W, gold_links)
        return W

    def ent_keys(self, W):
        nf = sum(1 for it in W['items'] if it['kind'] == 'fact')
        fs = torch.stack([it['fs'] for it in W['items'][:nf]])
        fo = torch.stack([it['fo'] for it in W['items'][:nf]])
        return self.h.ent(fs), self.h.ent(fo)

    @torch.no_grad()
    def add_links(self, W, gold_links=None):
        """Integrative encoding: when fact f's object is fact g's subject, store a linked item 'f. g.'
        Linking uses the learned entity codes (cosine > thr) unless gold links are supplied (training)."""
        facts = W['items']
        nf = len(facts)
        if gold_links is not None:
            pairs = gold_links
        else:
            ks, ko = self.ent_keys(W)
            sim = ko @ ks.T  # obj of f vs subj of g
            sim.fill_diagonal_(-1)
            pairs = [(int(i), int(j)) for i, j in (sim > self.args.link_thr).nonzero().tolist()]
            W['link_pairs'] = pairs
        pairs = pairs[: self.args.max_links]
        new = []
        for i, j in pairs:
            f, g = facts[i], facts[j]
            ids = f['ids'] + self.tok.encode(' ') + g['ids'] if not self.args.debug else f['ids'] + [32] + g['ids']
            new.append(dict(ids=ids, text=f['text'] + ' ' + g['text'], s=f['s'], o=g['o'], facts=(i, j), kind='link',
                            fs=f['fs'], fo=g['fo'], fr=torch.cat([f['fr'], g['fr']])))
        if new:
            H = self.hidden([it['ids'] for it in new])
            for it, (hk, hv) in zip(new, H):
                it['hv'] = hv
        W['items'] += new

    def keys(self, W):
        """All (item, role) keys for the memory arms. role f = filed under subject, r = under object."""
        its = W['items']
        fs = torch.stack([it['fs'] for it in its])
        fo = torch.stack([it['fo'] for it in its])
        isl = torch.tensor([it['kind'] == 'link' for it in its], device=self.dev)
        nf = int((~isl).sum())
        fr_fact = torch.stack([it['fr'] for it in its[:nf]])
        ke_s, ke_o = self.h.ent(fs), self.h.ent(fo)
        kr_f = [self.h.relf(fr_fact)]
        kr_r = [self.h.relr(fr_fact)]
        if len(its) > nf:
            fr_link = torch.stack([it['fr'] for it in its[nf:]])
            kr_f.append(self.h.linkf(fr_link))
            kr_r.append(self.h.linkr(fr_link))
        kr_f, kr_r = torch.cat(kr_f), torch.cat(kr_r)
        KE, KR, owner, role = [ke_s], [kr_f], list(range(len(its))), ['f'] * len(its)
        if self.two_way:
            KE.append(ke_o)
            KR.append(kr_r)
            owner += list(range(len(its)))
            role += ['r'] * len(its)
        return torch.cat(KE), torch.cat(KR), owner, role

    def dense_keys(self, W):
        return self.h.kd(torch.stack([it['mean'] for it in W['items']]))

    # ---- queries
    def q_feats(self, qs):
        H = self.hidden([self.qtok(q['q']) for q in qs])
        x, m = self.pad_float([h[0] for h in H])
        return x, m, (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)

    def pad_float(self, hs):
        T = max(h.shape[0] for h in hs)
        d = hs[0].shape[1]
        x = torch.zeros(len(hs), T, d, device=self.dev)
        m = torch.zeros(len(hs), T, dtype=torch.bool, device=self.dev)
        for i, h in enumerate(hs):
            x[i, :len(h)] = h
            m[i, :len(h)] = True
        return x, m

    def mem_scores(self, W, x, m, second=False):
        KE, KR, owner, role = self.keys(W)
        pool, he, hr = (self.h.q2pool, self.h.q2e, self.h.q2r) if second else (self.h.qpool, self.h.qe, self.h.qr)
        p = pool(x, m)
        qe, qr = he(p[:, 0]), hr(p[:, 1])
        return (qe @ KE.T + qr @ KR.T) * self.h.temp.exp(), owner, role

    def topk_items(self, scores, owner, k, exclude=()):
        out = []
        for b in range(scores.shape[0]):
            got = []
            for idx in scores[b].argsort(descending=True).tolist():
                it = owner[idx] if owner is not None else idx
                if it not in got and it not in exclude[b] if exclude else it not in got:
                    got.append(it)
                if len(got) == k:
                    break
            out.append(got)
        return out

    @staticmethod
    def nce(scores, pos_mask):
        """multi-positive InfoNCE; rows without positives are skipped."""
        has = pos_mask.any(1)
        if not has.any():
            return scores.sum() * 0
        lse = torch.logsumexp(scores, 1)
        lpos = torch.logsumexp(scores.masked_fill(~pos_mask, -1e4), 1)
        return (lse - lpos)[has].mean()

    # ---- gold targets (item indices) for each question
    def gold_items(self, W, q, read):
        its = W['items']
        if read == 1:
            g = set(q['g1'])
            if self.link and q['type'] == 'two':
                g |= {n for n, it in enumerate(its) if it['kind'] == 'link' and it['facts'] == tuple(q['link'])}
            return g
        return set(q['g2'])

    def gold_roles(self, W, q, read, owner, role):
        g = self.gold_items(W, q, read)
        need = 'r' if (q['type'] == 'rev' and read == 1) else 'f'
        return [o in g and r == need for o, r in zip(owner, role)]

    # ---- reader inputs
    def slots(self, W, items):
        hv = [W['items'][i]['hv'] for i in items]
        if not hv:
            return torch.zeros(0, self.emb.weight.shape[1], device=self.dev)
        x, m = self.pad_float(hv)
        s = self.h.slotproj(self.h.slotpool(x, m))  # [n, m, d]
        s = F.normalize(s, dim=-1) * self.h.slot_scale
        return s.reshape(-1, s.shape[-1])

    def build_batch(self, W, qs, retrieved, with_answer=True):
        """retrieved: list (per q) of item indices. Returns inputs_embeds, mask, labels, q-positions."""
        embs, labs, qpos = [], [], []
        for q, items in zip(qs, retrieved):
            if self.kind == 'mem':
                pre = self.slots(W, items)
                text = []
            else:
                pre = None
                text = []
                for i in items:
                    text += self.tok.encode('Fact: ' + W['items'][i]['text'] + '\n') if not self.args.debug else \
                        self.tok.encode('F: ' + W['items'][i]['text'] + '\n')
            qt = self.qtok(q['q'])
            at = self.atok(q['a']) if with_answer else []
            ids = torch.tensor(text + qt + at, device=self.dev)
            e = self.emb(ids)
            if pre is not None and pre.shape[0]:
                e = torch.cat([pre.to(e.dtype), e])
            n0 = e.shape[0] - len(qt) - len(at)
            lab = torch.full((e.shape[0],), -100, device=self.dev, dtype=torch.long)
            if with_answer:
                lab[e.shape[0] - len(at):] = torch.tensor(at, device=self.dev)
            embs.append(e)
            labs.append(lab)
            qpos.append((n0, n0 + len(qt)))
        T = max(e.shape[0] for e in embs)
        d = embs[0].shape[1]
        X = torch.zeros(len(embs), T, d, device=self.dev, dtype=embs[0].dtype)
        M = torch.zeros(len(embs), T, dtype=torch.long, device=self.dev)
        Y = torch.full((len(embs), T), -100, dtype=torch.long, device=self.dev)
        for i, (e, l) in enumerate(zip(embs, labs)):
            X[i, :len(e)], M[i, :len(e)], Y[i, :len(l)] = e, 1, l
        return X, M, Y, qpos

    def lm_run(self, X, M, Y=None, hidden=False):
        with self.ac():
            o = self.lm(inputs_embeds=X, attention_mask=M, output_hidden_states=hidden)
        logits = o.logits.float()
        out = {}
        if Y is not None:
            lg, yy = logits[:, :-1], Y[:, 1:]
            out['loss'] = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), yy.reshape(-1), ignore_index=-100)
            ok = (lg.argmax(-1) == yy) | (yy == -100)
            out['em'] = ok.all(1).float()  # teacher-forced all-token match == greedy exact match
        if hidden:
            out['hk'] = torch.cat([o.hidden_states[0], o.hidden_states[self.lk]], -1).float()
            out['hlast'] = o.hidden_states[-1].float()
        return out

    # ---- one pass of retrieve-and-answer (shared by train and eval)
    def answer(self, W, qs, train=False, rng=None):
        k = self.args.k
        stats = {'ctx_len': 0.0, 'r1': 0.0, 'r2': 0.0, 'n2': 0}
        aux = torch.zeros((), device=self.dev)
        if self.gold:
            retrieved = []
            for q in qs:
                g = list(q['g1']) + list(q['g2'])
                d = (rng or random).sample(range(len(W['items'])), 2)
                items = g + [x for x in d if x not in g]
                (rng or random).shuffle(items)
                retrieved.append(items)
        else:
            x, m, qmean = self.q_feats(qs)
            if self.kind == 'mem':
                S, owner, role = self.mem_scores(W, x, m)
                pos = torch.tensor([self.gold_roles(W, q, 1, owner, role) for q in qs], device=self.dev)
            else:
                S, owner = (self.h.qd(qmean) @ self.dense_keys(W).T) * self.h.temp.exp(), None
                pos = torch.tensor([[i in q['g1'] for i in range(len(W['items']))] for q in qs], device=self.dev)
            if train:
                aux = aux + self.nce(S, pos)
            retrieved = self.topk_items(S.detach(), owner, k)
            for b, q in enumerate(qs):
                g = self.gold_items(W, q, 1) if self.kind == 'mem' else set(q['g1'])
                stats['r1'] += float(bool(g & set(retrieved[b])))
                if train and not (g & set(retrieved[b])) and rng.random() < 0.5:
                    retrieved[b][-1] = sorted(g)[0]
            if self.twice:
                X, M, _, qpos = self.build_batch(W, qs, retrieved, with_answer=False)
                with torch.no_grad():
                    o = self.lm_run(X, M, hidden=True)
                if self.kind == 'mem':
                    h1 = [o['hk'][b, a:e] for b, (a, e) in enumerate(qpos)]
                    x2, m2 = self.pad_float(h1)
                    S2, owner2, role2 = self.mem_scores(W, x2, m2, second=True)
                    pos2 = torch.tensor([self.gold_roles(W, q, 2, owner2, role2) for q in qs], device=self.dev)
                else:
                    mm = M.bool()
                    hm = (o['hk'] * mm[..., None]).sum(1) / mm.sum(1, keepdim=True)
                    S2, owner2 = (self.h.q2d(hm) @ self.dense_keys(W).T) * self.h.temp.exp(), None
                    pos2 = torch.tensor([[i in q['g2'] for i in range(len(W['items']))] for q in qs], device=self.dev)
                if train:
                    aux = aux + self.nce(S2, pos2)
                r2 = self.topk_items(S2.detach(), owner2, k, exclude=[set(r) for r in retrieved])
                for b, q in enumerate(qs):
                    if q['g2']:
                        g2 = set(q['g2'])
                        stats['n2'] += 1
                        stats['r2'] += float(bool(g2 & (set(r2[b]) | set(retrieved[b]))))
                        if train and not (g2 & (set(r2[b]) | set(retrieved[b]))) and rng.random() < 0.5:
                            r2[b][-1] = sorted(g2)[0]
                retrieved = [a + b for a, b in zip(retrieved, r2)]
        X, M, Y, _ = self.build_batch(W, qs, retrieved)
        o = self.lm_run(X, M, Y)
        stats['ctx_len'] = float(M.sum(1).float().mean()) - sum(len(self.qtok(q['q'])) + len(self.atok(q['a'])) for q in qs) / len(qs)
        return o, aux, stats, retrieved

    def entity_loss(self, W, rng, n=256):
        """Role-invariant entity codes: mentions of the same entity (as subject or object) should match."""
        facts = [it for it in W['items'] if it['kind'] == 'fact']
        ment = [(it['fs'], it['s']) for it in facts] + [(it['fo'], it['o']) for it in facts]
        rng.shuffle(ment)
        ment = ment[:n]
        z = self.h.ent(torch.stack([x for x, _ in ment]))
        names = [nm for _, nm in ment]
        S = z @ z.T * self.h.temp.exp()
        eye = torch.eye(len(ment), dtype=torch.bool, device=self.dev)
        S = S.masked_fill(eye, -1e4)
        pos = torch.tensor([[a == b for b in names] for a in names], device=self.dev) & ~eye
        return self.nce(S, pos)

    # ---- training step on one fresh world
    def train_step(self, rng):
        namer = Namer(rng)
        facts = make_world(rng, namer)
        qs = make_questions(facts)
        types = {'fwd': [], 'rev': [], 'two': []}
        for q in qs:
            types[q['type']].append(q)
        mix = {'fwd': 0.35, 'rev': 0.25, 'two': 0.40}
        batch = []
        for t, p in mix.items():
            batch += rng.sample(types[t], min(len(types[t]), round(self.args.bs * p)))
        gl = sorted({tuple(q['link']) for q in qs if q['type'] == 'two'}) if self.link else None
        W = self.write(facts, gold_links=gl)
        self.lm.train()
        o, aux, st, _ = self.answer(W, batch, train=True, rng=rng)
        loss = o['loss'] + aux
        if self.kind == 'mem':
            loss = loss + self.entity_loss(W, rng)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(list(self.lm.parameters()) + list(self.h.parameters()), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(o['loss']), float(aux), float(o['em'].mean()), st['r1'] / len(batch)

    # ---- evaluation
    @torch.no_grad()
    def evaluate(self, worlds, max_q=400, rng=None):
        """worlds: list of (facts, questions). Returns per-type exact match + retrieval recall + cost."""
        self.lm.eval()
        res = {}
        for facts, qs in worlds:
            W = self.write(facts)
            if self.link and 'link_pairs' in W:
                gold = {(i, j) for i, f in enumerate(facts) for j, g in enumerate(facts) if i != j and f[2] == g[0]}
                got = set(W['link_pairs'])
                res.setdefault('_link_prec', []).append(len(got & gold) / max(1, len(got)))
                res.setdefault('_link_rec', []).append(len(got & gold) / max(1, len(gold)))
            qs = qs[:max_q]
            for i in range(0, len(qs), self.args.eval_bs):
                b = qs[i:i + self.args.eval_bs]
                o, _, st, _ = self.answer(W, b, rng=rng)
                for q, em in zip(b, o['em'].tolist()):
                    res.setdefault(q['type'], []).append(em)
                res.setdefault('_r1', []).append(st['r1'] / len(b))
                if st['n2']:
                    res.setdefault('_r2', []).append(st['r2'] / st['n2'])
                res.setdefault('_ctx', []).append(st['ctx_len'])
        return {k: (sum(v) / len(v), len(v)) for k, v in res.items()}


# ----------------------------------------------------------------------------- evaluation sets
def eval_worlds(seed, n, size=1.0, per_type=60):
    r = random.Random(seed)
    namer = Namer(r)
    out = []
    for _ in range(n):
        facts = make_world(r, namer, size)
        qs = make_questions(facts)
        pick = []
        for t in ('fwd', 'rev', 'two'):
            tq = [q for q in qs if q['type'] == t]
            pick += r.sample(tq, min(per_type, len(tq)))
        out.append((facts, pick))
    return out


def merged_world(worlds, r, per_type=100):
    """One big memory holding several worlds' facts (names are globally unique)."""
    facts, qs = [], []
    for f, q in worlds:
        off = len(facts)
        facts += f
        for x in q:
            y = dict(x)
            y['g1'] = [i + off for i in x['g1']]
            y['g2'] = [i + off for i in x['g2']]
            if 'link' in x:
                y['link'] = (x['link'][0] + off, x['link'][1] + off)
            qs.append(y)
    pick = []
    for t in ('fwd', 'rev', 'two'):
        tq = [q for q in qs if q['type'] == t]
        pick += r.sample(tq, min(per_type, len(tq)))
    return [(facts, pick)]


def mquake_worlds(path, chunk=100, max_cases=600, seed=0):
    """Real-name test built from MQuAKE-CF-3k-v2 two-hop cases: all (counterfactual) single hops are written
    to memory; ask the multi-hop question, each single hop (forward), and reverse questions for 1-1 relations."""
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    # map cloze templates to our relation ids
    pats = []
    for rid, R in REL.items():
        rx = '^' + re.escape(R['st'].replace(' __', '')).replace(re.escape('[X]'), '(.+?)') + '$'
        pats.append((re.compile(rx), rid))
    # MQuAKE wording for templates we changed slightly
    extra = {'The univeristy where [X] was educated is': 'P69'}
    for t, rid in extra.items():
        pats.append((re.compile('^' + re.escape(t).replace(re.escape('[X]'), '(.+?)') + '$'), rid))
    r = random.Random(seed)
    cases = [c for c in data if len(c['new_single_hops']) == 2]
    r.shuffle(cases)
    cases = cases[:max_cases]
    worlds = []
    for c0 in range(0, len(cases), chunk):
        facts, qs = [], []
        for c in cases[c0:c0 + chunk]:
            hop_ids = []
            ok = True
            for h in c['new_single_hops']:
                m = None
                for rx, rid in pats:
                    mm = rx.match(h['cloze'])
                    if mm:
                        m = (mm.group(1), rid)
                        break
                if m is None:
                    ok = False
                    break
                facts.append((m[0], m[1], h['answer']))
                hop_ids.append(len(facts) - 1)
            if not ok:
                del facts[len(facts) - len(hop_ids):]
                continue
            i, j = hop_ids
            if facts[i][2] != facts[j][0]:
                continue
            qs.append(dict(type='two', q=c['questions'][0], a=c['new_answer'], g1=[i], g2=[j], link=(i, j)))
            for f in (i, j):
                s, rid, o = facts[f]
                qs.append(dict(type='fwd', q=REL[rid]['q'].replace('[X]', s), a=o, g1=[f], g2=[]))
                if 'rq' in REL[rid]:
                    qs.append(dict(type='rev', q=REL[rid]['rq'].replace('[Y]', o), a=s, g1=[f], g2=[]))
        if qs:
            worlds.append((facts, qs))
    return worlds


# ----------------------------------------------------------------------------- main
ALL_ARMS = ['ctx_gold', 'mem_gold', 'rag1', 'rag2', 'mem1way', 'mem2way', 'mem2way_twice', 'memlink_once', 'memfull']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='all')
    ap.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--eval_bs', type=int, default=32)
    ap.add_argument('--k', type=int, default=4, help='items / sentences per read')
    ap.add_argument('--m', type=int, default=4, help='memory vectors per item')
    ap.add_argument('--dk', type=int, default=128)
    ap.add_argument('--lr_lm', type=float, default=5e-5)
    ap.add_argument('--lr_heads', type=float, default=1e-3)
    ap.add_argument('--link_thr', type=float, default=0.8)
    ap.add_argument('--lk_frac', type=float, default=0.5, help='layer (fraction of depth) used for retrieval features')
    ap.add_argument('--max_links', type=int, default=4000)
    ap.add_argument('--n_eval', type=int, default=12)
    ap.add_argument('--mquake', default='MQuAKE/datasets/MQuAKE-CF-3k-v2.json')
    ap.add_argument('--out', default='r32_results.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    if args.debug:
        args.lr_lm, args.n_eval, args.bs, args.eval_bs = 1e-3, 2, 8, 8
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arms = ALL_ARMS if args.arms == 'all' else args.arms.split(',')
    if not args.debug and not os.path.exists(args.mquake):
        try:
            subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/princeton-nlp/MQuAKE'], check=True)
        except Exception as e:
            print('MQuAKE download failed:', e)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    ev = eval_worlds(10_000 + args.seed, args.n_eval, per_type=60 if not args.debug else 6)
    big = merged_world(eval_worlds(20_000 + args.seed, 8 if not args.debug else 2, per_type=1000), random.Random(1),
                       per_type=150 if not args.debug else 6)
    mq = mquake_worlds(args.mquake, max_cases=600 if not args.debug else 40, chunk=100 if not args.debug else 20)
    for arm in arms:
        if arm in results and not args.debug:
            print('skip (done):', arm)
            continue
        t0 = time.time()
        torch.manual_seed(args.seed)
        S = System(args, arm, dev)
        log = []
        for step in range(1, args.steps + 1):
            rng = random.Random(args.seed * 1_000_003 + step)
            l, a, em, r1 = S.train_step(rng)
            log.append((l, a, em, r1))
            if step % 100 == 0 or step == args.steps or args.debug:
                n = min(100, len(log))
                avg = [sum(x[i] for x in log[-n:]) / n for i in range(4)]
                print(f'[{arm}] step {step} qa_loss {avg[0]:.3f} ret_loss {avg[1]:.3f} train_em {avg[2]:.3f} '
                      f'recall1 {avg[3]:.3f}  {time.time() - t0:.0f}s', flush=True)
        R = {'train_last100': [sum(x[i] for x in log[-100:]) / min(100, len(log)) for i in range(4)],
             'heldout': S.evaluate(ev, rng=random.Random(5)),
             'big_memory': S.evaluate(big, max_q=10_000, rng=random.Random(6)),
             'minutes': (time.time() - t0) / 60}
        if mq:
            R['mquake'] = S.evaluate(mq, max_q=10_000, rng=random.Random(7))
        results[arm] = R
        print(arm, json.dumps(R, indent=1), flush=True)
        json.dump(results, open(args.out, 'w'), indent=1)
        del S
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    # summary table
    print('\n=== exact match (held-out worlds | 8-world memory | MQuAKE real names) ===')
    print(f"{'arm':15s} {'fwd':>6s} {'rev':>6s} {'two':>6s} | {'fwd':>6s} {'rev':>6s} {'two':>6s} | {'fwd':>6s} {'rev':>6s} {'two':>6s} | ctx")
    for arm, R in results.items():
        row = []
        for key in ('heldout', 'big_memory', 'mquake'):
            for t in ('fwd', 'rev', 'two'):
                v = R.get(key, {}).get(t)
                row.append(f'{v[0]:6.3f}' if v else '   n/a')
        ctx = R['heldout'].get('_ctx', (0, 0))[0]
        print(f'{arm:15s} ' + ' '.join(row[:3]) + ' | ' + ' '.join(row[3:6]) + ' | ' + ' '.join(row[6:]) + f' | {ctx:.0f}')
    print('\n=== retrieval recall (held-out: read 1, read 2) ===')
    for arm, R in results.items():
        h = R['heldout']
        print(f"{arm:15s} r1 {h.get('_r1', (0,))[0]:.3f}  r2 {h.get('_r2', (0,))[0]:.3f}  links prec {h.get('_link_prec', (0,))[0]:.3f} rec {h.get('_link_rec', (0,))[0]:.3f}")
    verdicts(results)


def verdicts(results):
    """Pre-registered kill rules (Round 32, merged test)."""
    def g(arm, split, t):
        v = results.get(arm, {}).get(split, {}).get(t)
        return v[0] if v else None
    out = []
    def rule(name, ok, detail):
        out.append(f"{'PASS' if ok else 'FAIL' if ok is not None else 'n/a '}  {name}: {detail}")
    f, r = 'memfull', 'rag2'
    if f in results and r in results:
        d = {sp: (g(f, sp, 'rev') - g(r, sp, 'rev'), g(f, sp, 'two') - g(r, sp, 'two')) for sp in ('heldout', 'big_memory')
             if g(f, sp, 'rev') is not None}
        cost = (results[f]['heldout']['_ctx'][0], results[r]['heldout']['_ctx'][0])
        ok = all(a >= 0.05 and b >= 0.05 for a, b in d.values()) and cost[0] < cost[1]
        rule('K-main memfull beats rag2 by >=5 pts on reverse AND two-hop (held-out and 8-world memory), at lower cost', ok,
             ', '.join(f'{sp}: rev {a:+.3f} two {b:+.3f}' for sp, (a, b) in d.items()) + f'; context {cost[0]:.0f} vs {cost[1]:.0f}')
    if 'mem2way' in results and 'mem1way' in results:
        a = g('mem2way', 'heldout', 'rev') - g('mem1way', 'heldout', 'rev')
        rule('K-twoway two-way filing adds >=20 pts on reverse', a >= 0.20, f'{a:+.3f}')
    if 'memfull' in results and 'mem2way_twice' in results:
        a = g('memfull', 'heldout', 'two') - g('mem2way_twice', 'heldout', 'two')
        rule('K-link integrative linking adds >=10 pts on two-hop (else drop linking)', a >= 0.10, f'{a:+.3f}')
    if 'mem_gold' in results and 'ctx_gold' in results:
        a = g('ctx_gold', 'heldout', 'fwd') - g('mem_gold', 'heldout', 'fwd')
        rule('K-capacity 4 vectors carry a fact: mem_gold within 10 pts of ctx_gold on forward', a <= 0.10, f'gap {a:+.3f}')
    print('\n=== kill rules ===')
    print('\n'.join(out) if out else '(need more arms)')


if __name__ == '__main__':
    main()
