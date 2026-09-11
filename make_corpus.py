"""
Build a character-level corpus for the probe.

Order of preference:
  1. --path <file>                      any plain text you already have
  2. HuggingFace datasets wikitext-103  (works on Kaggle with internet enabled)
  3. Kaggle local dataset directories    (works with internet disabled)
  4. docstring harvest from installed packages  (last resort, always available)

The probe is character-level, so any large English text works. WikiText is the
right choice if you want numbers comparable to the rest of your benchmarks.
"""
import argparse, glob, os, re, sys


def from_hf(n_chars):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    buf, total = [], 0
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        buf.append(t); total += len(t)
        if total >= n_chars:
            break
    return "".join(buf)[:n_chars]


def from_kaggle_input(n_chars):
    pats = ["/kaggle/input/**/*.txt", "/kaggle/input/**/*.raw", "/kaggle/input/**/wiki*"]
    files = []
    for p in pats:
        files += [f for f in glob.glob(p, recursive=True) if os.path.isfile(f)]
    files.sort(key=os.path.getsize, reverse=True)
    if not files:
        raise FileNotFoundError("no text files under /kaggle/input")
    out, total = [], 0
    for f in files:
        s = open(f, encoding="utf8", errors="ignore").read()
        out.append(s); total += len(s)
        if total >= n_chars:
            break
    return "".join(out)[:n_chars]


def from_docstrings(n_chars):
    import ast, random
    files = []
    for root in sys.path:
        if root and os.path.isdir(root):
            files += glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
    random.seed(0); random.shuffle(files)
    txt, total = [], 0
    for f in files[:6000]:
        try:
            tree = ast.parse(open(f, encoding="utf8", errors="ignore").read())
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                d = ast.get_docstring(n)
                if d and len(d) > 120:
                    txt.append(d); total += len(d)
        if total >= n_chars:
            break
    return re.sub(r"\s+", " ", "\n\n".join(txt))[:n_chars]


def build(out="corpus.txt", n_chars=2_000_000, path=None):
    if path:
        s = open(path, encoding="utf8", errors="ignore").read()[:n_chars]
        src = path
    else:
        for name, fn in [("wikitext-103 (HF)", from_hf),
                         ("/kaggle/input", from_kaggle_input),
                         ("docstring harvest", from_docstrings)]:
            try:
                s = fn(n_chars); src = name
                break
            except Exception as e:
                print(f"  {name}: unavailable ({type(e).__name__})")
        else:
            raise SystemExit("no corpus source available")
    open(out, "w", encoding="utf8").write(s)
    print(f"wrote {out}: {len(s):,} chars from {src}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--n-chars", type=int, default=2_000_000)
    ap.add_argument("--path")
    a = ap.parse_args()
    build(a.out, a.n_chars, a.path)
