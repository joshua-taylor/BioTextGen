"""Data price of random per-token perturbation: how much more data than exact credit to reach the same loss."""
import json, numpy as np
rows = [json.loads(l) for l in open("res_e6.jsonl") if json.loads(l)["rule"].startswith("K")]
by = {}
for r in rows: by[(r["H"], r["kw"], r["K"])] = r["hist"]
def data_to_reach(hist, target):
    xs = [h[1] for h in hist]; ys = [h[3] for h in hist]
    if target > ys[0]: return None
    for i in range(1, len(ys)):
        if ys[i] <= target:
            # linear interpolation in log-data
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            f = (y0 - target) / (y0 - y1); return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return None
for (H, kw, K), hist in sorted(by.items()):
    if K == 0: continue
    ex = by.get((H, kw, 0))
    if not ex: continue
    out = []
    for h in hist:
        d = data_to_reach(ex, h[3])
        out.append(f">{h[1] / ex[0][1]:.1f}" if d is None else round(h[1] / d, 1))
    cone = kw
    print(f"H{H} kw{kw} K{K} cone={cone}: price (data multiple) at each eval = {out}")

# power-law fit of the exact curve, loss = a + b * n^-alpha, inverted to estimate the data price of K-draw runs
from scipy.optimize import curve_fit
print("\npower-law fit of exact curve, then invert:")
for (H, kw, K), hist in sorted(by.items()):
    ex = by.get((H, kw, 0))
    if K == 0 or not ex: continue
    n = np.array([h[1] for h in ex], float) / 1e6; y = np.array([h[3] for h in ex])
    f = lambda n, a, b, al: a + b * n ** (-al)
    (a, b, al), _ = curve_fit(f, n, y, p0=(1.5, 0.8, 0.5), maxfev=20000)
    out = []
    for h in hist:
        t = h[3]
        need = ((t - a) / b) ** (-1 / al) if t > a else float("inf")
        out.append(round(h[1] / 1e6 / need, 1))
    print(f"H{H} kw{kw} K{K}: fit a={a:.2f} b={b:.2f} alpha={al:.2f}; price at each eval = {out}")
