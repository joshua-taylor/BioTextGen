"""Train one model (given config), then score the same 40k val chars with growing amounts of history in memory."""
import json, sys, time, numpy as np
import sb2
def main(out, tag, cfg_json, caps_json):
    cfg = json.loads(cfg_json); c = dict(sb2.DEFAULT); c.update(cfg); c["evals"] = 1
    c["return_model"] = 1; r = sb2.run(**c); model = r.pop("model")
    for hist, cap in json.loads(caps_json):
        for kind in ("unit", "pair"):
            t = time.time(); v = sb2.evaluate_long(model, hist, cap, kind=kind, bcap=64)
            rec = dict(tag=tag, kind=kind, cfg=cfg, history=hist, cap=cap, bpc=round(v[0], 4), copy=round(v[1], 4),
                       other=round(v[2], 4), sec=round(time.time() - t, 1), train_hist=r["hist"])
            print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
if __name__ == "__main__":
    main(*sys.argv[1:5])
