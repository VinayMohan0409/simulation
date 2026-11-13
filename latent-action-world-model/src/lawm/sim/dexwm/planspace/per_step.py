"""Per-step hand motion on held-out windows: true motion, predicted motion and error (mm) at each of the 5 model steps, for latents from the encoder and from A2L maps.
The model is PLANSPACE_CHECKPOINT; windows are the same 300 held-out ones train_g evaluates (rng 2)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from lawm.sim.dexwm.planspace.g_model import ActionToLatent
from lawm.sim.dexwm.planspace.labels import load
from lawm.sim.dexwm.planspace.train_g import Windows, decode_window


@torch.no_grad()
def main():
    """python -m lawm.sim.dexwm.planspace.per_step --labels L --play DIR --a2l name=path [...] --out J."""
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--play", required=True)
    p.add_argument("--a2l", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=300)
    a = p.parse_args()
    dev = torch.device("cuda")
    model = load()
    lab = dict(np.load(a.labels))
    val = np.flatnonzero(lab["split"] == "val")
    rows = np.random.default_rng(2).choice(val, min(a.windows, len(val)), replace=False)
    maps = {k: ActionToLatent.load(v, dev) for k, v in (s.split("=", 1) for s in a.a2l)}
    wins = Windows(a.play, lab)
    acc = {k: dict(pred=[], err=[]) for k in ["encoder", *maps]}
    true_mm = []
    for i in rows:
        frames, seg = wins.get(i)
        h = torch.as_tensor(seg == 0, device=dev)
        zs = dict(encoder=torch.as_tensor(lab["z"][i], device=dev).float())
        for k, g in maps.items():
            u = (torch.as_tensor(lab["act"][i], dtype=torch.float32, device=dev)[None] - g.act_mean) / g.act_std
            zs[k] = g.latents(u)[0]
        for k, z in zs.items():
            pred, true = decode_window(model, frames, z)
            pred, true = pred[:, h], true[:, h]
            acc[k]["pred"].append((pred.norm(dim=-1).mean(1) * 1000).cpu().numpy())
            acc[k]["err"].append(((pred - true).norm(dim=-1).mean(1) * 1000).cpu().numpy())
        true_mm.append((true.norm(dim=-1).mean(1) * 1000).cpu().numpy())
    r = dict(windows=len(rows), true_mm=np.mean(true_mm, 0).round(1).tolist(), zero_err_mm=np.mean(true_mm, 0).round(1).tolist())
    for k, v in acc.items():
        r[k] = dict(pred_mm=np.mean(v["pred"], 0).round(1).tolist(), err_mm=np.mean(v["err"], 0).round(1).tolist())
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
