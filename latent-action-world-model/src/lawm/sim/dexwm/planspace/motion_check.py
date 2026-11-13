"""Does the world model predict the play motion? On held-out play windows (in the training frame): true vs predicted hand motion size and direction from the encoder's latents (the true frames), per step."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lawm.sim.dexwm.planspace.common import wait_for_gpu, write_json
from lawm.sim.dexwm.planspace.labels import load
from lawm.sim.dexwm.planspace.train_g import Windows, decode_window


def main():
    """python -m lawm.sim.dexwm.planspace.motion_check --labels labels.npz --play DIR --out motion_check.json [--windows 200]."""
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=200)
    a = p.parse_args()
    wait_for_gpu(2500)
    lab = dict(np.load(a.labels))
    val = np.flatnonzero(lab["split"] == "val")
    rows = np.random.default_rng(0).choice(val, min(a.windows, len(val)), replace=False)
    model, dev, wins = load(False), torch.device("cuda"), Windows(a.play, lab)
    true_mm, pred_mm, cos = [], [], []
    with torch.no_grad():
        for i in rows:
            frames, seg = wins.get(i)
            pred, true = decode_window(model, frames, torch.as_tensor(lab["z"][i], device=dev).float())
            h = torch.as_tensor(seg == 0, device=dev)
            p_, t_ = pred[:, h], true[:, h]  # (5, n_hand, 3)
            true_mm.append((t_.norm(dim=-1).mean(1) * 1000).cpu().numpy())
            pred_mm.append((p_.norm(dim=-1).mean(1) * 1000).cpu().numpy())
            cos.append(torch.nn.functional.cosine_similarity(p_.mean(1), t_.mean(1), dim=-1).cpu().numpy())
    true_mm, pred_mm, cos = map(np.asarray, (true_mm, pred_mm, cos))
    moving = true_mm[:, -1] > 10.0
    rep = dict(windows=len(rows), moving_windows=int(moving.sum()),
               true_hand_mm_per_step=true_mm.mean(0).round(1).tolist(), predicted_hand_mm_per_step=pred_mm.mean(0).round(1).tolist(),
               size_ratio_step5=round(float(pred_mm[moving, -1].mean() / true_mm[moving, -1].mean()), 3),
               direction_cosine_step5_median=round(float(np.median(cos[moving, -1])), 3), direction_cosine_step5_mean=round(float(cos[moving, -1].mean()), 3))
    write_json(Path(a.out), rep)
    print(rep, flush=True)


if __name__ == "__main__":
    main()
