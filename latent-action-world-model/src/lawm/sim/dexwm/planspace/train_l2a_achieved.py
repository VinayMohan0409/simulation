"""Latent-to-action map (L2A) trained on the ACHIEVED hand motion of each model step, not on the commanded targets.

Why: the commanded labels (window_actions) are "commanded target minus the measured start state"; at step 1 that includes the controller's lag
(about 3 cm), which the latent (encoder of frames k-1 -> k) never sees, so L2A could not learn step 1, the step the L2A arm executes.
Here the label of step k is the measured change of the hand state between frames k-1 and k: wrist position (world frame), wrist rotation
(world-frame rotation vector) and finger angles, read from the recorded root_pose and q (proprioception). At execution the L2A arm then sends
"current hand state + predicted step", the same kind of target the IK arm sends (the predicted achieved state).

Same architecture, data split, training loop and best-validation selection as train_g.fit_l2a; A2L is not touched.
Writes <out>/l2a_achieved.pt and <out>/l2a_achieved_report.json (per-step held-out and training errors vs always predicting the mean step)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lawm.sim import model_adapter
from lawm.sim.dexwm.planspace.common import CHECKPOINT, CONTROL_PER_MODEL_STEP, N_STEPS, PRIOR, write_json
from lawm.sim.dexwm.planspace.train_g import fit_l2a, l2a_batch, stats_of


def achieved_actions(rec, s):
    """(5, 22) cumulative achieved hand motion of the window starting at control step s, relative to frame 0 (as window_actions, but measured states)."""
    idx = [s + CONTROL_PER_MODEL_STEP * k for k in range(N_STEPS + 1)]
    P = rec["root_pose"][idx].astype(np.float64)
    q = rec["q"][idx].astype(np.float64)
    R0 = P[0, :3, :3]
    dpos = P[1:, :3, 3] - P[0, :3, 3]
    drot = Rotation.from_matrix(P[1:, :3, :3] @ R0.T).as_rotvec()
    return np.concatenate([dpos, drot, q[1:] - q[0]], axis=1).astype(np.float32)


def per_step(m, lab, rows, device):
    """Per-step RMS errors (wrist mm, wrist deg, fingers rad) of L2A and of the mean step, and the true step size, on some windows."""
    zn, prev, c, p, inc_n = l2a_batch(lab, rows, m, device)
    with torch.no_grad():
        pred = m.increments(zn)
    true = inc_n * m.inc_std + m.inc_mean
    out = {}
    for name, guess in (("L2A", pred), ("mean_step", m.inc_mean.expand_as(true))):
        e = guess - true
        out[name] = dict(wrist_mm=(e[..., :3].norm(dim=-1).pow(2).mean(0).sqrt() * 1000).cpu().numpy().round(1).tolist(),
                         wrist_deg=np.degrees(e[..., 3:6].norm(dim=-1).pow(2).mean(0).sqrt().cpu().numpy()).round(2).tolist(),
                         fingers_rad=e[..., 6:].pow(2).mean(dim=(0, 2)).sqrt().cpu().numpy().round(4).tolist())
    out["true_step"] = dict(wrist_mm=(true[..., :3].norm(dim=-1).mean(0) * 1000).cpu().numpy().round(1).tolist(),
                            wrist_deg=np.degrees(true[..., 3:6].norm(dim=-1).mean(0).cpu().numpy()).round(2).tolist(),
                            fingers_rad=true[..., 6:].abs().mean(dim=(0, 2)).cpu().numpy().round(4).tolist())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="encoder_latents.npz of the fine-tuned model")
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True, help="the maps folder")
    p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args()
    out = Path(a.out)
    logf = open(out / "train_l2a_achieved.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab = dict(np.load(a.labels))
    cache = {}
    act = np.zeros_like(lab["act"])
    for i, (f, s) in enumerate(zip(lab["rec_file"], lab["start"])):
        f = str(f)
        if f not in cache:
            cache[f] = {k: v for k, v in np.load(Path(a.play) / f).items() if k in ("root_pose", "q")}
        act[i] = achieved_actions(cache[f], int(s))
    lab["act"] = act  # the only change: the labels L2A learns are the achieved steps
    train, val = np.flatnonzero(lab["split"] == "train"), np.flatnonzero(lab["split"] == "val")
    z_mean, z_std = model_adapter.load_prior(PRIOR, CHECKPOINT)
    st = stats_of(lab, train, z_mean, z_std)
    log(f"{len(train)} train / {len(val)} val windows; labels = achieved hand steps; device {device}")
    m = fit_l2a(lab, train, val, st, device, a.steps, log)
    m.save(out / "l2a_achieved.pt", extra=dict(labels="achieved hand steps (root_pose, q)", checkpoint=str(CHECKPOINT)))
    rep = dict(labels="achieved hand steps", heldout_windows=len(val), val=per_step(m, lab, val, device), train=per_step(m, lab, train, device))
    write_json(out / "l2a_achieved_report.json", rep)
    for split in ("val", "train"):
        log(f"{split}: " + json.dumps(rep[split]))


if __name__ == "__main__":
    main()
