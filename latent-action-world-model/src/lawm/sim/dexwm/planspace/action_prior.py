"""The action-space search prior: per-step Gaussian of the play data's per-step action INCREMENTS (wrist dpos, wrist drot, finger dq of each model step).

Why: the action chunks are stored cumulatively (every step relative to the window start). A diagonal Gaussian fitted to the cumulative columns
drops their strong correlation (0.8 between step 1 and step 5), so independent draws gave steps of 10-15 cm instead of about 2 cm. Sampling each
step's increment independently and summing them (ActionAdapter with this prior) gives realistic steps, the same way the latent arm samples one
latent per step. Training split only. Writes <out>.npz with inc_mean, inc_std (5, 22)."""
from __future__ import annotations

import argparse

import numpy as np


def increments(act):
    """Per-step increments (n, 5, 22) of cumulative chunks (n, 5, 22)."""
    return np.diff(np.concatenate([np.zeros_like(act[:, :1]), act], axis=1), axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="encoder_latents.npz (holds act, split)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lab = np.load(a.labels)
    inc = increments(lab["act"][lab["split"] == "train"].astype(np.float64))
    m, s = inc.mean(0), np.maximum(inc.std(0), 1e-4)
    np.savez(a.out, inc_mean=m.astype(np.float32), inc_std=s.astype(np.float32), n_windows=len(inc))
    print(f"action prior from {len(inc)} training windows -> {a.out}; wrist step mm per step {np.round(np.linalg.norm(inc[..., :3], axis=-1).mean(0) * 1000, 1)}")


if __name__ == "__main__":
    main()
