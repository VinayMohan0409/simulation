"""The latent prior (per-step mean and spread of the encoder's latents) of a checkpoint, from its play-window labels (training split); stored with the checkpoint's hash like the original prior."""
from __future__ import annotations

import argparse

import numpy as np

from lawm.sim import model_adapter


def main():
    """python -m lawm.sim.dexwm.planspace.prior_from_labels --labels labels.npz --checkpoint CKPT --out prior.npz."""
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lab = np.load(a.labels)
    z = lab["z"][lab["split"] == "train"]
    model_adapter.save_prior(a.out, (z.mean(0), z.std(0)), a.checkpoint, len(z))
    print(f"prior from {len(z)} play windows -> {a.out}")


if __name__ == "__main__":
    main()
