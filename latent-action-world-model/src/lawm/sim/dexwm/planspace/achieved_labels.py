"""A copy of a labels file (encoder_latents.npz) whose action chunks are the ACHIEVED hand motion instead of the commanded targets.

act[i] (5, 22) becomes the measured hand state of frames 1..5 relative to frame 0 of the window (wrist position, world-frame rotation vector,
finger angles; train_l2a_achieved.achieved_actions), so step 1 no longer contains the controller's lead (~46 mm commanded vs ~21 mm achieved).
Used to retrain A2L and the action prior, so the action arms plan and execute in the same convention as the IK arms and L2A:
"the hand state after the step". Everything else in the file is unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lawm.sim.dexwm.planspace.train_l2a_achieved import achieved_actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lab = dict(np.load(a.labels))
    cache, act = {}, np.zeros_like(lab["act"])
    for i, (f, s) in enumerate(zip(lab["rec_file"], lab["start"])):
        f = str(f)
        if f not in cache:
            cache[f] = {k: v for k, v in np.load(Path(a.play) / f).items() if k in ("root_pose", "q")}
        act[i] = achieved_actions(cache[f], int(s))
    lab["act"] = act
    np.savez(a.out, **lab)
    inc = np.diff(np.concatenate([np.zeros_like(act[:, :1]), act], 1), axis=1)
    print(f"{len(act)} windows -> {a.out}; achieved wrist step mm per step {np.round(np.linalg.norm(inc[..., :3], axis=-1).mean(0) * 1000, 1)}")


if __name__ == "__main__":
    main()
