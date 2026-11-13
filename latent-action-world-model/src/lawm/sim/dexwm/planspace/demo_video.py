"""Video of a stored DexWM demonstration of a task (start frame to target frame), from the same 2 production cameras as the closed-loop runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np

from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, eval_dir
from lawm.sim.dexwm.planspace.production import ProductionScene
from lawm.sim.dexwm.runner import load_config


def main():
    """python -m lawm.sim.dexwm.planspace.demo_video --demo demo_133 [--task reach] --out FILE.mp4."""
    p = argparse.ArgumentParser()
    p.add_argument("--demo", required=True)
    p.add_argument("--task", default="reach")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = load_config(DEXWM_CONFIG)
    episode = benchmark.load_episode(eval_dir(), a.demo)
    sim = kitchen.open_demo(episode, cfg.env)
    try:
        obs = sim.reset(a.task, cfg.scene_seed)
        prod = ProductionScene(sim, seed=int(a.demo.split("_")[1]))
        prod.set_start(obs.root_pose)
        start, target = benchmark.get_indices(episode["waypoints"], a.task)
        frames = []
        for f in range(start, target + 1):
            sim.switch(f)
            frames.append(np.concatenate([v[0] for v in prod.views(sim.data)], axis=1))
    finally:
        sim.env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=int(cfg.env.control_freq))
    print(f"{a.demo} {a.task}: frames {start}..{target} ({len(frames)}) -> {a.out}")


if __name__ == "__main__":
    main()
