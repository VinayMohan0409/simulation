"""Render once, identify many times: the frames of place episodes from the production cameras, saved per demo so the object identification (object_track.py) can be
iterated on in seconds instead of rebuilding a kitchen every time.

Per demo, <out>/<demo>.npz holds frames j = 0 .. n-1 (frame_ids[j]: 0 = the place start, n-1 = the place target, between: every --stride demo frames, used ONLY to test
the per-replan label transport offline; the planner never sees them):
  f<j>_v<i>_rgb, f<j>_v<i>_depth, f<j>_v<i>_K, f<j>_v<i>_pose     camera i (cameras "dexwm": robot0_robotview_2 on DexWM's 224 x 392 grid; "two": free + robot0_robotview)
  f<j>_v<i>_true                                                  the simulator's object mask: SCORING ONLY, never an input
  f<j>_hand (n_hand, 3) FK hand points, f<j>_robot (n, 3) FK robot surface points, f<j>_root (4, 4) hand root pose, f<j>_object_body (3,) scoring only
  finger_mask (n_hand,) bool, frame_ids, n_views, lo / hi (the production crop box)
  pose_centred: True when the camera poses are the mono centre (production.py default since 2026-11-13); files without it (or False) hold the legacy LEFT-eye
  poses, which load_frame moves to the centre (what object_track.py needs)
Usage: python -m lawm.sim.dexwm.planspace.place_frames --out DIR [--cameras dexwm|two] [--stride 5] [--demos ...]   (default: all 50 DexWM demos; resumable per demo)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, eval_dir
from lawm.sim.dexwm.planspace.production import ProductionScene
from lawm.sim.dexwm.planspace.segment_check import true_masks
from lawm.sim.dexwm.runner import load_config
from lawm.sim.dexwm.scene.scene_cloud import visual_geom_ids

EPISODES = Path(__file__).resolve().parents[5] / "configs/stage3/dexwm_episodes.json"


IPD = 0.068  # MuJoCo's default stereo eye distance (vis.global_.ipd): the saved left-eye poses are moved by ipd/2 along the camera's x axis


def load_frame(z, j):
    """(views, hand, robot, root) of frame j of a place_frames file, camera poses at the mono centre."""
    views = []
    for i in range(int(z["n_views"])):
        pose = z[f"f{j}_v{i}_pose"].copy()
        if not ("pose_centred" in z and bool(z["pose_centred"])):
            pose[:3, 3] += pose[:3, 0] * IPD / 2
        views.append((z[f"f{j}_v{i}_rgb"], z[f"f{j}_v{i}_depth"], z[f"f{j}_v{i}_K"], pose))
    return views, z[f"f{j}_hand"], z[f"f{j}_robot"], z[f"f{j}_root"]


def _truth(prod, data, obj_geoms, n_views, shape):
    """The simulator's object mask per view (scoring only); MuJoCo's segmentation render sometimes indexes past ngeom (IndexError) -> all-False masks."""
    try:
        return true_masks(prod, data, obj_geoms)
    except IndexError:
        prod.renderer.disable_segmentation_rendering()
        return [np.zeros(shape, bool)] * n_views


def frame(sim, prod, obj_geoms):
    k = sim.kinematics
    root = k.bodies[k.root]
    R, p, q = sim.data.xmat[root].reshape(3, 3).copy(), sim.data.xpos[root].copy(), sim.data.qpos[k.qpos_adr].copy()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    out = dict(hand=k.points(R, p, q).astype(np.float32), robot=prod.robot(sim.data).astype(np.float32), root=T, object_body=sim.data.xpos[sim.object_id].copy())
    views = prod.views(sim.data)
    truth = _truth(prod, sim.data, obj_geoms, len(views), views[0][1].shape)
    for i, ((rgb, depth, K, pose), tm) in enumerate(zip(views, truth)):
        out.update({f"v{i}_rgb": rgb, f"v{i}_depth": depth.astype(np.float32), f"v{i}_K": K, f"v{i}_pose": pose, f"v{i}_true": tm})
    return out, len(views)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cameras", default="dexwm", choices=["dexwm", "two"])
    p.add_argument("--stride", type=int, default=5, help="demo frames between saved intermediate frames (transport test only)")
    p.add_argument("--demos", nargs="*", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(DEXWM_CONFIG)
    for demo in a.demos or [e["demo"] for e in json.loads(EPISODES.read_text())["episodes"]]:
        f = out / f"{demo}.npz"
        if f.exists():
            continue
        sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), demo), cfg.env)
        try:
            obs = sim.reset("place", cfg.scene_seed)
            prod = ProductionScene(sim, seed=int(demo.split("_")[1]), cameras=a.cameras)
            prod.set_start(obs.root_pose)
            obj_geoms = visual_geom_ids(sim.model, sim.object_id)
            start, target = benchmark.get_indices(sim.episode["waypoints"], "place")
            ids = list(range(start, target, a.stride)) + [target]
            data = dict(finger_mask=sim.finger_mask, frame_ids=np.array(ids), lo=prod.lo, hi=prod.hi, pose_centred=prod.centre_pose)
            for j, fr in enumerate(ids):
                sim.switch(fr)
                fd, n = frame(sim, prod, obj_geoms)
                data.update({f"f{j}_{key}": v for key, v in fd.items()})
            data["n_views"] = n
        finally:
            sim.env.close()
        np.savez_compressed(out / f".{demo}.tmp.npz", **data)
        (out / f".{demo}.tmp.npz").rename(f)
        print(f"{demo}: frames {ids[0]}..{ids[-1]} ({len(ids)}), true object px start {int(data['f0_v0_true'].sum())} goal {int(data[f'f{len(ids) - 1}_v0_true'].sum())}", flush=True)


if __name__ == "__main__":
    main()
