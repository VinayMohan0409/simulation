"""Random play in non-evaluation DexWM kitchens, recorded every control step as scene clouds, hand states and the commanded hand targets.
--kinds explore (task-agnostic, the default from 2026-11-12): random wrist paths anywhere in a workspace box around the kitchen's hand start, rotations within 20 deg of the
start orientation (palm stays down), random finger motion with grasp-like closures; motions are chained (each starts where the previous ended), the kitchen resets every
CHAIN motions; nothing aims at the object. --kinds legacy: the old object-biased wander / grasp / fingers / push mix (archived; do not use for new data)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lawm.sim.dexwm.control.executor import ACTION_DIM
from lawm.sim.dexwm.control.hand_fit import hand_state_to_action
from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, play_scenes, write_json
from lawm.sim.dexwm.runner import load_config

T0 = 40  # knot clock of the motion shapes (control steps of the original 4 s motions)
SLOW = 2  # every motion is played SLOW times slower: about 2 cm of hand motion per 0.2 s model step, as in the training data
T = T0 * SLOW  # control steps per motion (8 s at 10 Hz)
ABDUCTION = (0, 4, 8)
KINDS = ("wander", "grasp", "fingers", "push")
KIND_P = (0.4, 0.4, 0.1, 0.1)
CHAIN = 5  # explore: motions per kitchen reset
BOX_XY = 0.40  # explore: workspace half-width (m) around the hand's reset position, in x and y
BOX_Z = (-0.25, 0.30)  # explore: workspace height range (m) around the hand's reset height
FLOOR_ABOVE_OBJECT = 0.06  # explore: the wrist stays at least this high above the object's centre (the counter top)
MAX_ROT = np.radians(20.0)  # explore: rotation from the reset (palm-down) orientation
PATH_M = (0.6, 0.9)  # explore: wrist path length per motion (m): about 2 cm per 0.2 s model step over T control steps


def finger_poses(limits):
    """(open, closed) finger angles of the hand within its limits."""
    lo, hi = limits[:, 0], limits[:, 1]
    open_q = np.clip(np.zeros(16), lo, hi)
    closed = lo + 0.8 * (hi - lo)
    closed[list(ABDUCTION)] = 0.0
    closed[12], closed[13] = lo[12] + 0.9 * (hi[12] - lo[12]), lo[13] + 0.6 * (hi[13] - lo[13])
    return open_q, closed


def random_fingers(rng, open_q, closed, limits, s=None):
    """Finger angles: per finger an independent (or shared) closure in [0, 1] plus noise, clipped to the limits."""
    if s is None:
        s = rng.uniform(0, 1, 4) if rng.random() < 0.5 else np.full(4, rng.uniform(0, 1))
    s = np.repeat(np.broadcast_to(s, (4,)), 4)
    q = open_q + s * (closed - open_q) + rng.normal(0, 0.1, 16)
    return np.clip(q, limits[:, 0], limits[:, 1])


def spline(times, values):
    """Values at control steps 0..T-1 of a clamped cubic spline through (times, values), times on the T0 knot clock."""
    return CubicSpline(np.asarray(times, float), np.asarray(values, float), bc_type="clamped", axis=0)(np.linspace(0, T0 - 1, T))


def rand_rotvec(rng, max_angle=0.52, sigma=0.3):
    """A random rotation vector of norm at most max_angle."""
    rv = rng.normal(0, sigma, 3)
    n = np.linalg.norm(rv)
    return rv * (max_angle / n) if n > max_angle else rv


def motion(kind, rng, p0, q0, obj, palm_off, limits):
    """Per control step targets: wrist position (T, 3), wrist rotation as a world-frame rotation vector applied to the start rotation (T, 3), finger angles (T, 16)."""
    open_q, closed = finger_poses(limits)
    grasp_at = obj - palm_off
    if kind == "wander":
        lam = rng.uniform(0.3, 1.0)
        c = p0 + lam * (grasp_at - p0)
        knots = [p0] + [c + np.array([rng.uniform(-.15, .15), rng.uniform(-.15, .15), rng.uniform(-.03, .15)]) for _ in range(3)]
        pos = spline([0, 13, 26, T0 - 1], knots)
        rot = spline([0, 13, 26, T0 - 1], [np.zeros(3)] + [rand_rotvec(rng) for _ in range(3)])
        fq = spline([0, 10, 20, 30, T0 - 1], [q0] + [random_fingers(rng, open_q, closed, limits) for _ in range(4)])
    elif kind == "grasp":
        target = grasp_at + rng.normal(0, 0.02, 3) + np.array([0, 0, rng.uniform(0.0, 0.04)])
        lift = target + np.array([0, 0, rng.uniform(0.0, 0.10)])
        pos = spline([0, 10, 18, 26, T0 - 1], [p0, target + np.array([0, 0, 0.08]), target, target, lift])
        rv = rand_rotvec(rng, 0.3, 0.1)
        rot = spline([0, 18, T0 - 1], [np.zeros(3), rv, rv])
        pre = random_fingers(rng, open_q, closed, limits, s=rng.uniform(0, 0.3))
        shut = random_fingers(rng, open_q, closed, limits, s=rng.uniform(0.5, 1.0))
        fq = spline([0, 10, 18, 26, T0 - 1], [q0, pre, pre, shut, shut])
    elif kind == "fingers":
        pos, rot = np.repeat(p0[None], T, 0), np.zeros((T, 3))
        fq = spline([0, 8, 16, 24, 32, T0 - 1], [q0] + [random_fingers(rng, open_q, closed, limits) for _ in range(5)])
    elif kind == "push":
        a = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(a), np.sin(a), 0.0])
        z = np.array([0, 0, rng.uniform(0.0, 0.03)])
        pos = spline([0, 12, T0 - 1], [p0, grasp_at - 0.10 * d + z, grasp_at + 0.12 * d + z])
        rot = np.zeros((T, 3))
        half = random_fingers(rng, open_q, closed, limits, s=0.4)
        fq = spline([0, 12, T0 - 1], [q0, half, half])
    else:
        raise ValueError(kind)
    return pos, rot, np.clip(fq, limits[:, 0], limits[:, 1])


def explore_box(p_ref, obj, floor_above_object=FLOOR_ABOVE_OBJECT):
    """(lo, hi) corners of the explore workspace box for a kitchen reset at wrist position p_ref."""
    lo = np.r_[p_ref[:2] - BOX_XY, max(p_ref[2] + BOX_Z[0], obj[2] + floor_above_object)]
    hi = np.r_[p_ref[:2] + BOX_XY, p_ref[2] + BOX_Z[1]]
    return lo, np.maximum(hi, lo + 0.05)


def explore_motion(rng, p0, R0, q0, anchor, limits):
    """Task-agnostic motion from the current state: a random walk of 3 knots towards uniform points of the box (so directions are unbiased, and the box is never left),
    path length PATH_M; wrist rotation targets within MAX_ROT of the reset orientation (absolute, so chaining never accumulates rotation); fingers random, or a
    grasp-like close-and-reopen in half the motions. Returns (pos, rot as world rotvec applied to R0, fq) like motion()."""
    lo, hi, R_ref = anchor["lo"], anchor["hi"], anchor["R_ref"]
    seg = rng.uniform(*PATH_M) / 3 * rng.uniform(0.7, 1.3, 3)
    knots, prev = [p0], np.clip(p0, lo, hi)
    for li in seg:
        target = rng.uniform(lo, hi)
        d = target - prev
        prev = prev + d * min(1.0, li / max(np.linalg.norm(d), 1e-9))
        knots.append(prev)
    pos = np.clip(spline([0, 13, 26, T0 - 1], knots), lo, hi)
    rv_now = Rotation.from_matrix(R0 @ R_ref.T).as_rotvec()
    rel = spline([0, 13, 26, T0 - 1], [rv_now] + [rand_rotvec(rng, MAX_ROT, 0.2) for _ in range(3)])
    n = np.linalg.norm(rel, axis=1, keepdims=True)
    rel = np.where(n > MAX_ROT, rel * MAX_ROT / np.maximum(n, 1e-9), rel)
    rot = (Rotation.from_rotvec(rel) * Rotation.from_matrix(R_ref @ R0.T)).as_rotvec()  # R_t = Rot(rel_t) R_ref = Rot(rot_t) R0
    open_q, closed = finger_poses(limits)
    if rng.random() < 0.5:
        pre = random_fingers(rng, open_q, closed, limits, s=rng.uniform(0, 0.3))
        shut = random_fingers(rng, open_q, closed, limits, s=rng.uniform(0.5, 1.0))
        t_close = int(rng.integers(8, 22))
        fq = spline([0, t_close - 4, t_close + 4, t_close + 12, T0 - 1], [q0, pre, shut, shut, random_fingers(rng, open_q, closed, limits)])
    else:
        fq = spline([0, 10, 20, 30, T0 - 1], [q0] + [random_fingers(rng, open_q, closed, limits) for _ in range(4)])
    return pos, rot, np.clip(fq, limits[:, 0], limits[:, 1])


def palm_offset(sim, obs):
    """World vector from the hand root to the centroid of its palm points."""
    k = sim.kinematics
    pts = k.points(obs.root_pose[:3, :3], obs.root_pose[:3, 3], obs.q)
    return pts[k.point_body == k.root].mean(0) - obs.root_pose[:3, 3]


def hand_object_contact(sim):
    """True when any hand geom touches the task object (logged only: the play contact rate; never a training input)."""
    env = sim.env
    return bool(env.check_contact(env.robots[0].gripper["right"], env.objects["obj"]))


def run_motion(sim, kind, task, rng, prod=None, anchor=None, floor_above_object=FLOOR_ABOVE_OBJECT):
    """Drive one random motion and return the recorded arrays (and, with a ProductionScene, the camera frames: rgb, depth in mm, K, poses), plus the explore anchor.
    anchor None: reset to the task's start state first (cameras and, for explore, the workspace box are fixed there); otherwise continue from the current state."""
    if anchor is None:
        obs = sim.reset(task)
        if prod is not None:
            prod.set_start(obs.root_pose)
    else:
        obs = sim.observe()
    cams = dict(rgb=[], depth_mm=[], K=[], cam_pose=[], n_scene_seen=[])
    p0, R0, q0 = obs.root_pose[:3, 3].copy(), obs.root_pose[:3, :3].copy(), obs.q.copy()
    obj = sim.data.xpos[sim.object_id].copy()
    if kind == "explore":
        if anchor is None:
            lo, hi = explore_box(p0, obj, floor_above_object)
            anchor = dict(p_ref=p0, R_ref=R0, lo=lo, hi=hi)
        pos, rot, fq = explore_motion(rng, p0, R0, q0, anchor, sim.kinematics.limits)
    else:
        pos, rot, fq = motion(kind, rng, p0, q0, obj, palm_offset(sim, obs), sim.kinematics.limits)
    rec = dict(coords=[], root_pose=[], q=[], cmd_pos=[], cmd_rot=[], cmd_q=[], contact=[])
    for t in range(T + 1):
        if t > 0:
            obs = sim.observe()
        if prod is None:
            rec["coords"].append(obs.coords.astype(np.float32))
        else:
            coords, segment, views, seen = prod(sim.data, obs.root_pose, obs.q)
            rec["coords"].append(coords)
            cams["rgb"].append(np.stack([v[0] for v in views]))
            cams["depth_mm"].append(np.stack([np.clip(v[1] * 1000.0, 0, 65535).astype(np.uint16) for v in views]))
            cams["K"].append(np.stack([v[2] for v in views]))
            cams["cam_pose"].append(np.stack([v[3] for v in views]))
            cams["n_scene_seen"].append(seen)
        rec["contact"].append(hand_object_contact(sim))
        rec["root_pose"].append(obs.root_pose)
        rec["q"].append(obs.q)
        if t == T:
            break
        R_t = Rotation.from_rotvec(rot[t]).as_matrix() @ R0
        p, rv, f = hand_state_to_action(R_t, pos[t], fq[t], obs.root_pose[:3, :3], obs.root_pose[:3, 3], obs.ref_pos, obs.ref_rot)
        action = np.zeros(ACTION_DIM, np.float32)
        action[0:3], action[3:6], action[15:31] = p, rv, f
        sim.step(action)
        rec["cmd_pos"].append(pos[t])
        rec["cmd_rot"].append(R_t)
        rec["cmd_q"].append(fq[t])
    out = {k: np.asarray(v, np.float32) for k, v in rec.items()}
    if anchor is not None:
        out.update(box_lo=anchor["lo"], box_hi=anchor["hi"], R_ref=anchor["R_ref"])
    if prod is None:
        out["segment"] = np.asarray(sim.segment)
        return out, None, anchor
    out["segment"] = segment
    out["n_scene_seen"] = np.asarray(cams.pop("n_scene_seen"))
    return out, {k: np.asarray(v) for k, v in cams.items()}, anchor


def collect_scene(args):
    """Pool worker: all motions of one scene into out_dir/<demo>_<i>.npz (finished files are skipped)."""
    folder, demo, n_motions, out_dir, seed, scene_kind, kinds, cameras, floor = args
    out_dir = Path(out_dir)
    todo = [i for i in range(n_motions) if not (out_dir / f"{demo}_{i:03d}.npz").exists()]
    if not todo:
        return demo, 0
    env_settings = load_config(DEXWM_CONFIG).env
    sim = kitchen.open_demo(benchmark.load_episode(folder, demo), env_settings)
    prod = None
    if scene_kind == "production":
        from lawm.sim.dexwm.planspace.production import ProductionScene
        prod = ProductionScene(sim, seed=int(demo.split("_")[1]), cameras=cameras)
    try:
        anchor, last = None, None
        for i in todo:
            rng = np.random.default_rng((seed, int(demo.split("_")[1]), i))
            if kinds == "explore":
                kind, task = "explore", ("grasp" if (i // CHAIN) % 2 else "reach")
                if i % CHAIN == 0 or last != i - 1:  # a new chain, or the previous clip of this chain was recorded in an earlier run
                    anchor = None
            else:
                kind, task, anchor = KINDS[rng.choice(len(KINDS), p=KIND_P)], ("grasp" if i % 2 else "reach"), None
            rec, cams, anchor = run_motion(sim, kind, task, rng, prod, anchor, floor)
            if kinds != "explore":
                anchor = None
            last = i
            path = out_dir / f"{demo}_{i:03d}.npz"
            if cams is not None:  # camera frames in their own file, written first (the clip counts as done once its main file exists)
                np.savez_compressed(out_dir / f"{demo}_{i:03d}.rgbd.npz", **cams)
            np.savez(path.with_suffix(".tmp.npz"), kind=kind, task=task, scene=demo, scene_kind=scene_kind, **rec)
            path.with_suffix(".tmp.npz").replace(path)
    finally:
        sim.env.close()
    return demo, len(todo)


def main():
    """python -m lawm.sim.dexwm.planspace.play --out DIR [--total 480] [--workers 4] [--scenes N]: random play in the non-evaluation kitchens."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=480)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--scenes", type=int, default=None, help="use only the first N scenes (smoke tests)")
    p.add_argument("--scene-index", type=int, nargs="*", default=None, help="only these scene indices (split the kitchens over several jobs)")
    p.add_argument("--per-scene", type=int, default=None, help="motions per kitchen (default: ceil(total / kitchens))")
    p.add_argument("--kinds", choices=("explore", "legacy"), default="explore")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene", choices=("production", "privileged"), default="production", help="production: FK hand + 2 RGB-D cameras (no privileged scene state); privileged: mesh-sampled scene")
    p.add_argument("--cameras", choices=("two", "dexwm"), default="two", help="production cameras: two (planspace) or dexwm (robot0_robotview_2 only, DexWM's input grid)")
    p.add_argument("--floor-above-object", type=float, default=FLOOR_ABOVE_OBJECT, help="explore: lowest wrist height above the object's centre (m); lower = more chance contact")
    a = p.parse_args()
    import multiprocessing as mp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = play_scenes()[:a.scenes]
    per = a.per_scene or int(np.ceil(a.total / len(scenes)))
    manifest = dict(scenes=[d for _, d in scenes], motions_per_scene=per, T=T, slow=SLOW, seed=a.seed, scene=a.scene, kinds=a.kinds, cameras=a.cameras)
    manifest.update(dict(chain=CHAIN, box_xy=BOX_XY, box_z=BOX_Z, floor_above_object=a.floor_above_object, max_rot_deg=float(np.degrees(MAX_ROT)), path_m=PATH_M)
                    if a.kinds == "explore" else dict(legacy_kinds=KINDS, kind_p=KIND_P))
    if not (out / "play_manifest.json").exists():
        write_json(out / "play_manifest.json", manifest)
    else:
        old = json.loads((out / "play_manifest.json").read_text())
        assert old.get("cameras", "two") == a.cameras and old.get("floor_above_object", FLOOR_ABOVE_OBJECT) == a.floor_above_object, f"{out} was collected with other settings"
    if a.scene_index is not None:
        scenes = [scenes[i] for i in a.scene_index]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for demo, n in pool.imap_unordered(collect_scene, [(f, d, per, str(out), a.seed, a.scene, a.kinds, a.cameras, a.floor_above_object) for f, d in scenes]):
            print(f"{demo}: {n} motions recorded", flush=True)


if __name__ == "__main__":
    main()
