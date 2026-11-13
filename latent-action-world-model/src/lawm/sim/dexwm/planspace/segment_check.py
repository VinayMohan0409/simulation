"""How well the SAM 3 object (segment.py) matches the simulator's own object, on place episodes, before any planning uses it. Simulator segmentation is used here for scoring only.
For each episode: the task's start frame, evenly spaced stored demo frames up to the target, and the target frame; both production cameras.
Per view: mask IoU with the rendered object geoms. Per frame: point precision / recall of the SEG_OBJECT rows (truth: within 1 cm of the object's surface), and the distance of
the SAM-object centroid to the true object's visible-point centroid and to the object body position. Writes <out>/segment_check.json (+ a few overlay PNGs)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from lawm.sim import clouds
from lawm.sim.costs import SEG_OBJECT
from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planspace.common import DEXWM_CONFIG, eval_dir, write_json
from lawm.sim.dexwm.planspace.production import ProductionScene
from lawm.sim.dexwm.planspace.segment import Segmenter, identify_goal, identify_start
from lawm.sim.dexwm.runner import load_config
from lawm.sim.dexwm.scene.scene_cloud import visual_geom_ids

TRUE_SURFACE_M = 0.01


def true_masks(prod, data, obj_geoms):
    """Per view: pixels whose visible geom belongs to the object (MuJoCo segmentation render)."""
    r, out = prod.renderer, []
    r.enable_segmentation_rendering()
    try:
        for cam, _ in prod.cams:
            r.update_scene(data, camera=cam, scene_option=prod.opt)
            seg = r.render()
            m = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], obj_geoms)
            if prod.camera_set == "dexwm":  # onto DexWM's input grid, exactly as the RGB-D views (production.dexwm_grid: crop, then nearest resize)
                import cv2
                from lawm.sim.dexwm.planspace.production import DEXWM_CROP, DEXWM_OUT
                m = cv2.resize(m[DEXWM_CROP:-DEXWM_CROP].astype(np.uint8), DEXWM_OUT[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append(m)
    finally:
        r.disable_segmentation_rendering()
    return out


def check_episode(demo, task, n_mid, cfg, segmenter, png_dir=None, cameras="two"):
    sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), demo), cfg.env)
    try:
        obs = sim.reset(task, cfg.scene_seed)
        prod = ProductionScene(sim, seed=int(demo.split("_")[1]), cameras=cameras)
        prod.set_start(obs.root_pose)
        obj_geoms = visual_geom_ids(sim.model, sim.object_id)
        surf = clouds.SurfaceSampler(sim.model, obj_geoms, 4000, np.random.default_rng(0))
        start, target = benchmark.get_indices(sim.episode["waypoints"], task)
        frames = [start, target]  # identification happens at the start (held object) and the goal (matched to it); mid frames are tracked (CoTracker), not checked here
        start_desc = None
        rows = []
        for f in frames:
            sim.switch(f)
            k = sim.kinematics
            root = k.bodies[k.root]
            R, p = sim.data.xmat[root].reshape(3, 3).copy(), sim.data.xpos[root].copy()
            q = sim.data.qpos[k.qpos_adr].copy()
            hand_pts = k.points(R, p, q)
            robot_pts = prod.robot(sim.data)
            t0 = time.time()
            picked = {}

            def masks(views):
                nonlocal start_desc
                if f == start:
                    picked["out"], start_desc = identify_start(segmenter, views, robot_pts, hand_pts)
                else:
                    picked["out"] = identify_goal(segmenter, views, robot_pts, start_desc)
                return [m for m, _ in picked["out"]]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, p  # production expects the 4x4 root pose (as evaluate.production_scene builds it)
            coords, segment, views, seen = prod(sim.data, T, q, masks=masks)
            seg_s = time.time() - t0
            truth = true_masks(prod, sim.data, obj_geoms)
            per_view = []
            robot_px = [info.pop("robot_px", None) for _, info in picked["out"]]
            for (m, info), t in zip(picked["out"], truth):
                inter, union = int((m & t).sum()), int((m | t).sum())
                per_view.append(dict(iou=inter / union if union else (1.0 if not m.any() else 0.0), sam_px=int(m.sum()), true_px=int(t.sum()), info=info))
            scene = coords[segment != 0]
            is_obj = segment[segment != 0] == SEG_OBJECT
            d, _ = cKDTree(surf(sim.data)).query(scene)
            true_obj = d < TRUE_SURFACE_M
            tp = int((is_obj & true_obj).sum())
            obj_pos = sim.data.xpos[sim.object_id].copy()
            row = dict(frame=int(f), kind="start" if f == start else "target" if f == target else "mid", seconds=round(seg_s, 2), views=per_view,
                       n_obj_points=int(is_obj.sum()), n_true_obj_points=int(true_obj.sum()),
                       precision=tp / max(int(is_obj.sum()), 1), recall=tp / max(int(true_obj.sum()), 1))
            if is_obj.any():
                c = scene[is_obj].mean(0)
                row["centroid_to_body_mm"] = float(np.linalg.norm(c - obj_pos) * 1000)
                if true_obj.any():
                    row["centroid_to_true_visible_mm"] = float(np.linalg.norm(c - scene[true_obj].mean(0)) * 1000)
            rows.append(row)
            if png_dir is not None and f in (start, target):
                import imageio
                tiles = []
                for (rgb, *_), (m, info), t, rp in zip(views, picked["out"], truth, robot_px):
                    over = rgb.copy()
                    if rp is not None:  # blue: the robot mask (the scene cloud's own robot removal)
                        over[rp] = (0.5 * over[rp] + [0, 0, 127]).astype(np.uint8)
                    over[m] = (0.5 * over[m] + [0, 127, 0]).astype(np.uint8)
                    over[t & ~m] = (0.5 * over[t & ~m] + [127, 0, 0]).astype(np.uint8)
                    if info.get("positive_px"):  # yellow: the positive SAM point
                        u0, v0 = info["positive_px"]
                        over[max(v0 - 2, 0):v0 + 3, max(u0 - 2, 0):u0 + 3] = [255, 255, 0]
                    tiles.append(over)
                imageio.imwrite(png_dir / f"{demo}_{task}_f{f}.png", np.concatenate(tiles, axis=1))
        return rows
    finally:
        sim.env.close()


def main():
    """python -m lawm.sim.dexwm.planspace.segment_check --episodes episodes_place_20.json --out DIR [--task place] [--n-mid 3] [--png 4]."""
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="place")
    p.add_argument("--n-mid", type=int, default=3)
    p.add_argument("--png", type=int, default=4, help="overlay images (green SAM, red missed truth) for the first N episodes")
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--cameras", choices=("two", "dexwm"), default="two", help="two (planspace) or dexwm (robot0_robotview_2 on DexWM's 224 x 392 grid)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(exist_ok=True)
    cfg = load_config(DEXWM_CONFIG)
    demos = a.demos or json.loads(Path(a.episodes).read_text())["selected"]
    res_file = out / "segment_check.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    seg = Segmenter(log=log)
    try:
        for i, demo in enumerate(demos):
            if demo in res:
                continue
            rows = check_episode(demo, a.task, a.n_mid, cfg, seg, out / "png" if i < a.png else None, a.cameras)
            res[demo] = rows
            write_json(res_file, res)
            log(f"{demo}: IoU " + " ".join(f"{np.mean([v['iou'] for v in r['views']]):.2f}" for r in rows) +
                " | prec " + " ".join(f"{r['precision']:.2f}" for r in rows) + " | rec " + " ".join(f"{r['recall']:.2f}" for r in rows))
    finally:
        seg.close()
    write_json(out / "segment_check_summary.json", summarize(res))


def summarize(res):
    """Mean / median of every score, by frame kind and overall."""
    rows = [r for v in res.values() for r in v]
    out = {}
    for kind in ("start", "mid", "target", "all"):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not rs:
            continue
        ious = [v["iou"] for r in rs for v in r["views"] if v["true_px"] > 0]
        stat = lambda xs: dict(mean=float(np.mean(xs)), median=float(np.median(xs)), n=len(xs)) if xs else None
        out[kind] = dict(view_iou=stat(ious), precision=stat([r["precision"] for r in rs]), recall=stat([r["recall"] for r in rs]),
                         empty_frames=sum(r["n_obj_points"] == 0 for r in rs),
                         centroid_to_true_visible_mm=stat([r["centroid_to_true_visible_mm"] for r in rs if "centroid_to_true_visible_mm" in r]),
                         centroid_to_body_mm=stat([r["centroid_to_body_mm"] for r in rs if "centroid_to_body_mm" in r]), seconds=stat([r["seconds"] for r in rs]))
    return out


if __name__ == "__main__":
    main()
