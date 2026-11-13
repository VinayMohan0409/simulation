"""Offline check of the SAM-free object identification (object_track.py) on saved place frames (place_frames.py): start + goal identification from the start and goal
frames only, then the per-replan transport through the demo's intermediate frames (as the planner would see them; the frames are only used for this test).
Scores against the simulator's object mask (scoring only): IoU per frame (all views together) and the 3D centroid error of the identified vs true visible points.
Also the place cost's goal term (one-sided Chamfer, plan_costs.object_chamfer_cost_mm convention): start object -> goal object vs tracked object at the goal frame -> goal object.
Usage: python -m lawm.sim.dexwm.planspace.place_ident_check --frames DIR --out DIR [--demos ...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lawm.sim.dexwm.planspace import object_track as ot
from lawm.sim.dexwm.planspace.place_frames import load_frame

GOOD_IOU = 0.5


def score(views, truths, obj, lo, hi):
    """(IoU over all views or None if neither has pixels, centroid error mm or None, masks)."""
    masks = [ot.pixel_mask(v, obj, lo, hi) for v in views]
    inter = sum(int((m & t).sum()) for m, t in zip(masks, truths))
    union = sum(int((m | t).sum()) for m, t in zip(masks, truths))
    true_pts = np.concatenate([ot.view_points(v, lo, hi, t)[2] for v, t in zip(views, truths)])
    cerr = round(float(np.linalg.norm(np.mean(obj, 0) - true_pts.mean(0)) * 1000)) if len(obj) and len(true_pts) else None
    if not sum(int(t.sum()) for t in truths):  # the object is not visible in this frame: nothing to score
        return None, cerr, masks
    return (round(inter / union, 2) if union else None), cerr, masks


def chamfer_mm(a, b):
    return round(float(cKDTree(b).query(a)[0].mean() * 1000), 1) if len(a) and len(b) else None


def overlay(rgb, m, t):
    o = rgb.copy()
    o[m] = (0.5 * o[m] + [0, 127, 0]).astype(np.uint8)  # green: identified
    o[t & ~m] = (0.5 * o[t & ~m] + [127, 0, 0]).astype(np.uint8)  # red: true pixels missed
    return o


def check(f, png=None):
    z = np.load(f)
    n, nv = len(z["frame_ids"]), int(z["n_views"])
    lo, hi = z["lo"], z["hi"]
    truth = lambda j: [z[f"f{j}_v{i}_true"] for i in range(nv)]
    s_views, s_hand, s_robot, _ = load_frame(z, 0)
    g_views, g_hand, g_robot, _ = load_frame(z, n - 1)
    so, go, info = ot.identify(s_views, g_views, s_hand, s_robot, g_hand, g_robot, lo, hi)
    row = dict(start=info["start"], goal=info["goal"], frames=[int(x) for x in z["frame_ids"]])
    row["start_iou"], row["start_cerr_mm"], ms = score(s_views, truth(0), so, lo, hi)
    row["goal_iou"], row["goal_cerr_mm"], mg = score(g_views, truth(n - 1), go, lo, hi)
    tracker, track, obj, track_tiles = ot.ObjectTracker(so, s_hand, lo, hi, s_views, s_robot), [], so, []
    for j in range(1, n):
        views, hand, robot, _ = load_frame(z, j)
        obj, ti = tracker.update(views, hand, robot)
        iou, cerr, mt = score(views, truth(j), obj, lo, hi)
        track_tiles.append(np.concatenate([overlay(v[0], m, t) for v, m, t in zip(views, mt, truth(j))], axis=0))
        track.append(dict(frame=int(z["frame_ids"][j]), iou=iou, cerr_mm=cerr, source=ti["source"], hypothesis=ti.get("hypothesis"), flag=ti.get("flag"), n=len(obj)))
    row["track"] = track
    scored = [t["iou"] for t in track if t["iou"] is not None]
    row["track_good_frac"] = round(float(np.mean([x >= GOOD_IOU for x in scored])), 2) if scored else None
    row["chamfer_start_to_goal_mm"], row["chamfer_tracked_end_to_goal_mm"] = chamfer_mm(so, go), chamfer_mm(obj, go)
    if png is not None:
        import imageio
        tiles = [overlay(v[0], m, t) for v, m, t in zip(s_views, ms, truth(0))] + [overlay(v[0], m, t) for v, m, t in zip(g_views, mg, truth(n - 1))]
        imageio.imwrite(png, np.concatenate(tiles, axis=1))  # start view(s) | goal view(s): identification from the two frames
        if track_tiles:  # the per-replan transport through the demo's frames (every 5th), views stacked; green = identified, red = true object missed
            imageio.imwrite(str(png).replace(".png", "_track.png"), np.concatenate(track_tiles, axis=1)[::2, ::2])
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--demos", nargs="*", default=None)
    a = p.parse_args()
    out = Path(a.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    files = [Path(a.frames) / f"{d}.npz" for d in a.demos] if a.demos else sorted(Path(a.frames).glob("demo_*.npz"))
    rows = {}
    for f in files:
        r = rows[f.stem] = check(f, out / "png" / f"{f.stem}.png")
        print(f"{f.stem}: start IoU {r['start_iou']} ({r['start'].get('flag') or r['start']['source']}, {r['start_cerr_mm']} mm) | goal IoU {r['goal_iou']} "
              f"({r['goal'].get('flag') or r['goal']['source']}, {r['goal_cerr_mm']} mm) | track good {r['track_good_frac']} "
              f"[{' '.join(str(t['iou']) for t in r['track'])}] | chamfer start->goal {r['chamfer_start_to_goal_mm']} end->goal {r['chamfer_tracked_end_to_goal_mm']} mm", flush=True)
    (out / "ident.json").write_text(json.dumps(rows, indent=1, default=str))
    for k in ("start", "goal"):
        vis = [r[f"{k}_iou"] for r in rows.values() if r[f"{k}_iou"] is not None]
        print(f"{k}: IoU >= {GOOD_IOU} on {sum(x >= GOOD_IOU for x in vis)}/{len(vis)} demos where the object is visible (of {len(rows)})")
    tg = [r["track_good_frac"] for r in rows.values() if r["track_good_frac"] is not None]
    print(f"transport: mean fraction of good frames {np.mean(tg):.2f} over {len(tg)} demos")


if __name__ == "__main__":
    main()
