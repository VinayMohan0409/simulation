"""PLY files showing production.py's camera-pose offset (MuJoCo's LEFT stereo eye used as the camera position) against the FK hand / robot, from saved place frames.
Per demo and frame: <demo>_<frame>_legacy.ply (pose as the play data, fine-tunes and all planspace runs used it) and <demo>_<frame>_fixed.ply (mono centre), each with
  the camera points in their RGB colour (view 1 of two cameras tinted blue, so the two views' mutual misalignment is visible), the FK hand points red, the FK robot surface green;
and <demo>_<frame>_removed_{legacy,fixed}.ply: only the camera points that survive production's robot removal (1.5 cm from the FK robot) -> robot ghosts in legacy.
Usage: python -m lawm.sim.dexwm.planspace.pose_check_ply --frames DIR --out DIR --demos demo_133 [--frame 0]"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lawm.sim.dexwm.planspace.place_frames import IPD
from lawm.sim.dexwm.planspace.production import ROBOT_MARGIN, back_project


def write_ply(path, pts, cols):
    pts, cols = np.asarray(pts, np.float32), np.clip(np.asarray(cols), 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())


def camera_points(z, j, centred, lo, hi):
    pts, cols = [], []
    for i in range(int(z["n_views"])):
        rgb, depth, K, pose = z[f"f{j}_v{i}_rgb"], z[f"f{j}_v{i}_depth"], z[f"f{j}_v{i}_K"], z[f"f{j}_v{i}_pose"].copy()
        if centred:
            pose[:3, 3] += pose[:3, 0] * IPD / 2
        v, u = np.nonzero((depth > 0.1) & (depth < 3.0))
        p = back_project(depth, K, pose)
        c = rgb[v, u].astype(float)
        if i == 1:
            c = 0.6 * c + 0.4 * np.array([0, 0, 255])
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        pts.append(p[inside]), cols.append(c[inside])
    return np.concatenate(pts), np.concatenate(cols)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--demos", nargs="+", required=True)
    a.add_argument("--frame", type=int, default=0)
    a = a.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for demo in a.demos:
        z = np.load(Path(a.frames) / f"{demo}.npz")
        j, lo, hi = a.frame, z["lo"], z["hi"]
        hand, robot = z[f"f{j}_hand"], z[f"f{j}_robot"]
        robot_in = robot[np.all((robot >= lo) & (robot <= hi), axis=1)]
        tree = cKDTree(robot)
        for name, centred in (("legacy", False), ("fixed", True)):
            p, c = camera_points(z, j, centred, lo, hi)
            d = tree.query(p)[0]
            near = d < 0.08
            print(f"{demo} f{j} {name}: median camera->FK-robot distance of points near the robot {np.median(d[near]) * 1000:.1f} mm; "
                  f"points left after production's {ROBOT_MARGIN * 100:.1f} cm robot removal within 5 cm of the robot: {int(((d >= ROBOT_MARGIN) & (d < 0.05)).sum())}")
            write_ply(out / f"{demo}_f{j}_{name}.ply", np.r_[p, hand, robot_in], np.r_[c, np.tile([255, 0, 0], (len(hand), 1)), np.tile([0, 255, 0], (len(robot_in), 1))])
            keep = d >= ROBOT_MARGIN
            write_ply(out / f"{demo}_f{j}_removed_{name}.ply", np.r_[p[keep], hand], np.r_[c[keep], np.tile([255, 0, 0], (len(hand), 1))])


if __name__ == "__main__":
    main()
