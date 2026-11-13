"""Scene-point tracks of the production-scene play clips with CoTracker3 (as PointWorld gets its targets): per window, the frame-0 camera scene points (already cropped, robot-removed,
voxelised) are projected into each camera, kept where that camera sees them (depth agrees), tracked over the window's frames, and lifted back to 3D with each frame's depth.
A point counts for a window only if one camera keeps it visible, with valid depth, at every model step. Writes <clip>.tracks.npz next to each clip."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from lawm.sim.dexwm.planspace.common import CONTROL_PER_MODEL_STEP, N_STEPS, wait_for_gpu
from lawm.sim.dexwm.planspace.labels import window_starts

DEPTH_AGREE = 0.01  # metres: a projected point counts as seen by a camera when its depth there agrees this well
QUERY_CHUNK = 1000
TRACK_EVERY = 2  # every 2nd label window gets tracks (the fine-tune trains on those)
MAX_QUERIES = 1024  # scene rows tracked per window (a random subset; the rest are input only)
EDGE = 0.02  # metres of depth range in a pixel's 3x3 neighbourhood above which it sits on a depth edge (lifting there slips between surfaces)
JUMP = 0.08  # metres: a lifted track that moves more than this in one model step is dropped (a slip onto another surface)


def project(points, K, pose):
    """Pixel (u, v) and camera depth z of world points, for a camera with intrinsics K and optical pose (4, 4)."""
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    z = cam[:, 2]
    return cam[:, 0] / z * K[0, 0] + K[0, 2], cam[:, 1] / z * K[1, 1] + K[1, 2], z


def lift(u, v, depth, K, pose):
    """World points of pixels (u, v) (rounded) with their depth; NaN where the pixel is outside or has no depth."""
    H, W = depth.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    z = np.full(len(u), np.nan)
    z[ok] = depth[vi[ok], ui[ok]]
    z[z <= 0] = np.nan
    if ok.any():  # drop pixels on a depth edge
        pad = np.pad(depth, 1, mode="edge")
        nb = np.stack([pad[vi[ok] + dv, ui[ok] + du] for dv in range(3) for du in range(3)])
        edge = np.zeros(len(u), bool)
        edge[ok] = (nb.max(0) - nb.min(0)) > EDGE
        z[edge] = np.nan
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return cam @ pose[:3, :3].T + pose[:3, 3]


def clip_tracks(tracker, rec, cams, device):
    """For every window start: (scene row indices (M,), world positions at model steps 1..5 (5, M, 3) float32) of the tracked scene rows."""
    seg = rec["segment"]
    scene_rows = np.flatnonzero(seg != 0)
    rgb, depth = cams["rgb"], cams["depth_mm"].astype(np.float32) / 1000.0  # (T, C, H, W, 3), (T, C, H, W)
    K, pose = cams["K"], cams["cam_pose"]
    span = N_STEPS * CONTROL_PER_MODEL_STEP
    out = {}
    all_scene = scene_rows
    for s in window_starts(len(rec["cmd_pos"]))[::TRACK_EVERY]:
        scene_rows = np.sort(np.random.default_rng(s).choice(all_scene, min(MAX_QUERIES, len(all_scene)), replace=False))
        p0 = rec["coords"][s][scene_rows]
        best = np.full((N_STEPS, len(scene_rows), 3), np.nan)
        for c in range(rgb.shape[1]):
            u, v, z = project(p0, K[s, c], pose[s, c])
            H, W = depth.shape[2:]
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inside = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            seen = np.zeros(len(p0), bool)
            seen[inside] = np.abs(depth[s, c][vi[inside], ui[inside]] - z[inside]) < DEPTH_AGREE
            todo = seen & np.isnan(best[0, :, 0])  # rows no earlier camera tracked
            if not todo.any():
                continue
            video = torch.as_tensor(rgb[s:s + span + 1, c], device=device).permute(0, 3, 1, 2)[None].float()
            q = torch.as_tensor(np.stack([np.zeros(todo.sum()), u[todo], v[todo]], 1), dtype=torch.float32, device=device)[None]
            trs, viss = [], []
            for j in range(0, q.shape[1], QUERY_CHUNK):  # the correlation volume grows with the queries: chunks keep it within a shared 11 GB GPU
                with torch.no_grad():
                    tr, vis = tracker(video, queries=q[:, j:j + QUERY_CHUNK])
                trs.append(tr[0].cpu().numpy())
                viss.append(vis[0].cpu().numpy())
            tr, vis = np.concatenate(trs, 1), np.concatenate(viss, 1)  # (T, M, 2), (T, M)
            rows = np.flatnonzero(todo)
            for k in range(1, N_STEPS + 1):
                f = s + CONTROL_PER_MODEL_STEP * k
                pk = lift(tr[CONTROL_PER_MODEL_STEP * k, :, 0], tr[CONTROL_PER_MODEL_STEP * k, :, 1], depth[f, c], K[f, c], pose[f, c])
                pk[~vis[CONTROL_PER_MODEL_STEP * k]] = np.nan
                best[k - 1, rows] = pk
        prev = np.concatenate([p0[None], best[:-1]], 0)
        ok = np.all(np.isfinite(best).all(-1) & (np.linalg.norm(best - prev, axis=-1) < JUMP), axis=0)
        out[s] = (scene_rows[ok].astype(np.int32), best[:, ok].astype(np.float32))
    return out


def save(path, tracks):
    """Flat arrays: starts (W,), counts (W,), rows (sum,), pos (5, sum, 3)."""
    starts = np.array(sorted(tracks), np.int32)
    np.savez(path, starts=starts, counts=np.array([len(tracks[s][0]) for s in starts], np.int32),
             rows=np.concatenate([tracks[s][0] for s in starts]), pos=np.concatenate([tracks[s][1] for s in starts], axis=1))


def load(path):
    """{start: (rows, pos (5, M, 3))} of a tracks file."""
    d = np.load(path)
    ends = np.cumsum(d["counts"])
    return {int(s): (d["rows"][e - n:e], d["pos"][:, e - n:e]) for s, n, e in zip(d["starts"], d["counts"], ends)}


def main():
    """python -m lawm.sim.dexwm.planspace.tracks --play DIR [--limit N]: CoTracker3 scene tracks of every production clip (finished clips are skipped)."""
    p = argparse.ArgumentParser()
    p.add_argument("--play", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--kinds", default="grasp,push", help="clip kinds to track (the scene only moves in these; the kind is our own command, not privileged)")
    a = p.parse_args()
    kinds = set(a.kinds.split(","))
    wait_for_gpu(3000)
    device = torch.device("cuda")
    tracker = torch.hub.load("/vol/dissolve/vinay/work/third_party/co-tracker", "cotracker3_offline", source="local").to(device).eval()
    clips = sorted(f for f in Path(a.play).glob("demo_*.npz") if not f.name.endswith((".rgbd.npz", ".tmp.npz", ".tracks.npz")))[:a.limit]
    if os.environ.get("TRACKS_ORDER", "reverse") == "reverse":  # a second tracker started later works from the other end (finished clips are skipped either way)
        clips = clips[::-1]
    t0 = time.time()
    for i, f in enumerate(clips):
        dst = f.with_name(f.name.replace(".npz", ".tracks.npz"))
        if dst.exists():
            continue
        rec = dict(np.load(f))
        if str(rec["kind"]) not in kinds:
            continue
        cams = dict(np.load(f.with_name(f.name.replace(".npz", ".rgbd.npz"))))
        tr = clip_tracks(tracker, rec, cams, device)
        save(dst.with_name(dst.name.replace(".tracks.npz", ".tracks.tmp.npz")), tr)
        dst.with_name(dst.name.replace(".tracks.npz", ".tracks.tmp.npz")).replace(dst)
        n = np.mean([len(v[0]) for v in tr.values()])
        print(f"{i + 1}/{len(clips)} {f.name}: {n:.0f} tracked scene rows per window ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
