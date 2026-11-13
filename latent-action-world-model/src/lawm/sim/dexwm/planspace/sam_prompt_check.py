"""What does SAM 3 detect on DexWM's camera? For each text prompt, on the saved start / goal frames (place_frames.py): how many masks it returns and whether
ANY of them covers the true object (best IoU >= 0.5; the simulator mask is used for scoring only). Overlays of every mask per prompt for a few demos.
Usage: python -m lawm.sim.dexwm.planspace.sam_prompt_check --frames DIR --out DIR [--full] [--threshold 0.3]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lawm.sim.dexwm.planspace.segment import Segmenter

PROMPTS = ("object", "objects", "item", "thing", "small object", "household object", "food", "container", "robot hand")
COLOURS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)


def iou(m, t):
    u = (m | t).sum()
    return float((m & t).sum() / u) if u else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--png", type=int, default=6, help="overlays for the first N demos")
    p.add_argument("--prompts", nargs="*", default=None, help="text prompts (default: PROMPTS)")
    p.add_argument("--tag", default="", help="suffix of the output files")
    a = p.parse_args()
    out = Path(a.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    sfx = "_full" if a.full else ""
    prompts = tuple(a.prompts) if a.prompts else PROMPTS
    seg = Segmenter(prompts=prompts, threshold=a.threshold)
    stats = {pr: {k: dict(n_masks=[], hit=[], best_iou=[]) for k in ("start", "goal")} for pr in prompts}
    files = sorted(Path(a.frames).glob("demo_*.npz"))
    for di, f in enumerate(files):
        z = np.load(f)
        tiles = []
        for kind in ("start", "goal"):
            rgb, t = z[f"{kind}_rgb{sfx}"], z[f"{kind}_true_mask{sfx}"]
            res = seg.segment([rgb])[0]
            row = []
            for pr in prompts:
                ms = res[pr]["masks"]
                b = max((iou(m, t) for m in ms), default=0.0)
                if t.any():  # only frames where the object is visible count
                    stats[pr][kind]["n_masks"].append(len(ms))
                    stats[pr][kind]["best_iou"].append(b)
                    stats[pr][kind]["hit"].append(b >= 0.5)
                if di < a.png:
                    o = rgb.copy()
                    for j, m in enumerate(ms):
                        o[m] = (0.45 * o[m] + 0.55 * COLOURS[j % len(COLOURS)]).astype(np.uint8)
                    o[t & ~np.any(ms, axis=0) if len(ms) else t] = [255, 255, 255]  # white: true object pixels no mask covers
                    row.append(o)
            if row:
                tiles.append(np.concatenate(row, axis=1))
        if tiles:
            import imageio
            imageio.imwrite(out / "png" / f"{f.stem}{sfx}{a.tag}.png", np.concatenate(tiles, axis=0))
    seg.close()
    lines = [f"SAM 3 text prompts on {len(files)} place demos, robot0_robotview_2 {'600x960' if a.full else '224x392'}, threshold {a.threshold}",
             "(frames where the object is visible; hit = some mask has IoU >= 0.5 with the true object)", "",
             f"{'prompt':<30} | {'start: hit':>11} {'masks/img':>9} {'best IoU':>8} | {'goal: hit':>10} {'masks/img':>9} {'best IoU':>8}"]
    for pr in prompts:
        s, g = stats[pr]["start"], stats[pr]["goal"]
        lines.append(f"{pr:<30} | {sum(s['hit']):>4}/{len(s['hit']):<6} {np.mean(s['n_masks']):>9.1f} {np.mean(s['best_iou']):>8.2f} | "
                     f"{sum(g['hit']):>4}/{len(g['hit']):<5} {np.mean(g['n_masks']):>9.1f} {np.mean(g['best_iou']):>8.2f}")
    print("\n".join(lines))
    (out / f"prompts{sfx}{a.tag}.txt").write_text("\n".join(lines) + "\n")
    (out / f"prompts{sfx}{a.tag}.json").write_text(json.dumps(stats, default=float))


if __name__ == "__main__":
    main()
