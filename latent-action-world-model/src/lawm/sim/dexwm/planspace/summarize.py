"""Summary of a planspace evaluation run dir: EVAL_SUMMARY.md (per task: success, final / min error, replans, wall time per arm; per-episode table with links to the videos)
and compare/<task>/<demo>.mp4 (A1 | A2 | L | L2A | groundtruth stacked, labelled, for side-by-side review). CPU only; safe to run while jobs are running (derived files only, written atomically)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ARMS = ("A1", "A2", "L", "L2A")
TASKS = ("reach", "grasp", "place")  # a task folder that does not exist in the run is skipped
DESCRIPTION = dict(A1="action search, IK of the predicted hand", A2="action search, action executed directly",
                   L="latent search, IK of the predicted hand", L2A="latent search, L2A action executed directly")


def episodes(run, task):
    """The task's selected demos: the 20-episode selection when it is finished (its first 10 are the original ones), else the original 10."""
    base = Path("/vol/dissolve/vinay/work/runs/planspace/episodes")  # shared by every planspace run
    path = base / f"episodes_{task}_20.json"
    if not (path.exists() and json.loads(path.read_text()).get("done")):
        path = base / f"episodes_{task}.json"
    if not path.exists():
        return []
    e = json.loads(path.read_text())
    return e.get("selected", []) if e.get("done", True) else []


def load(run, task, arm, demo):
    f = run / task / arm / f"{demo}_{task}.json"
    return json.loads(f.read_text()) if f.exists() else None


def label(frame, text):
    """Frame with a black text bar on top."""
    from PIL import Image, ImageDraw
    bar = np.zeros((22, frame.shape[1], 3), np.uint8)
    img = Image.fromarray(bar)
    ImageDraw.Draw(img).text((6, 4), text, fill=(255, 255, 255))
    return np.concatenate([np.asarray(img), frame], axis=0)


def compare_video(run, task, demo):
    """compare/<task>/<demo>.mp4 from the arms' videos that exist (missing arms: grey); rebuilt only when a source is newer."""
    import imageio
    sources = [(arm, run / task / arm / f"{demo}_{task}.mp4") for arm in ARMS] + [("groundtruth (demo)", run / task / "groundtruth" / f"{demo}_{task}.mp4")]
    have = [p for _, p in sources if p.exists()]
    if not have:
        return None
    out = run / "compare" / task / f"{demo}.mp4"
    if out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in have):
        return out
    clips = {}
    for name, p in sources:
        if p.exists():
            clips[name] = [np.asarray(f) for f in imageio.mimread(p, memtest=False)]
    shape = next(iter(clips.values()))[0].shape
    n = max(len(c) for c in clips.values())
    frames = []
    for t in range(n):
        rows = []
        for name, p in sources:
            arm = name.split()[0]
            r = load(run, task, arm, demo) if arm in ARMS else None
            tag = f"{name}" + (f"  [{'SUCCESS' if r['success'] else 'fail'}, final {r['final_error_mm']:.0f} mm]" if r else "")
            c = clips.get(name)
            f = c[min(t, len(c) - 1)] if c else np.full(shape, 60, np.uint8)
            rows.append(label(f, f"{tag}  {'' if c else '(not run yet)'}"))
        frames.append(np.concatenate(rows, axis=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{demo}.{os.getpid()}.tmp.mp4"
    imageio.mimsave(tmp, frames, fps=10)
    os.replace(tmp, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--title", default="Planspace evaluation: 4 arms, production scene, fine-tuned checkpoint")
    a = p.parse_args()
    run = Path(a.run)
    md = [f"# {a.title}", "",
          "Arms: " + "; ".join(f"**{k}** = {v}" for k, v in DESCRIPTION.items()) + ".",
          "Errors in mm: reach = mean distance of the 5 hand bodies to the target; grasp = wrist to object; place = object to its target. Min = lowest error seen at any replan.",
          "Videos: `<task>/<arm>/<demo>_<task>.mp4` (both cameras, every control step); `compare/<task>/<demo>.mp4` stacks all arms + the demo.", ""]
    for task in TASKS:
        if not (Path(run) / task).is_dir():
            continue
        demos = episodes(run, task)
        md += [f"## {task} ({len(demos)} episodes)", ""]
        if not demos:
            md += ["Episode selection not finished yet.", ""]
            continue
        md += ["| Arm | Done | Success | Mean final (mm) | Mean min (mm) | Mean replans | Mean wall (min) |", "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            rs = [r for r in (load(run, task, arm, d) for d in demos) if r]
            if not rs:
                md.append(f"| {arm} | 0/{len(demos)} | | | | | |")
                continue
            mins = [min(x["error_mm"] for x in r["records"]) for r in rs]
            md.append(f"| {arm} | {len(rs)}/{len(demos)} | {sum(r['success'] for r in rs)}/{len(rs)} | {np.mean([r['final_error_mm'] for r in rs]):.0f} | "
                      f"{np.mean(mins):.0f} | {np.mean([r['n_replans'] for r in rs]):.1f} | {np.mean([r['wall_seconds'] for r in rs]) / 60:.1f} |")
        md += ["", "| Episode | " + " | ".join(ARMS) + " | Compare |", "|---|" + "---|" * (len(ARMS) + 1)]
        for d in demos:
            cells = []
            for arm in ARMS:
                r = load(run, task, arm, d)
                cells.append(f"[{'✓' if r['success'] else '✗'} {r['final_error_mm']:.0f}]({task}/{arm}/{d}_{task}.mp4)" if r else "—")
            cmp_path = None if a.no_video else compare_video(run, task, d)
            md.append(f"| {d} | " + " | ".join(cells) + f" | {f'[video](compare/{task}/{d}.mp4)' if cmp_path else '—'} |")
        md.append("")
    tmp = run / f".EVAL_SUMMARY.{os.getpid()}.tmp.md"
    tmp.write_text("\n".join(md))
    os.replace(tmp, run / "EVAL_SUMMARY.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
