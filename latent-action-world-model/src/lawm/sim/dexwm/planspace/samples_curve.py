"""Experiment S: task success vs CEM samples per replan, latent search (L) vs action search (A1), both with IK execution (only the search space differs).
Reads <run>/<task>/{L,A1}/ (the main eval: 64 x 4 = 256 samples) and <run>/samples/<task>/{L,A1}_s<N>/ over the 20-episode selection; checks every folder's arm.json budget;
writes <run>/samples/SAMPLES.md (tables) and <run>/samples/samples_curve.png (success rate with 95% Wilson intervals; mean final and best error)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("L", "A1")
LABEL = dict(L="latent search + IK (L)", A1="action search + IK (A1)")


def wilson(k, n, z=1.96):
    """95% Wilson interval of k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def folders(run, task, arm):
    """{samples per replan: folder} of one arm: the main eval and every samples/<task>/<arm>_s<N> folder."""
    out = {}
    for f in [run / task / arm] + sorted((run / "samples" / task).glob(f"{arm}_s*")):
        meta = f / "arm.json"
        if not meta.exists():
            continue
        cfg = json.loads(meta.read_text())
        assert cfg["arm"] == arm and cfg.get("task", "reach") == task, f"{meta}: wrong arm/task"
        n = int(cfg["cem"]["pop"]) * int(cfg["cem"]["iters"])
        if f.name != arm:
            assert f.name == f"{arm}_s{n}", f"{meta}: folder name and budget ({n}) disagree"
        assert n not in out, f"two folders with {n} samples for {arm}"
        out[n] = f
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore")
    p.add_argument("--task", default="reach")
    p.add_argument("--episodes", default="/vol/dissolve/vinay/work/runs/planspace/episodes")
    p.add_argument("--suffix", default="_20", help="episode file episodes_<task><suffix>.json (default _20: the 20-episode selection the runs used; '' = the old 10)")
    a = p.parse_args()
    run = Path(a.run)
    demos = json.loads((Path(a.episodes) / f"episodes_{a.task}{a.suffix}.json").read_text())["selected"]
    rows, curves = [], {}
    for arm in ARMS:
        for n, f in sorted(folders(run, a.task, arm).items()):
            rs = [json.loads((f / f"{d}_{a.task}.json").read_text()) for d in demos if (f / f"{d}_{a.task}.json").exists()]
            k = sum(r["success"] for r in rs)
            lo, hi = wilson(k, len(rs))
            fin = np.mean([r["final_error_mm"] for r in rs]) if rs else float("nan")
            best = np.mean([min(x["error_mm"] for x in r["records"]) for r in rs]) if rs else float("nan")
            curves.setdefault(arm, []).append((n, len(rs), k, lo, hi, fin, best))
            rows.append(f"| {arm} | {n} | {len(rs)}/{len(demos)} | {k}/{len(rs)} | {lo:.2f}-{hi:.2f} | {fin:.0f} | {best:.0f} | `{f.relative_to(run)}` |")
    out = run / "samples"
    out.mkdir(exist_ok=True)
    md = [f"# Experiment S: success vs samples per replan ({a.task}, {len(demos)} episodes)", "",
          "Same checkpoint, prior, A2L, episodes, seeds, cost and loop; CEM iterations fixed at 4, the population varies. Success = DexWM's rule.", "",
          "| Arm | Samples/replan | Done | Success | 95% Wilson | Mean final (mm) | Mean best (mm) | Folder |", "|---|---|---|---|---|---|---|---|"] + rows
    (out / "SAMPLES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arm, pts in curves.items():
        pts = [q for q in pts if q[1] > 0]
        if not pts:
            continue
        n, cnt, k, lo, hi, fin, best = map(np.array, zip(*pts))
        rate = k / cnt
        line = ax[0].errorbar(n, rate, yerr=[rate - lo, hi - rate], marker="o", capsize=4, label=LABEL[arm])
        ax[1].plot(n, fin, marker="o", color=line[0].get_color(), label=f"{arm} final")
        ax[1].plot(n, best, marker="s", ls="--", color=line[0].get_color(), label=f"{arm} best")
    for x in ax:
        x.set_xscale("log", base=2)
        x.set_xlabel("CEM samples per replan")
        x.grid(alpha=0.3)
    ax[0].set_ylabel("success rate")
    ax[0].set_ylim(-0.05, 1.05)
    ax[1].set_ylabel("mean error (mm)")
    ax[0].legend()
    ax[1].legend(fontsize=8)
    ax[0].set_title(f"{a.task}: success vs samples (95% Wilson)")
    ax[1].set_title(f"{a.task}: final and best error")
    fig.tight_layout()
    fig.savefig(out / "samples_curve.png", dpi=130)
    print(f"-> {out / 'samples_curve.png'}")


if __name__ == "__main__":
    main()
