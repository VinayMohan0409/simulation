"""summary.md of a planspace run: per arm success, final error, replans, planning time and decodes; the per-episode paired table; g's reliability numbers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_arm(d):
    return {json.loads(p.read_text())["demo"]: json.loads(p.read_text()) for p in sorted(Path(d).glob("demo_*_reach.json"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run)
    lines = ["# Planspace run: reach, action-space vs latent-space CEM through LAWM", ""]
    eps = json.loads((run / "episodes.json").read_text())
    lines += [f"Episodes ({len(eps['selected'])}): {', '.join(eps['selected'])}", ""]
    mc = run / "motion_check.json"
    if mc.exists():
        m = json.loads(mc.read_text())
        lines += ["## Does the WM predict the play motion (encoder latents, held-out windows, hand points)", "",
                  f"true hand mm per step {m['true_hand_mm_per_step']}, predicted {m['predicted_hand_mm_per_step']}; "
                  f"step-5 size ratio {m['size_ratio_step5']}, direction cosine median {m['direction_cosine_step5_median']}", ""]
    rep_path = run / "g" / "g_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        lines += ["## Action-to-latent map A2L, per step (held-out play kitchens, decoded error in mm; g = A2L)", "",
                  "| latents | hand | object | hand step 1 | hand step 5 |", "|---|---|---|---|---|"]
        for k, v in rep["final"].items():
            lines.append(f"| {k} | {v['hand_mm']} | {v['object_mm']} | {v['hand_step1_mm']} | {v['hand_step5_mm']} |")
        lines += ["", f"A2L / encoder hand error = {rep['g_over_encoder_hand']} (gate <= {rep['gate_ratio']}: {'PASS' if rep['passed'] else 'FAIL'}, reported only); "
                  f"A2L / zero motion = {rep['g_over_zero_hand']}; chosen = {rep['chosen']}", ""]
    lr = run / "g" / "l2a_report.json"
    if lr.exists():
        r = json.loads(lr.read_text())["heldout_step_errors_rms"]
        lines += ["## Latent-to-action map L2A, per step (held-out play kitchens, RMS error of one step's action)", "", "| predictor | wrist mm | wrist deg | fingers rad |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['wrist_mm']} | {v['wrist_deg']} | {v['fingers_rad']} |" for k, v in r.items()] + [""]
    arms = {k: load_arm(run / "eval" / k) for k in ("A1", "A2", "L") if (run / "eval" / k).exists()}
    lines += ["## Closed loop (reach)", "", "| arm | search | execution | episodes | success | median final error mm | mean replans | mean plan s/replan | decodes/replan |",
              "|---|---|---|---|---|---|---|---|---|"]
    desc = dict(A1=("action", "hand_fit of prediction"), A2=("action", "direct action"), L=("latent", "hand_fit of prediction"))
    for k, res in arms.items():
        if not res:
            continue
        recs = [r for e in res.values() for r in e["records"]]
        lines.append(f"| {k} | {desc[k][0]} | {desc[k][1]} | {len(res)} | {sum(e['success'] for e in res.values())}/{len(res)} | "
                     f"{np.median([e['final_error_mm'] for e in res.values()]):.1f} | {np.mean([e['n_replans'] for e in res.values()]):.1f} | "
                     f"{np.mean([r['plan_seconds'] for r in recs]):.2f} | {np.mean([r['wm_calls'] for r in recs]):.0f} |")
    lines += ["", "## Per episode (success / final error mm / replans)", "", "| demo | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for d in eps["selected"]:
        cells = []
        for k in arms:
            e = arms[k].get(d)
            cells.append("-" if e is None else f"{'OK' if e['success'] else 'fail'} / {e['final_error_mm']:.0f} / {e['n_replans']}")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "hold share (replans where holding still beat every sampled plan):"]
    for k, res in arms.items():
        recs = [r for e in res.values() for r in e["records"]]
        if recs:
            lines.append(f"- {k}: {np.mean([r['executed'] == 'hold' for r in recs]):.2f}")
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
