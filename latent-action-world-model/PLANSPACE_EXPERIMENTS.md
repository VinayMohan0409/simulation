# Planspace experiments

**Question:** does CEM search through the LAWM world model work better in latent space or in Allegro action space?

Same world model, same episodes, same cost and search budget for every arm; only the search space and the execution differ.

| Arm | Search | Execution |
|---|---|---|
| A1 | action (through the action-to-latent map, A2L) | IK of the predicted hand cloud |
| A2 | action (through A2L) | the chosen action, directly |
| L | latent (prior from the fine-tuned model's encoder latents) | IK of the predicted hand cloud |
| L2A | latent | the latent-to-action map (L2A) of the chosen latent, directly |

## Fixed for every run
- **Production scene.** Hand: FK from the joint state (proprioception). Scene: 2 simulated RGB-D cameras, back-projected, robot removed, voxelized. No privileged scene state in planning.
- **Episodes:** `work/runs/planspace/episodes/` (10 reach, 10 grasp, DexWM 50-demo eval set). Play kitchens are never from the eval set (asserted).
- **Original checkpoint:** `checkpoints/adaln_s0_k4/best.pt` is never modified; fine-tunes go to new folders.

## Current run: `work/runs/planspace_2026-11-12_explore/` (see its README)
- **Play data:** task-agnostic exploration (`planspace/play.py --kinds explore`).
  - Random wrist paths in a box around the start (±40 cm, −25/+30 cm), with no object term.
  - Rotation ≤ 20° from palm-down; random fingers with grasp-like closures.
  - Chained starts.
  - 156 clips in 13 non-eval kitchens.
- **Fine-tune (`checkpoints/adaln_s0_k4_explore_lora/`):**
  - LoRA rank 2 on encoder + decoder (stage 1 only, **no joint WM + A2L stage**).
  - Hand targets: FK. Scene targets: CoTracker tracks lifted with depth only.
  - Loss: movement-weighted Huber with per-point uncertainty.
  - Early stopping on the held-out hand error.
- **Maps:** A2L and L2A trained on the frozen fine-tuned model, from the same data.
- **Eval:** grasp first, then reach.

## Archived
`work/runs/planspace/archive_biased_object_centric_play/`:
- object-centric play runs (README there);
- reach results for all 4 arms, partial grasp results.

Checkpoint: `checkpoints/archive_biased_object_centric_play/`.
