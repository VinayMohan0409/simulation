# Latent ablations backup (2026-11-13)

4 arms (A1, A2, L, L2A) x 20 episodes x reach / grasp / place.

- reach, grasp: `work/runs/planspace_2026-11-12_explore/{reach,grasp}/<arm>/` (+ `EVAL_SUMMARY.md`, `compare/`)
  - `*_old_*` / `*_intermediate_*` subdirs are superseded runs, not results.
- place: `work/runs/place_sam3/place_geom/<arm>/` — A2 has 19/20 (demo_731 missing at backup time).
- episode lists: `work/runs/planspace/episodes/`
- code (untracked in latent-action-world-model at backup time): `latent-action-world-model/src/lawm/sim/dexwm/{planspace,bench}`;
  uncommitted edits to tracked files: `latent-action-world-model/tracked_changes.diff`.
- Not included (size): checkpoint `checkpoints/adaln_s0_k4_explore_lora/` — in the full tarball at
  `/vol/dissolve/vinay/backups/latent_ablations_2026-11-13.tar.gz`.

Paths mirror `/vol/dissolve/vinay/`.
