# Exploratory-data run (2026-11-12)

Replaces the archived object-biased run (`../planspace/archive_biased_object_centric_play/`). Nothing here is shared with it except the read-only episode lists in
`../planspace/episodes/` (same 10 grasp and 10 reach episodes) and the never-modified original checkpoint `checkpoints/adaln_s0_k4/best.pt`.

## Data: task-agnostic exploratory play (`play/`, 12 clips x 13 non-eval kitchens = 156)
- Wrist: random walk towards uniform points of a box around the kitchen's hand start (x, y +/-40 cm; z -25..+30 cm, never below the counter); nothing aims at the object.
- Rotation: at most 20 deg from the start (palm-down) orientation, absolute (no accumulation).
- Fingers: random; half of the motions have a grasp-like close-and-reopen at a random time.
- About 2 cm per 0.2 s model step; motions chained (each starts where the previous ended), kitchen reset every 5 motions.
- Check video: `check_play/demo_1370_chain_explore.mp4`.
- DexWM's own exploratory data (HF facebook/dexwm) was checked and not used: no sim states / kitchen files (cannot be replayed as scenes),
  94% forward / 92% upward motion, 34-97 deg rotations.

## Model: `checkpoints/adaln_s0_k4_explore_lora/`
- LoRA (rank 2) fine-tune of the WM only, from the original `checkpoints/adaln_s0_k4/best.pt` (read only). NO joint WM + A2L stage.
- Supervision: hand points = FK hand from proprioception (joint angles + hand mesh); scene points = CoTracker tracks lifted with depth only;
  untracked scene points are input only. No simulator ground-truth correspondences. Loss: movement-weighted Huber with per-point uncertainty.
- Early stopping: held-out hand error (validation kitchens) checked every 100 updates; best kept; stop after 3 checks without a 0.5 mm gain; max 3000.
- Then: encoder latents of every play window with the fine-tuned WM (`encoder_latents.npz`) -> latent prior (`prior.npz`) ->
  A2L (`maps/g.pt`) and L2A (`maps/l2a.pt`), both trained on the frozen fine-tuned WM from the same data.
- Window index (`index.npz`): which play windows train / validate (no encoder latents; that is all stage 1 reads).

## Run it (from the Slurm submit host; everything waits for its inputs)
    cd /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
    sbatch -p a100,a40,a30,a16,t4 tracks.sbatch
    P=$(sbatch --parsable -p a100,a40 pipeline.sbatch)
    for a in A1 A2 L L2A; do sbatch --dependency=afterok:$P --job-name=ex_$a -p a100,a40,a30,a16,t4 eval.sbatch $a grasp; done
- Collection (`collect.sh`) runs on gpu19. Reach later: same eval line with `reach`.
- Each eval sbatch = 10 workers; per-episode locks give each worker its own episode. If anything dies, resubmit the same line:
  finished stages / episodes are skipped, nothing is overwritten.
- Progress: `ls markers/`, `tail logs/*.out`. Results: grasp/<arm>/<demo>_grasp.{json,mp4}, compare/, EVAL_SUMMARY.md.
