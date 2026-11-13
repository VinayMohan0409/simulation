#!/usr/bin/env bash
# Fine-tune pipeline of the explore run (no joint WM + A2L stage). Resumable: markers/<stage>.done. Runs anywhere with /vol/dissolve and a GPU (pipeline.sbatch on Slurm).
# 1 wait: all EXPECTED clips collected and CoTracker-tracked (tracks.sbatch; any missing tracks are done here too)
# 2 window index (which windows to train on, train/val split; no encoder, no GPU)
# 3 LoRA stage 1 on the original checkpoint (read only), early stopping on the held-out hand error (check every 100 updates, patience 3, min 0.5 mm, max 3000)
# 4 encoder latents of every play window with the fine-tuned WM -> 5 latent prior -> 6 A2L + L2A on the frozen fine-tuned WM -> markers/maps.done (eval workers start)
# 7 motion check (after maps.done, off the critical path)
set -uo pipefail
source /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/env.sh
unset PLANSPACE_CHECKPOINT PLANSPACE_PRIOR
mkdir -p $N/markers $N/logs $CK; cd $REPO
stage() { local name=$1; shift
  if [ -f $N/markers/$name.done ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  if "$@" > $N/logs/$name.log 2>&1; then touch $N/markers/$name.done; log "done $name"; else log "FAILED $name (see logs/$name.log)"; exit 1; fi; }
until [ "$(n_clips)" -ge $EXPECTED ] && ! pgrep -f "planspace.play --out $N/play" > /dev/null; do log "waiting for collection ($(n_clips)/$EXPECTED)"; sleep 60; done
touch $N/markers/collect.done
stage tracks $PY -m lawm.sim.dexwm.planspace.tracks --play $N/play --kinds explore
[ "$(n_tracks)" -ge $EXPECTED ] || { log "only $(n_tracks)/$EXPECTED clips tracked"; exit 1; }
stage index $PY -m lawm.sim.dexwm.planspace.labels --play $N/play --out $N/index.npz --index-only --clips-per-scene 12
stage finetune $PY -m lawm.sim.dexwm.planspace.finetune_wm --stage 1 --labels $N/index.npz --play $N/play --out $CK --steps 3000 --accum 4 --check-every 100 --patience 3 --min-delta 0.5 --check-windows 100
export PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
stage encoder_latents $PY -m lawm.sim.dexwm.planspace.labels --play $N/play --out $CK/encoder_latents.npz --clips-per-scene 12 --batch 12
stage prior $PY -m lawm.sim.dexwm.planspace.prior_from_labels --labels $CK/encoder_latents.npz --checkpoint $CK/stage1.pt --out $CK/prior.npz
stage maps $PY -m lawm.sim.dexwm.planspace.train_g --labels $CK/encoder_latents.npz --play $N/play --out $CK/maps
log "fine-tuned WM and maps ready: $CK (stage1.pt, prior.npz, maps/g.pt, maps/l2a.pt); eval workers start now"
stage motion_check $PY -m lawm.sim.dexwm.planspace.motion_check --labels $CK/encoder_latents.npz --play $N/play --out $CK/motion_check.json
log "pipeline done"
# appended 2026-11-12 (Slurm job-submit limit): run the grasp eval inside this job on its GPU
bash $N/eval_local.sh grasp
