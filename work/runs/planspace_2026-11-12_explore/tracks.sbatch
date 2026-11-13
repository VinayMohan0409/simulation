#!/bin/bash
#SBATCH --job-name=ex_tracks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# CoTracker scene tracks of every play clip, alongside the collection: repeats until all EXPECTED clips are collected and tracked (finished clips are skipped).
# Usage: sbatch -p a100,a40,a30,a16,t4 tracks.sbatch
source /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/env.sh
cd $REPO
until [ "$(n_clips)" -ge $EXPECTED ] && [ "$(n_tracks)" -ge $EXPECTED ]; do
  log "clips $(n_clips)/$EXPECTED, tracked $(n_tracks)"
  $PY -m lawm.sim.dexwm.planspace.tracks --play $N/play --kinds explore
  [ "$(n_tracks)" -ge "$(n_clips)" ] && sleep 60
done
log "all $EXPECTED clips tracked"
