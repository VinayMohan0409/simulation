#!/usr/bin/env bash
# Experiment S (NOT started automatically): reach success vs CEM samples per replan, L vs A1 (both IK execution).
# Jobs: the main reach eval at 256 samples (all 4 arms; L and A1 are the 256 point), then L and A1 at 128 (pop 32) and 64 (pop 16).
# Several workers on this machine's GPU (11 GB each, at most MAX_WORKERS); per-episode locks share the work, also with other copies of this script elsewhere.
# Finished episodes are skipped; nothing is overwritten. Then samples_curve.py -> samples/SAMPLES.md + samples/samples_curve.png.
# Usage: bash samples.sh        (or sbatch -p a100 samples.sbatch)
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
source $N/env.sh
JOBS=(L: A1: A2: L2A: L:32 A1:32 L:16 A1:16)
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0} | head -1)
K=$(( FREE / 11000 )); [ $K -gt ${MAX_WORKERS:-4} ] && K=${MAX_WORKERS:-4}; [ $K -lt 1 ] && K=1
log "samples experiment: $FREE MB free -> $K workers"
for i in $(seq 0 $((K - 1))); do
  ( export SLURM_CPUS_PER_TASK=1
    for j in $(seq 0 $(( ${#JOBS[@]} - 1 ))); do
      job=${JOBS[$(( (i + j) % ${#JOBS[@]} ))]}; arm=${job%%:*}; pop=${job#*:}
      POP=$pop bash $N/run_arm.sh $arm reach
    done ) > $N/logs/samples_$(hostname -s)_w$i.log 2>&1 &
  sleep 90
done
wait
cd $REPO && $PY -m lawm.sim.dexwm.planspace.samples_curve --run $N --task reach
log "samples experiment done: $N/samples/SAMPLES.md"
