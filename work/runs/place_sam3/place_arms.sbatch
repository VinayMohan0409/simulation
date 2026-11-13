#!/bin/bash
#SBATCH --job-name=place_arms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/place_sam3/logs/%x_%j.out
# Place, arms A1 / A2 / L2A on the 20 place episodes (same order and same settings as arm L: place_geom/L, 12/20). One job = K workers on its GPU
# (K from the free memory, ~9 GB per worker, at most 4). Every worker goes through the three arms (starting at a different one), and inside an arm takes
# the next episode nobody else holds (per-episode lock files, shared with the gpu19 queue and every other job). Never overwrites: finished episodes
# (place_geom/<ARM>/<demo>_place.json) are skipped, and evaluate refuses an arm folder written with other settings. Resubmit any time to resume.
# Submit one job per partition (one partition per job), e.g.:
#   sbatch -p a100 place_arms.sbatch
#   sbatch -p a30 place_arms.sbatch
#   sbatch -p a40 place_arms.sbatch
R=/vol/dissolve/vinay/work/runs/place_sam3
export DEMOS="demo_133 demo_606 demo_1208 demo_1341 demo_870 demo_686 demo_1076 demo_665 demo_1142 demo_793 demo_131 demo_907 demo_769 demo_118 demo_831 demo_170 demo_1364 demo_1311 demo_1008 demo_731"
export NEED_MB=9000
ARMS=(A1 A2 L2A)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); K=$(( FREE / 10000 )); [ $K -gt 4 ] && K=4; [ $K -lt 1 ] && K=1
echo "$(date +%H:%M:%S) $(hostname -s) job ${SLURM_JOB_ID}: $K workers"
for i in $(seq 0 $((K - 1))); do
  ( for j in 0 1 2; do bash $R/run_arm.sh ${ARMS[$(( (i + j) % 3 ))]}; done ) >> $R/logs/place_arms_$(hostname -s)_${SLURM_JOB_ID:-local}_w$i.log 2>&1 &
  sleep 90  # stagger the kitchen builds and the model loads
done
wait
echo "$(date +%H:%M:%S) job ${SLURM_JOB_ID} done"
