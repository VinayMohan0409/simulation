#!/bin/bash
#SBATCH --job-name=place
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/place_sam3/logs/%x_%j.out
# Everything for Place in one job: workers for the 4 arms on this GPU (per-episode locks; resubmit to resume; several jobs can run at once).
# Object: object_track.py (no segmentation model). Results: place_geom/<ARM>. Usage: sbatch -p a100 place.sbatch
# First 10 only: sbatch -p a100 --export=ALL,DEMOS="$(python3 -c "import json;print(' '.join(json.load(open('/vol/dissolve/vinay/work/runs/planspace/episodes/episodes_place_20.json'))['selected'][:10]))")" place.sbatch
R=/vol/dissolve/vinay/work/runs/place_sam3
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); K=$(( FREE / 16000 )); [ $K -gt 4 ] && K=4; [ $K -lt 1 ] && K=1
ARMS=(L A1 A2 L2A)
for i in $(seq 0 $((K - 1))); do
  ( for j in 0 1 2 3; do bash $R/run_arm.sh ${ARMS[$(( (i + j) % 4 ))]}; done ) >> $R/logs/place_$(hostname -s)_${SLURM_JOB_ID:-local}_w$i.log 2>&1 &
  sleep 120
done
wait
