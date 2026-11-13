#!/bin/bash
#SBATCH --job-name=ex_a2l
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# 1) retrain A2L on ACHIEVED step labels (train_g on encoder_latents_achieved.npz -> maps_achieved/), 2) release the A1/A2 hold,
# 3) rerun A1 and A2 grasp with it (up to 4 workers; shares episodes with gpu19 through the locks). Usage: sbatch -p a100 a2l_achieved.sbatch
source /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/env.sh
export PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
cd $REPO
if [ ! -f $N/markers/maps_achieved.done ]; then
  log "training A2L on achieved labels"
  $PY -m lawm.sim.dexwm.planspace.train_g --labels $CK/encoder_latents_achieved.npz --play $N/play --out $CK/maps_achieved > $N/logs/maps_achieved.log 2>&1 || { log "FAILED (logs/maps_achieved.log)"; exit 1; }
  touch $N/markers/maps_achieved.done
fi
$PY -c "import json; r=json.load(open('$CK/maps_achieved/g_report.json')); print('A2L (achieved) held-out hand error mm:', {k: r['final'][k]['hand_mm'] for k in ('g','encoder','zero')})"
sed -i '/^A1$/d; /^A2$/d' $N/skip_arms
log "A1/A2 released; rerunning grasp"
ARMS_OVERRIDE="A1 A2" bash $N/eval_local.sh grasp
