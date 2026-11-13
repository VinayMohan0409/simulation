#!/usr/bin/env bash
# gpu19: retrain A2L on achieved labels (same command as a2l_achieved.sbatch), then release the A1/A2 hold. gpu19's action_gpu19.sh then runs A1, A2.
source /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/env.sh
export PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
cd $REPO
[ -f $N/markers/maps_achieved.done ] && { log "already trained"; exit 0; }
log "training A2L on achieved labels (gpu19)"
$PY -m lawm.sim.dexwm.planspace.train_g --labels $CK/encoder_latents_achieved.npz --play $N/play --out $CK/maps_achieved > $N/logs/maps_achieved.log 2>&1 || { log "FAILED"; exit 1; }
touch $N/markers/maps_achieved.done
$PY -c "import json; r=json.load(open('$CK/maps_achieved/g_report.json')); print('A2L (achieved) held-out hand error mm:', {k: r['final'][k]['hand_mm'] for k in ('g','encoder','zero')})"
sed -i '/^A1$/d; /^A2$/d' $N/skip_arms; log "A1/A2 released"
