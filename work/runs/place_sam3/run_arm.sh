#!/usr/bin/env bash
# One worker of one arm of the PLACE evaluation (2-camera production scene, object from object_track.py -- no segmentation model --, place_simple cost with the
# one-sided Chamfer object goal), with exactly the arm settings of the explore ablation (checkpoint adaln_s0_k4_explore_lora, same maps / priors).
# Optional: DEMOS="demo_133 demo_606 ..." runs only those episodes (still from the 20; resumable, later runs add the rest).
# Resumable, never overwrites (per-episode json + locks). Usage: run_arm.sh ARM    (ARM = A1 | A2 | L | L2A)
set -uo pipefail
ARM=${1:?ARM = A1|A2|L|L2A}
source /vol/dissolve/vinay/work/runs/place_sam3/env.sh
CK=$REPO/checkpoints/adaln_s0_k4_explore_lora; EPF=/vol/dissolve/vinay/work/runs/planspace/episodes/episodes_place_20.json
export PYOPENGL_PLATFORM=egl PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
log() { echo "[$(date +%H:%M:%S)] [$ARM@$(hostname)${SLURM_JOB_ID:+ job $SLURM_JOB_ID}] $*"; }
case $ARM in A1|A2) MAP="--g $CK/maps_achieved/g.pt --action-prior $CK/maps_achieved/action_prior.npz";; L) MAP="";; L2A) MAP="--l2a $CK/maps/l2a_achieved.pt";; *) log "unknown arm $ARM"; exit 2;; esac
until [ -f $EPF ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$EPF')).get('done') else 1)"; do log "waiting for $EPF"; sleep 120; done
OUT=$R/place_geom/$ARM; mkdir -p $OUT; cd $REPO
log "start place -> $OUT"
$PY -m lawm.sim.dexwm.planspace.evaluate --arm $ARM --task place --episodes $EPF ${DEMOS:+--demos $DEMOS} $MAP --out $OUT --scene production --video --need-mb ${NEED_MB:-12000} 2>&1 | tee -a $OUT/run.log
log "done place"
