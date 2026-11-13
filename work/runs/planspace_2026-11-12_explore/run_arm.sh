#!/usr/bin/env bash
# One worker of one arm of the EXPLORE-data evaluation (fine-tuned checkpoint adaln_s0_k4_explore_lora, no joint stage). Waits for the fine-tune pipeline.
# Usage: run_arm.sh ARM [TASK ...]      ARM = A1 | A2 | L | L2A, tasks default "reach grasp".
#   POP=16 run_arm.sh L reach -> sample-count experiment: CEM pop 16 x 4 iterations (64 samples), results in samples/reach/L_s64/ (main results untouched).
# Resumable and never overwrites: finished episodes (<demo>_<task>.json) are skipped; episodes another worker is running (fresh .lock) are skipped.
set -uo pipefail
ARM=${1:?ARM = A1|A2|L|L2A}; shift
TASKS=${*:-reach grasp}
REPO=/vol/dissolve/vinay/latent-action-world-model; R=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; CK=$REPO/checkpoints/adaln_s0_k4_explore_lora
EP=/vol/dissolve/vinay/work/runs/planspace/episodes  # episodes_reach.json, episodes_grasp.json (shared by every run)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LAWM_DATA_ROOT=/vol/dissolve/vinay/work/data DEXWM_ROOT=/vol/dissolve/vinay/work/dexwm OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO/src:$DEXWM_ROOT/robot_sim_dexwm/robosuite_murp:$DEXWM_ROOT/robot_sim_dexwm/robocasa-murp
export PLANSPACE_CHECKPOINT=$CK/stage1.pt PLANSPACE_PRIOR=$CK/prior.npz
PY=/vol/dissolve/vinay/envs/lawm-model/bin/python
log() { echo "[$(date +%H:%M:%S)] [$ARM@$(hostname)${SLURM_JOB_ID:+ job $SLURM_JOB_ID}] $*"; }
case $ARM in A1|A2) MAP="--g $CK/maps_achieved/g.pt --action-prior $CK/maps_achieved/action_prior.npz";; L) MAP="";; L2A) MAP="--l2a $CK/maps/l2a_achieved.pt";; *) log "unknown arm $ARM"; exit 2;; esac
POP=${POP:-}
if [ -n "$POP" ]; then CEM="--pop $POP --iters 4"; SUB=samples/; SUF=_s$((POP * 4)); else CEM=""; SUB=""; SUF=""; fi

if [ -f $R/skip_arms ] && grep -qx "$ARM" $R/skip_arms; then log "arm $ARM is listed in $R/skip_arms -> skipped"; exit 0; fi
until [ -f $R/markers/maps.done ]; do log "waiting for the fine-tune pipeline (markers/maps.done)"; sleep 120; done

# Preflight: fail fast with a clear message on a host that cannot run this.
[ -x $PY ] || { log "PREFLIGHT FAIL: $PY not found (is /vol/dissolve mounted here?)"; exit 3; }
cd $REPO
$PY - <<'PY' || { log "PREFLIGHT FAIL (see above)"; exit 3; }
import torch, mujoco, numpy as np
assert torch.cuda.is_available(), "no CUDA GPU visible"
free, total = torch.cuda.mem_get_info()
print(f"preflight: {torch.cuda.get_device_name(0)}, {free/2**30:.1f}/{total/2**30:.1f} GB free")
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (32, 32, 3)
import spconv.pytorch  # noqa: F401  (CUDA-only backbone dependency)
print("preflight: MuJoCo EGL render ok, spconv ok")
PY

for T in $TASKS; do
  EPF=$EP/episodes_$T${EPISODES_SUFFIX:-}.json  # EPISODES_SUFFIX=_20: the 20-episode selection (the first 10 are the original ones; finished episodes are skipped)
  until [ -f $EPF ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$EPF')).get('done') else 1)"; do log "waiting for $EPF (episode selection)"; sleep 120; done
  while [ -f $R/$T/$ARM.retiring ]; do log "waiting: the previous $T/$ARM run is being moved aside ($R/$T/$ARM.retiring)"; sleep 60; done
  OUT=$R/$SUB$T/$ARM$SUF
  mkdir -p $OUT $R/$T/groundtruth
  log "start $T -> $OUT"
  $PY -m lawm.sim.dexwm.planspace.evaluate --arm $ARM --task $T --episodes $EPF $MAP $CEM --out $OUT --scene production --video --need-mb 9000 2>&1 | tee -a $OUT/run.log
  [ -n "$POP" ] && { log "done $T (pop $POP)"; continue; }
  for d in $($PY -c "import json; print(' '.join(json.load(open('$EPF'))['selected']))"); do
    f=$R/$T/groundtruth/${d}_$T.mp4
    [ -f $f ] || $PY -m lawm.sim.dexwm.planspace.demo_video --demo $d --task $T --out $f >> $R/$T/groundtruth/run.log 2>&1
  done
  log "done $T"
done
if [ -z "$POP" ]; then
  $PY -m lawm.sim.dexwm.planspace.summarize --run $R --title "Planspace evaluation: 4 arms, exploratory play data, fine-tuned checkpoint adaln_s0_k4_explore_lora (no joint stage)" > /dev/null 2>&1 && log "summary: $R/EVAL_SUMMARY.md"
fi
