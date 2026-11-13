# Shared environment for the explore-run scripts (sourced).
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; REPO=/vol/dissolve/vinay/latent-action-world-model; CK=$REPO/checkpoints/adaln_s0_k4_explore_lora
export LAWM_DATA_ROOT=/vol/dissolve/vinay/work/data DEXWM_ROOT=/vol/dissolve/vinay/work/dexwm MUJOCO_GL=egl OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO/src:$DEXWM_ROOT/robot_sim_dexwm/robosuite_murp:$DEXWM_ROOT/robot_sim_dexwm/robocasa-murp
PY=/vol/dissolve/vinay/envs/lawm-model/bin/python
EXPECTED=${EXPECTED:-156}
log() { echo "[$(date '+%F %T')] [$(hostname)${SLURM_JOB_ID:+ job $SLURM_JOB_ID}] $*"; }
n_clips() { ls $N/play 2>/dev/null | grep -c '_[0-9][0-9][0-9]\.npz$'; }
n_tracks() { ls $N/play 2>/dev/null | grep -c '_[0-9][0-9][0-9]\.tracks\.npz$'; }
