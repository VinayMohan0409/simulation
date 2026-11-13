# Shared environment for the Place + SAM 3 run (sourced).
R=/vol/dissolve/vinay/work/runs/place_sam3; REPO=/vol/dissolve/vinay/latent-action-world-model
export LAWM_DATA_ROOT=/vol/dissolve/vinay/work/data DEXWM_ROOT=/vol/dissolve/vinay/work/dexwm MUJOCO_GL=egl OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO/src:$DEXWM_ROOT/robot_sim_dexwm/robosuite_murp:$DEXWM_ROOT/robot_sim_dexwm/robocasa-murp
export SAM3_PYTHON=/vol/dissolve/vinay/envs/sam3/bin/python
PY=/vol/dissolve/vinay/envs/lawm-model/bin/python
