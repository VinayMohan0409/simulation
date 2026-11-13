#!/usr/bin/env bash
# Exploratory play collection (kinds=explore), production scene, 12 clips x 13 non-eval kitchens = 156 clips -> play/. Resumable per clip.
REPO=/vol/dissolve/vinay/latent-action-world-model; N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
export MUJOCO_GL=egl LAWM_DATA_ROOT=/vol/dissolve/vinay/work/data DEXWM_ROOT=/vol/dissolve/vinay/work/dexwm OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO/src:$DEXWM_ROOT/robot_sim_dexwm/robosuite_murp:$DEXWM_ROOT/robot_sim_dexwm/robocasa-murp
cd $REPO && /vol/dissolve/vinay/envs/lawm-model/bin/python -m lawm.sim.dexwm.planspace.play --out $N/play --per-scene 12 --workers ${WORKERS:-6} --kinds explore --scene production ${SCENES:+--scene-index $SCENES}
