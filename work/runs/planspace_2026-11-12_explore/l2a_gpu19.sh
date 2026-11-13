#!/usr/bin/env bash
# gpu19: run the grasp L2A arm with the fixed map once retire_old_l2a.sh has moved the old-map run aside and re-enabled L2A.
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until ! grep -qx L2A $N/skip_arms && [ -d $N/grasp/L2A_old_commanded_labels ]; do sleep 60; done
echo "[$(date +%T)] L2A re-enabled -> starting on gpu19"
exec bash $N/run_arm.sh L2A grasp
