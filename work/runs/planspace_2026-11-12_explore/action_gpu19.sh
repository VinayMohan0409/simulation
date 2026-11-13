#!/usr/bin/env bash
# gpu19: after its L2A worker is done and A2L (achieved) is trained, help rerun A1 and A2 grasp.
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
while kill -0 3915038 2>/dev/null || [ ! -f $N/markers/maps_achieved.done ] || grep -qx A1 $N/skip_arms; do sleep 60; done
echo "[$(date +%T)] gpu19: A1, A2 grasp"
bash $N/run_arm.sh A1 grasp; bash $N/run_arm.sh A2 grasp
