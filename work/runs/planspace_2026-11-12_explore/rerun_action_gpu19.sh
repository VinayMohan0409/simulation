#!/usr/bin/env bash
# gpu19: once the old A1/A2 grasp runs are moved aside (retire_arm.sh) and gpu19's L2A worker is done, rerun A1 and A2 with the fixed action prior.
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until grep -q "moved to" $N/logs/retire_A1.log && grep -q "moved to" $N/logs/retire_A2.log && ! pgrep -f "l2a_gpu19.sh" > /dev/null; do sleep 60; done
echo "[$(date +%T)] starting A1, A2 (fixed action prior) on gpu19"
bash $N/run_arm.sh A1 grasp; bash $N/run_arm.sh A2 grasp
