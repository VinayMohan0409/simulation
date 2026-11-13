#!/usr/bin/env bash
# gpu19: after its grasp work (action_gpu19.sh) ends, run the reach eval for all 4 arms (shares episodes with the A100 workers via locks).
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
while ps -eo args | awk '$1=="bash" && $2 ~ /action_gpu19.sh/' | grep -q .; do sleep 60; done
echo "[$(date +%T)] gpu19: reach for L, A1, A2, L2A"
for a in L A1 A2 L2A; do bash $N/run_arm.sh $a reach; done
