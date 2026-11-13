#!/usr/bin/env bash
# gpu19 helper: waits for the fine-tune pipeline (markers/maps.done), then runs eval workers here as well (as many as fit, at most 1 on the 11 GB 2080 Ti).
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until [ -f $N/markers/maps.done ]; do sleep 120; done
MAX_WORKERS=1 exec bash $N/eval_local.sh ${1:-grasp}
