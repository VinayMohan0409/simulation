#!/usr/bin/env bash
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until grep -q "moved to" $N/logs/retire_A2_intermediate.log; do sleep 30; done; rm -f $N/grasp/A2.retiring; echo "[$(date +%T)] A2 retire done, marker removed"
