#!/bin/bash
#SBATCH --job-name=ex_action
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# Rerun A1/A2 grasp with the fixed action prior on one GPU (up to 4 workers; shares episodes with gpu19 through the locks).
# Waits until the old A1/A2 runs are moved aside. Usage: sbatch -p a100 rerun_action.sbatch
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until grep -q "moved to" $N/logs/retire_A1.log && grep -q "moved to" $N/logs/retire_A2.log; do sleep 60; done
ARMS_OVERRIDE="A1 A2" exec bash $N/eval_local.sh grasp
