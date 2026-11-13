#!/bin/bash
#SBATCH --job-name=ex_eval
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%A_%a.out
# 10 workers of one arm; the per-episode locks give each worker a different episode. Waits for the fine-tune pipeline, so it can be submitted early.
# Usage: sbatch --job-name=ex_L -p a100 eval.sbatch L grasp   (one partition per job; POP=32 EPISODES_SUFFIX=_20 in the environment for the samples experiment)
ARM=${1:?usage: sbatch eval.sbatch ARM [TASK ...]}; shift
exec bash /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/run_arm.sh "$ARM" ${*:-grasp}
