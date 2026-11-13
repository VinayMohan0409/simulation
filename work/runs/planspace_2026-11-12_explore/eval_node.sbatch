#!/bin/bash
#SBATCH --job-name=ex_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# ONE job with several eval workers on its GPU (as many as fit, at most 4); shares the episodes with the other workers through the locks.
# Waits for the fine-tune pipeline. Usage: sbatch -p a100 eval_node.sbatch grasp
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
until [ -f $N/markers/maps.done ]; do sleep 120; done
exec bash $N/eval_local.sh ${1:-grasp}
