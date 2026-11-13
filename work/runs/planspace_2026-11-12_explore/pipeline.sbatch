#!/bin/bash
#SBATCH --job-name=ex_pipeline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# Usage: sbatch -p a100,a40 pipeline.sbatch   (waits for collection and tracks by itself)
exec bash /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/pipeline.sh
