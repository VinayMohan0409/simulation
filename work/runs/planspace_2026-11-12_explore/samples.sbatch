#!/bin/bash
#SBATCH --job-name=ex_samples
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# Experiment S on one GPU. Usage: sbatch -p a100 samples.sbatch
exec bash /vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/samples.sh
