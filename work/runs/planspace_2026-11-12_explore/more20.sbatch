#!/bin/bash
#SBATCH --job-name=ex_more20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore/logs/%x_%j.out
# 10 more episodes per task (episodes_<task>_20.json; the first 10 are already done and skipped) for all 4 arms: reach first, then grasp.
# Each job runs as many workers as fit on its GPU (<= 4); every worker takes any free (arm, episode) through the per-episode locks and moves on.
# Waits for the episode selection to finish, so it can be submitted now. Usage: sbatch -p a100,a40,a30 more20.sbatch   (gpu19: bash more20.sbatch gpu19)
export EPISODES_SUFFIX=_20
EP=/vol/dissolve/vinay/work/runs/planspace/episodes
until python3 -c "import json,sys; sys.exit(0 if all(len(json.load(open('$EP/episodes_%s_20.json' % t))['selected']) == 20 for t in ('reach', 'grasp')) else 1)"; do sleep 60; done
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore
echo grasp > $N/then_task
[ "${1:-}" = gpu19 ] && export MAX_WORKERS=1
exec bash $N/eval_local.sh reach
