#!/usr/bin/env bash
# Retire an arm's grasp results made with superseded code, without srun: reserve every unclaimed episode with a fresh lock (running workers skip
# them and exit after their current episode), wait for the running episodes, then move the folder to grasp/<ARM>_<SUFFIX>/ (kept, not results).
# Usage: retire_arm.sh ARM SUFFIX "reason"
ARM=$1; SUF=$2; WHY=$3; N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; D=$N/grasp/$ARM
log() { echo "[$(date +%T)] [$ARM] $*"; }
running=$(cd $D && ls *.lock 2>/dev/null); log "running at start: $(echo $running)"
reserve() { for d in $(python3 -c "import json;print(' '.join(json.load(open('/vol/dissolve/vinay/work/runs/planspace/episodes/episodes_grasp.json'))['selected']))"); do
  [ -f $D/${d}_grasp.json ] && continue; echo "$running" | grep -qx "${d}_grasp.lock" && continue
  echo '{"reserved": "'"$WHY"'"}' > $D/${d}_grasp.lock; done; }
still_running() { for l in $running; do [ -f $D/$l ] && return 0; done; return 1; }
reserve
while still_running; do reserve; touch $D/*.lock 2>/dev/null; sleep 60; done
log "running episodes finished"; for i in 1 2 3; do touch $D/*.lock 2>/dev/null; sleep 60; done
mv $D $N/grasp/${ARM}_$SUF && rm -f $N/grasp/${ARM}_$SUF/*.lock && echo "$WHY Not part of the results." > $N/grasp/${ARM}_$SUF/README.txt
log "moved to grasp/${ARM}_$SUF"
