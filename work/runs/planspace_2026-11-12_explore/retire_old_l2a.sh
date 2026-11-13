#!/usr/bin/env bash
# Stop the grasp L2A worker that runs the OLD L2A map (commanded-step labels) without srun: reserve every other L2A grasp episode with a fresh lock
# (the worker skips locked episodes and exits after its current one), then move its result(s) to grasp/L2A_old_commanded_labels/ and re-enable L2A
# (run_arm.sh now uses maps/l2a_achieved.pt), so later workers run L2A with the fixed map into a clean grasp/L2A/.
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; D=$N/grasp/L2A
log() { echo "[$(date +%T)] $*"; }
running=$(ls $D/*.lock 2>/dev/null | xargs -rn1 basename); log "old L2A running: $running"
reserve() { for d in $(python3 -c "import json;print(' '.join(json.load(open('/vol/dissolve/vinay/work/runs/planspace/episodes/episodes_grasp.json'))['selected']))"); do
  f=$D/${d}_grasp.lock; [ -f $D/${d}_grasp.json ] && continue
  echo "$running" | grep -qx "${d}_grasp.lock" && continue
  echo '{"reserved": "L2A label fix, retire_old_l2a.sh"}' > $f; done; }
reserve
while ls $D/*.lock 2>/dev/null | xargs -rn1 basename | grep -qxF -f <(echo "$running"); do reserve; touch $D/*.lock 2>/dev/null; sleep 120; done
log "old L2A episode finished; letting the worker skip the reserved episodes"
for i in 1 2 3; do touch $D/*.lock 2>/dev/null; sleep 120; done
mv $D $N/grasp/L2A_old_commanded_labels && rm -f $N/grasp/L2A_old_commanded_labels/*.lock
echo "L2A with the OLD map (commanded-step labels, maps/l2a.pt); superseded by maps/l2a_achieved.pt. Not part of the results." > $N/grasp/L2A_old_commanded_labels/README.txt
sed -i '/^L2A$/d' $N/skip_arms
log "moved to grasp/L2A_old_commanded_labels; L2A re-enabled with maps/l2a_achieved.pt"
