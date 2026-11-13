#!/usr/bin/env bash
# Several eval workers on THIS machine's GPU (no extra Slurm jobs): K = how many fit (11 GB each, at most MAX_WORKERS); worker i starts with arm i mod 4,
# then helps with the other arms. Per-episode locks: no episode runs twice; finished ones are skipped. Usage: eval_local.sh TASK (grasp | reach)
TASK=${1:-grasp}; N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; ARMS=(${ARMS_OVERRIDE:-L A1 A2 L2A})
source $N/env.sh
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0} | head -1)
K=$(( FREE / 11000 )); [ $K -gt ${MAX_WORKERS:-4} ] && K=${MAX_WORKERS:-4}; [ $K -lt 1 ] && K=1
log "eval $TASK: $FREE MB free -> $K workers"
for i in $(seq 0 $((K - 1))); do
  ( export SLURM_CPUS_PER_TASK=1; for j in $(seq 0 $(( ${#ARMS[@]} - 1 ))); do a=${ARMS[$(( (i + j) % ${#ARMS[@]} ))]}; bash $N/run_arm.sh $a $TASK; done ) >> $N/logs/eval_${TASK}_$(hostname -s)_w$i.log 2>&1 &
  sleep 90  # stagger model loading so the free-memory checks see each other
done
wait
log "eval $TASK: all workers done"
# Follow-up task queued in $N/then_task (e.g. "reach"): the same workers then run it for all 4 arms (finished episodes are skipped).
if [ -f $N/then_task ] && [ "$TASK" != "$(cat $N/then_task)" ]; then
  NEXT=$(cat $N/then_task); log "follow-up: $NEXT for all arms"
  ARMS_OVERRIDE="L A1 A2 L2A" exec bash $N/eval_local.sh $NEXT
fi
