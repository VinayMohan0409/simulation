#!/usr/bin/env bash
# Hand the fine-tune over from gpu19 to a Slurm pipeline job: when a new logs/ex_pipeline_<job>.out appears, stop gpu19's pipeline + fine-tune
# and move gpu19's partial stage-1 outputs to checkpoints/adaln_s0_k4_explore_lora_gpu19_stopped/ (kept, never used). One checkpoint only.
N=/vol/dissolve/vinay/work/runs/planspace_2026-11-12_explore; CK=/vol/dissolve/vinay/latent-action-world-model/checkpoints/adaln_s0_k4_explore_lora
OLD=$CK/../adaln_s0_k4_explore_lora_gpu19_stopped
known=$(ls $N/logs/ex_pipeline_*.out 2>/dev/null)
while true; do
  new=$(ls $N/logs/ex_pipeline_*.out 2>/dev/null | grep -vxF "$known")
  [ -n "$new" ] && break
  sleep 5
done
echo "[$(date +%T)] Slurm pipeline started: $new -> stopping gpu19"
for p in $(pgrep -f "bash ./pipeline.sh") $(pgrep -f "finetune_wm --stage 1"); do kill $p 2>/dev/null; done
sleep 5
for p in $(pgrep -f "finetune_wm --stage 1"); do kill -9 $p 2>/dev/null; done
mkdir -p $OLD
for f in stage1.pt stage1_sigma.pt stage1_report.json; do [ -f $CK/$f ] && mv $CK/$f $OLD/; done
echo "gpu19 fine-tune stopped at $(date +%T) after starting 08:19; partial outputs (if any) here, NOT used" > $OLD/README.txt
echo "[$(date +%T)] done; gpu19 partial outputs: $(ls $OLD)"
