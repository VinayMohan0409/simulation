# Planspace experiments: latent vs action search through the LAWM world model (DexWM kitchens)

| Folder | What |
|---|---|
| `episodes/` | the evaluation episodes shared by every run: `episodes_reach.json`, `episodes_grasp.json` (10 each, DexWM 50-demo eval set) |
| `../planspace_2026-11-12_explore/` | **the current run**: task-agnostic exploratory play, LoRA (stage 1 only), 4 arms. Moves here as `explore_2026-11-12/` once its jobs end |
| `archive_biased_object_centric_play/` | superseded runs (object-centric play); see its README |

Arms: A1 action search + IK execution; A2 action search + direct action; L latent search + IK; L2A latent search + latent-to-action map.
Original checkpoint (never modified): `latent-action-world-model/checkpoints/adaln_s0_k4/best.pt`.
Other folders in `work/runs/` (cross_decode, calibrate, check_retarget, export_extra, reach_demo133_fp32, _cache*) belong to other experiments.
