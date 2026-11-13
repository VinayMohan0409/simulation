"""Window labels of the play data: for evenly spaced windows of every recording, the encoder's latents, the pooled frame-0 feature, the hand-state features and the 5-step action chunk."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lawm.sim import model_adapter
from lawm.sim.dexwm.planspace.common import R_MODEL, CHECKPOINT, CONTROL_PER_MODEL_STEP, N_STEPS, RUN_MANIFEST, wait_for_gpu, write_json
from lawm.sim.dexwm.planspace.g_model import pooled_feature, state_features, window_actions

WINDOW_STRIDE = 3  # control steps between window starts


def window_starts(T):
    """Start control steps of the windows of a recording with T + 1 frames."""
    span = N_STEPS * CONTROL_PER_MODEL_STEP
    return list(range(0, T - span + 1, WINDOW_STRIDE))


def window(rec, s):
    """(frames (6, N, 3) of the window in the training frame, ctx (28,), actions (5, 22)) of the window starting at control step s."""
    idx = [s + CONTROL_PER_MODEL_STEP * k for k in range(N_STEPS + 1)]
    cmd = [s + CONTROL_PER_MODEL_STEP * h - 1 for h in range(1, N_STEPS + 1)]  # the command that drove the hand into frame h
    frames = rec["coords"][idx] @ R_MODEL.T  # in the training frame, as the model is fed
    ctx = state_features(rec["root_pose"][s], rec["q"][s], rec["coords"][s], rec["segment"])
    act = window_actions(rec["root_pose"][s], rec["q"][s], rec["cmd_pos"][cmd], rec["cmd_rot"][cmd], rec["cmd_q"][cmd])
    return frames, ctx, act


def split_of(scene, val_every=5):
    """'val' for every val_every-th scene (by demo number), else 'train'."""
    return "val" if int(scene.split("_")[1]) % val_every == 0 else "train"


@torch.no_grad()
def encode(model, frames):
    """(z (5, nq, zd), pooled (C,)) of one window, centred on its frame-0 mean as in training."""
    device = next(model.parameters()).device
    c = torch.as_tensor(frames - frames[0].mean(0, keepdims=True), dtype=torch.float32, device=device)
    chain, z = model.encode_window(c, torch.tensor([c.shape[1]], device=device).int())
    return z.squeeze(1).cpu().numpy(), pooled_feature(chain).cpu().numpy()


@torch.no_grad()
def encode_batch(model, windows):
    """encode() of several windows in one call, batched as in training (windows concatenated, each centred on its own frame-0 mean, offsets): (z (B, 5, nq, zd), pooled (B, C))."""
    device = next(model.parameters()).device
    cs = [w - w[0].mean(0, keepdims=True) for w in windows]
    c = torch.as_tensor(np.concatenate(cs, axis=1), dtype=torch.float32, device=device)
    offset = torch.as_tensor(np.cumsum([w.shape[1] for w in cs]), device=device).int()
    chain, z = model.encode_window(c, offset)
    ends = chain[0].offset.tolist()
    pooled = torch.stack([chain[0].feat[b:e].float().mean(0) for b, e in zip([0] + ends[:-1], ends)])
    return z.permute(1, 0, 2, 3).cpu().numpy(), pooled.cpu().numpy()


def clip_index(f, split):
    """The window index of one recording (the rows of labels.npz without encoder latents): what LoRA stage 1 needs."""
    rec = np.load(f)
    starts = window_starts(len(rec["cmd_pos"]))
    n = len(starts)
    return dict(rec_file=np.array([f.name] * n), start=np.array(starts), split=np.array([split] * n), kind=np.array([str(rec["kind"])] * n))


def clip_labels(model, f, split, batch=12):
    """All windows of one recording: dict of arrays (the rows of labels.npz)."""
    rec = dict(np.load(f))
    starts = window_starts(len(rec["cmd_pos"]))
    wins = [window(rec, s) for s in starts]
    zs, pooled = [], []
    for j in range(0, len(wins), batch):
        z, p = encode_batch(model, [w[0] for w in wins[j:j + batch]])
        zs.append(z)
        pooled.append(p)
    n = len(starts)
    return dict(rec_file=np.array([f.name] * n), start=np.array(starts), split=np.array([split] * n), z=np.concatenate(zs), pooled=np.concatenate(pooled),
                ctx=np.stack([w[1] for w in wins]), act=np.stack([w[2] for w in wins]), kind=np.array([str(rec["kind"])] * n))


def main():
    """python -m lawm.sim.dexwm.planspace.labels --play DIR --out labels.npz [--clips-per-scene N] [--shard k --num-shards n | --combine]: every clip's labels are saved in <out stem>_parts/ as soon as it is done
    (a restart skips finished clips); labels.npz is the concatenation of the chosen clips."""
    p = argparse.ArgumentParser()
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fake-backbone", action="store_true", help="CPU smoke test with the tests' stand-in backbone")
    p.add_argument("--clips-per-scene", type=int, default=None, help="use only clips <demo>_000 .. <demo>_{N-1} of every kitchen")
    p.add_argument("--split-by-file", type=int, default=0, help="smoke tests: every N-th recording is held out instead of whole scenes")
    p.add_argument("--batch", type=int, default=12, help="windows per encoder call (batched as in training)")
    p.add_argument("--shard", type=int, default=0, help="this worker labels clips shard, shard + num_shards, ... and does not write labels.npz")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--combine", action="store_true", help="only concatenate the finished parts into labels.npz")
    p.add_argument("--index-only", action="store_true", help="only the window index (rec_file, start, split, kind): no encoder, no GPU (all LoRA stage 1 reads)")
    a = p.parse_args()
    if not (a.fake_backbone or a.combine or a.index_only):
        wait_for_gpu(2500)
    model = None if (a.combine or a.index_only) else load(a.fake_backbone)
    files = sorted(f for f in Path(a.play).glob("demo_*.npz") if not f.name.endswith((".rgbd.npz", ".tmp.npz", ".tracks.npz")))
    if a.clips_per_scene is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) < a.clips_per_scene]
    parts = Path(a.out).with_name(Path(a.out).stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for i, f in enumerate(files):
        part = parts / f.name
        if a.combine or part.exists() or i % a.num_shards != a.shard:
            continue
        split = ("val" if i % a.split_by_file == a.split_by_file - 1 else "train") if a.split_by_file else split_of(f.name.rsplit("_", 1)[0])
        np.savez(part.with_suffix(".tmp.npz"), **(clip_index(f, split) if a.index_only else clip_labels(model, f, split, a.batch)))
        part.with_suffix(".tmp.npz").replace(part)
        done += 1
        if done % 10 == 0:
            print(f"{i + 1}/{len(files)} recordings ({done} this run, {time.time() - t0:.0f}s)", flush=True)
    if a.num_shards > 1 and not a.combine:
        print(f"shard {a.shard}/{a.num_shards} done", flush=True)
        return
    missing = [f.name for f in files if not (parts / f.name).exists()]
    assert not missing, f"{len(missing)} clips not labelled yet, e.g. {missing[:3]}"
    chunks = [dict(np.load(parts / f.name)) for f in files]
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    np.savez(a.out, **out)
    write_json(Path(a.out).with_suffix(".json"), dict(windows=len(out["start"]), train=int((out["split"] == "train").sum()), val=int((out["split"] == "val").sum()), recordings=len(files),
                                                      index_only=bool(a.index_only)))
    print(f"{len(out['start'])} windows -> {a.out}", flush=True)


def load(fake=False):
    """The LAWM model on the GPU, or with the stand-in backbone on the CPU for smoke tests."""
    if fake:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tests"))
        import fake_backbone
        from lawm.model.chunked import ChunkedLAWM
        run = json.loads(Path(RUN_MANIFEST).read_text())
        args, ptv3 = run["args"], {k: tuple(v) if isinstance(v, list) else v for k, v in run["run"]["ptv3_config"].items()}
        n_steps = run["config"]["window"]["n_steps"]
        model = ChunkedLAWM(dict(per_step_mean=np.zeros((n_steps, 3)), per_step_var=np.ones((n_steps, 3))), n_steps=n_steps, cfg=ptv3, z_dim=args["z_dim"],
                            n_queries=args["n_queries"], ada_weight_init_std=args["ada_weight_init_std"], backbone=fake_backbone.FakeBackbone(),
                            decoder_self_attention=args["decoder_self_attention"], voxel_dedup=False)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith("ptv3.")}, strict=False)
        return model.eval()
    return model_adapter.load_model(CHECKPOINT, RUN_MANIFEST, torch.device("cuda"))


if __name__ == "__main__":
    main()
