"""Fine-tune LAWM on the production-scene play data with LAC-WM's recipe (Huang et al., ICML 2026, sec. 3.2), into a NEW checkpoint (the original is only read):
  stage 1: LoRA (rank 2) on the latent encoder and the decoder, end to end on the play windows (encoder latents);
  stage 2: (labels + train_g with PLANSPACE_CHECKPOINT = stage 1) the action-to-latent map on the frozen stage-1 model;
  stage 3: LoRA on the decoder jointly with the action-to-latent map, the decoder now fed the map's latents.
Loss: the training run's movement-weighted Huber on normalised displacements with per-point uncertainty, w * (Huber / sigma + lam * log sigma), over the tracked rows only:
the FK hand rows always; camera scene rows only where tracks exist (<clip>.tracks.npz, from CoTracker; without them the scene rows are input only)."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lawm.model.losses import HUBER_DELTA, MOVE_KAPPA, MOVE_TAU
from lawm.sim.dexwm.planspace import tracks as tracks_io
from lawm.sim.dexwm.planspace.common import CHECKPOINT, R_MODEL, REPO, wait_for_gpu, write_json

ORIGINAL = REPO / "checkpoints/adaln_s0_k4/best.pt"
from lawm.sim.dexwm.planspace.g_model import ActionToLatent
from lawm.sim.dexwm.planspace.labels import load, window
from lawm.sim.dexwm.planspace.lora import add_lora, merge


class Sigma(nn.Module):
    """Per-point, per-horizon log sigma read from the decoder head's input (a forward hook), zero-initialised so training starts from the plain loss."""

    def __init__(self, decoder, n_steps):
        super().__init__()
        head_in = decoder.head[0].in_features if isinstance(decoder.head, nn.Sequential) else decoder.head.in_features
        self.lin = nn.Linear(head_in, n_steps)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.n_steps, self.feat = n_steps, None
        decoder.head.register_forward_hook(lambda mod, inp, out: setattr(self, "feat", inp[0]))

    def log_sigma(self):
        """(N, n_steps) log sigma of the last decode (one cloud), horizon h from its own channel."""
        rep = self.lin(self.feat).reshape(self.n_steps, -1, self.n_steps)
        return torch.stack([rep[h, :, h] for h in range(self.n_steps)], dim=1).clamp(-4.0, 4.0)


class PlayWindows:
    """Play windows (training frame): the measured frames (6, N, 3) the encoder sees, the targets (6, N, 3) (measured hand rows; CoTracker positions for the tracked scene rows),
    the tracked mask (N,) (FK hand rows always, scene rows where tracked) and the actions (5, 22)."""

    def __init__(self, play_dir, lab):
        self.dir, self.lab, self.cache, self.tracks = Path(play_dir), lab, {}, {}

    def _tracks(self, f):
        if f not in self.tracks:
            path = self.dir / f.replace(".npz", ".tracks.npz")
            self.tracks[f] = tracks_io.load(path) if path.exists() else {}
        return self.tracks[f]

    def has_tracks(self, i):
        """Whether window i has scene tracks."""
        return int(self.lab["start"][i]) in self._tracks(str(self.lab["rec_file"][i]))

    def get(self, i):
        f = str(self.lab["rec_file"][i])
        if f not in self.cache:
            self.cache[f] = dict(np.load(self.dir / f))
        rec = self.cache[f]
        s = int(self.lab["start"][i])
        frames, _, act = window(rec, s)
        targets, tracked = frames.copy(), rec["segment"] == 0
        tr = self._tracks(f).get(s)
        if tr is not None:
            rows, pos = tr
            targets[1:, rows] = pos @ R_MODEL.T
            tracked = tracked.copy()
            tracked[rows] = True
        return frames, targets, tracked, act


def window_loss(model, sigma, frames, targets, tracked, z, lam):
    """Uncertainty-weighted, movement-weighted Huber over the tracked rows of one window, decoded from latents z (5, nq, zd) on frame 0."""
    from lawm.sim import model_adapter
    dev = z.device
    c = torch.as_tensor(targets - frames[0].mean(0, keepdims=True), dtype=torch.float32, device=dev)
    with torch.no_grad():
        chain = model_adapter.encode_cloud(model, frames[0])  # the frozen backbone
    outs = model.decode_all_steps(chain, z.unsqueeze(1))
    log_s = sigma.log_sigma()
    mask = torch.as_tensor(tracked, device=dev)
    total = 0.0
    for k, o in enumerate(outs):
        gt = c[k + 1] - c[0]
        gt_n = (gt - model.out_mean_windowed[k]) / model.out_std_windowed[k]
        err = nn.functional.huber_loss(o["delta_norm"].squeeze(1).float(), gt_n, delta=HUBER_DELTA, reduction="none").mean(-1)
        w = torch.sigmoid(MOVE_KAPPA * (gt.norm(dim=-1) - MOVE_TAU)) * mask
        w = w / w.sum().clamp_min(1e-6)
        ls = log_s[:, k]
        total = total + (w * (err * torch.exp(-ls) + lam * ls)).sum()
    return total / len(outs)


def encoder_latents(model, frames):
    """Latents (5, nq, zd) of a window from the (LoRA) encoder, with gradients."""
    dev = next(model.parameters()).device
    c = torch.as_tensor(frames - frames[0].mean(0, keepdims=True), dtype=torch.float32, device=dev)
    _, z = model.encode_window(c, torch.tensor([c.shape[1]], device=dev).int())
    return z.squeeze(1)


@torch.no_grad()
def hand_check(model, wins, rows, a2l=None):
    """Held-out hand rows, step 5: true and predicted motion (mm), their direction cosine (median) and the mean hand error over the steps (mm); latents from the encoder, or from a2l."""
    dev = next(model.parameters()).device
    true_mm, pred_mm, cos, err = [], [], [], []
    for i in rows:
        frames, _, _, act = wins.get(i)
        if a2l is None:
            z = encoder_latents(model, frames)
        else:
            u = (torch.as_tensor(act, device=dev)[None] - a2l.act_mean) / a2l.act_std
            z = a2l.latents(u)[0]
        from lawm.sim import model_adapter
        chain = model_adapter.encode_cloud(model, frames[0])
        outs = model.decode_all_steps(chain, z.unsqueeze(1))
        h = torch.as_tensor(wins.cache[str(wins.lab["rec_file"][i])]["segment"] == 0, device=dev)  # FK hand rows (tracked exactly)
        pred = torch.stack([o["delta"].squeeze(1) for o in outs])[:, h]
        c = torch.as_tensor(frames, dtype=torch.float32, device=dev)
        true = (c[1:] - c[0][None])[:, h]
        true_mm.append(float(true[-1].norm(dim=-1).mean()) * 1000)
        pred_mm.append(float(pred[-1].norm(dim=-1).mean()) * 1000)
        cos.append(float(nn.functional.cosine_similarity(pred[-1].mean(0), true[-1].mean(0), dim=0)))
        err.append(float((pred - true).norm(dim=-1).mean()) * 1000)
    t, p = np.array(true_mm), np.array(pred_mm)
    mv = t > 10.0
    return dict(true_step5_mm=round(float(t[mv].mean()), 1), pred_step5_mm=round(float(p[mv].mean()), 1), size_ratio=round(float(p[mv].mean() / t[mv].mean()), 3),
                cosine_median=round(float(np.median(np.array(cos)[mv])), 3), hand_err_mm=round(float(np.mean(err)), 1), windows=int(mv.sum()))


def save_merged(model, path, extra):
    """A checkpoint in the original format ({"model": state}) with the adapters folded in; the training model is left as is."""
    m = copy.deepcopy(model)
    merge(m)
    torch.save(dict(model=m.state_dict(), extra=extra), path)


def train(params, loss_fn, steps, accum, lr, log, check=None, every=250, metric=None, patience=None, min_delta=0.0, on_best=None):
    """AdamW with cosine decay over `steps` updates of `accum` windows each; check() is logged every `every` updates.
    Early stopping (metric set): check()[metric] is the held-out score (lower is better); on_best() runs at every improvement of at least min_delta,
    and training stops after `patience` checks in a row without one. Returns the validation curve [(step, check result)]."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    curve, best, bad = [], float("inf"), 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for _ in range(accum):
            loss = loss_fn() / accum
            loss.backward()
            tot += float(loss)
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            log(f"step {step}/{steps} loss {tot:.4f} ({time.time() - t0:.0f}s)")
        if check is not None and (step + 1) % every == 0:
            res = check()
            curve.append((step + 1, res))
            log(f"check after {step + 1}: {res}")
            if metric is not None:
                if res[metric] < best - min_delta:
                    best, bad = res[metric], 0
                    if on_best is not None:
                        on_best()
                    log(f"  best {metric} {best:.2f} -> saved")
                else:
                    bad += 1
                    if patience is not None and bad >= patience:
                        log(f"  stop: no improvement of {min_delta} in {metric} for {patience} checks (best {best:.2f})")
                        break
    return curve


def main():
    """python -m lawm.sim.dexwm.planspace.finetune_wm --stage {1,3} --labels labels.npz --play DIR --out CKPT_DIR [--a2l g.pt (stage 3)]; the base model is PLANSPACE_CHECKPOINT (the original for stage 1, stage 1's for stage 3)."""
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=(1, 3), required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--a2l", default=None)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--lam", type=float, default=1.0, help="weight of log sigma in the uncertainty loss")
    p.add_argument("--check-windows", type=int, default=150)
    p.add_argument("--check-every", type=int, default=250, help="updates between held-out checks")
    p.add_argument("--patience", type=int, default=None, help="stage 1: stop after this many checks without improvement of the held-out hand error; keep the best")
    p.add_argument("--min-delta", type=float, default=0.5, help="mm of held-out hand error that counts as an improvement")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert out.resolve() != ORIGINAL.parent.resolve(), "never write into the original checkpoint's folder"
    logf = open(out / f"stage{a.stage}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(0)
    wait_for_gpu(4000, log)
    dev = torch.device("cuda")
    model = load(False)
    for prm in model.parameters():
        prm.requires_grad_(False)
    lab = dict(np.load(a.labels))
    train_rows, val_rows = np.flatnonzero(lab["split"] == "train"), np.flatnonzero(lab["split"] == "val")
    check_rows = np.random.default_rng(2).choice(val_rows, min(a.check_windows, len(val_rows)), replace=False)
    wins, rng = PlayWindows(a.play, lab), np.random.default_rng(a.stage)
    with_tracks = np.array([wins.has_tracks(i) for i in train_rows])  # every window trains its FK hand rows; scene rows only where CoTracker tracked them
    log(f"{int(with_tracks.sum())} of {len(with_tracks)} training windows have CoTracker scene tracks")
    sigma = Sigma(model.decoder, model.n_steps).to(dev)
    log(f"base {CHECKPOINT}; {len(train_rows)} train / {len(val_rows)} val windows; rank {a.rank}, lam {a.lam}")

    if a.stage == 1:
        model.eval()
        before = hand_check(model, wins, check_rows)
        log(f"before (base model, encoder latents): {before}")
        params = add_lora(model.latent_encoder, a.rank, a.rank) + add_lora(model.decoder, a.rank, a.rank) + list(sigma.parameters())

        def loss_fn():
            frames, targets, tracked, _ = wins.get(int(rng.choice(train_rows)))
            return window_loss(model, sigma, frames, targets, tracked, encoder_latents(model, frames), a.lam)
        if a.patience is None:
            curve = train(params, loss_fn, a.steps, a.accum, a.lr, log, check=lambda: hand_check(model, wins, check_rows), every=a.check_every)
            after = hand_check(model, wins, check_rows)
            log(f"after stage 1 (encoder latents): {after}")
            save_merged(model, out / "stage1.pt", dict(stage=1, base=str(CHECKPOINT), rank=a.rank, lam=a.lam, steps=a.steps))
            torch.save(sigma.state_dict(), out / "stage1_sigma.pt")
        else:
            def on_best():
                save_merged(model, out / "stage1.pt", dict(stage=1, base=str(CHECKPOINT), rank=a.rank, lam=a.lam, max_steps=a.steps, early_stopping="held-out hand_err_mm"))
                torch.save(sigma.state_dict(), out / "stage1_sigma.pt")
            curve = train(params, loss_fn, a.steps, a.accum, a.lr, log, check=lambda: hand_check(model, wins, check_rows), every=a.check_every,
                          metric="hand_err_mm", patience=a.patience, min_delta=a.min_delta, on_best=on_best)
            assert (out / "stage1.pt").exists(), "no held-out check ran"
            after = min((r for _, r in curve), key=lambda r: r["hand_err_mm"])
            log(f"stage 1 best (saved, encoder latents): {after}")
        write_json(out / "stage1_report.json", dict(base=str(CHECKPOINT), before=before, after=after, curve=curve, check_every=a.check_every, patience=a.patience))
    else:
        assert a.a2l, "stage 3 needs the stage-2 action-to-latent map"
        a2l = ActionToLatent.load(a.a2l, dev)
        a2l.train()
        before = hand_check(model, wins, check_rows, a2l)
        log(f"before stage 3 (A2L latents): {before}")
        params = add_lora(model.decoder, a.rank, a.rank) + list(sigma.parameters()) + list(a2l.parameters())

        def loss_fn():
            frames, targets, tracked, act = wins.get(int(rng.choice(train_rows)))
            u = (torch.as_tensor(act, device=dev)[None] - a2l.act_mean) / a2l.act_std
            return window_loss(model, sigma, frames, targets, tracked, a2l.latents(u)[0], a.lam)
        train(params, loss_fn, a.steps, a.accum, a.lr, log, check=lambda: hand_check(model, wins, check_rows, a2l))
        a2l.eval()
        after = hand_check(model, wins, check_rows, a2l)
        log(f"after stage 3 (A2L latents): {after}")
        save_merged(model, out / "final.pt", dict(stage=3, base=str(CHECKPOINT), rank=a.rank, lam=a.lam, steps=a.steps))
        a2l.save(out / "a2l_joint.pt", extra=dict(stage=3))
        write_json(out / "stage3_report.json", dict(base=str(CHECKPOINT), before=before, after=after))


if __name__ == "__main__":
    main()
