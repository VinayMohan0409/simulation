"""Train the per-step action-to-latent map (A2L; saved as g.pt for the planner) on the play-window labels: latent regression, then a fine-tune through the frozen decoder on the true point displacements, checked against the encoder on held-out kitchens; then the per-step latent-to-action map (L2A, l2a.pt), checked on held-out kitchens."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lawm.sim import model_adapter
from lawm.sim.dexwm.planspace.common import PRIOR, CHECKPOINT, wait_for_gpu, write_json
from lawm.sim.dexwm.planspace.g_model import ActionToLatent, LatentToAction, pooled_feature, step_inputs
from lawm.sim.dexwm.planspace.labels import load, window

GATE_RATIO = 1.25  # g passes when its held-out hand error is at most this multiple of the encoder's (the best any g can do); reported, never stops the pipeline


def stats_of(lab, train, z_mean, z_std):
    """Normalisation statistics from the training windows. The latent statistics are the planner's prior pooled over steps (one mean and spread for every step, so the per-step maps never depend on the step index)."""
    def ms(x):
        return x.mean(0), np.maximum(x.std(0), 1e-3)
    act = lab["act"][train]
    prev, inc = (t.numpy() for t in step_inputs(torch.as_tensor(act)))
    act_m, act_s = ms(act)
    ctx_m, ctx_s = ms(lab["ctx"][train])
    pm, ps = ms(lab["pooled"][train])
    prev_m, prev_s = ms(prev.reshape(-1, prev.shape[-1]))
    inc_m, inc_s = ms(inc.reshape(-1, inc.shape[-1]))
    zm = z_mean.mean(0, keepdims=True)
    zs = np.sqrt((z_std ** 2 + (z_mean - zm) ** 2).mean(0, keepdims=True))
    n = z_mean.shape[0]
    return dict(act_mean=act_m, act_std=act_s, ctx_mean=ctx_m, ctx_std=ctx_s, pooled_mean=pm, pooled_std=ps, z_mean=np.repeat(zm, n, 0), z_std=np.repeat(np.maximum(zs, 1e-3), n, 0),
                prev_mean=prev_m, prev_std=prev_s, inc_mean=inc_m, inc_std=inc_s)


def l2a_batch(lab, rows, m, device):
    """(normalised latents, state before each step, start state, pooled, normalised true increments) of some windows."""
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    prev, inc = step_inputs(t(lab["act"][rows]))
    return (t(lab["z"][rows]) - m.z_mean) / m.z_std, prev, t(lab["ctx"][rows]), t(lab["pooled"][rows]), (inc - m.inc_mean) / m.inc_std


def l2a_errors(m, lab, rows, device):
    """Held-out RMS error of the step increments, per group (wrist mm, wrist deg, fingers rad), for L2A and for always predicting the mean increment."""
    zn, prev, c, p, inc_n = l2a_batch(lab, rows, m, device)
    with torch.no_grad():
        pred = m.increments(zn)
    true = inc_n * m.inc_std + m.inc_mean
    out = {}
    for name, guess in (("l2a", pred), ("mean_increment", m.inc_mean.expand_as(true))):
        e = (guess - true).reshape(-1, true.shape[-1])
        out[name] = dict(wrist_mm=round(float(e[:, :3].norm(dim=1).pow(2).mean().sqrt()) * 1000, 2), wrist_deg=round(float(np.degrees(float(e[:, 3:6].norm(dim=1).pow(2).mean().sqrt()))), 2),
                         fingers_rad=round(float(e[:, 6:].pow(2).mean().sqrt()), 4))
    return out


def fit_l2a(lab, train, val, stats, device, steps, log):
    """Train the latent-to-action map on the encoder's latents (MSE on normalised increments); keeps the best validation state."""
    m = LatentToAction(lab["pooled"].shape[1], lab["z"].shape[1:], stats=stats).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vb = l2a_batch(lab, val, m, device)
    best, best_state, rng = np.inf, None, np.random.default_rng(3)
    for step in range(steps):
        m.train()
        zn, prev, c, p, y = l2a_batch(lab, rng.choice(train, min(256, len(train))), m, device)
        loss = ((m(zn + 0.02 * torch.randn_like(zn)) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            m.eval()
            with torch.no_grad():
                vl = float(((m(vb[0]) - vb[4]) ** 2).mean())
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in m.state_dict().items()}
            log(f"L2A step {step}: train {float(loss):.4f} val {vl:.4f}")
    m.load_state_dict(best_state)
    return m.eval()


def tensors(lab, rows, g, device):
    """(u, ctx, pooled, z normalised) tensors of some windows."""
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    u = (t(lab["act"][rows]) - g.act_mean) / g.act_std
    zn = (t(lab["z"][rows]) - g.z_mean) / g.z_std
    return u, t(lab["ctx"][rows]), t(lab["pooled"][rows]), zn


def fit_z(g, lab, train, val, device, steps, log):
    """Regress the encoder's latents (MSE in prior-normalised units); keeps the best validation state."""
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vt = tensors(lab, val, g, device)
    best, best_state = np.inf, None
    rng = np.random.default_rng(0)
    for step in range(steps):
        g.train()
        u, c, p, zn = tensors(lab, rng.choice(train, min(256, len(train))), g, device)
        u = u + 0.02 * torch.randn_like(u)
        loss = ((g(u, c, p) - zn) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            g.eval()
            with torch.no_grad():
                vl = float(((g(*vt[:3]) - vt[3]) ** 2).mean())
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in g.state_dict().items()}
            log(f"z-fit step {step}: train {float(loss):.4f} val {vl:.4f}")
    g.load_state_dict(best_state)
    return best


class Windows:
    """Play windows re-read from the recordings (cached), for decoding."""

    def __init__(self, play_dir, lab):
        self.dir, self.lab, self.cache = Path(play_dir), lab, {}

    def get(self, i):
        f = str(self.lab["rec_file"][i])
        if f not in self.cache:
            self.cache[f] = dict(np.load(self.dir / f))
        rec = self.cache[f]
        frames, _, _ = window(rec, int(self.lab["start"][i]))
        seg = rec["segment"]
        if str(rec.get("scene_kind", "privileged")) == "production":  # camera points are re-measured every frame (no row correspondence): only the FK hand rows are tracked, so only they are scored
            seg = np.where(seg == 0, 0, -1)
        return frames, seg


def decode_window(model, frames, z):
    """Predicted cumulative displacements (5, N, 3) torch of latents z (5, nq, zd) torch on the window's frame 0, with gradients to z; and the true ones."""
    device = z.device
    coords = frames[0]
    chain = model_adapter.encode_cloud(model, coords)
    with torch.no_grad():
        queries = model.decoder.point_queries(chain)
    steps = model.decode_all_steps(None, z.unsqueeze(1), queries)
    pred = torch.stack([s["delta"].squeeze(1) for s in steps], 0)
    true = torch.as_tensor(frames[1:] - frames[0][None], dtype=torch.float32, device=device)
    return pred, true


def hand_obj_error(pred, true, segment):
    """Mean point error (m) over the 5 steps on hand points, and on object points."""
    err = torch.linalg.norm(pred - true, dim=-1)  # (5, N)
    seg = torch.as_tensor(segment, device=pred.device)
    return err[:, seg == 0].mean(), err[:, seg == 1].mean() if (seg == 1).any() else torch.zeros((), device=pred.device)


def fit_decoder(g, model, wins, lab, train, device, steps, batch, log):
    """Fine-tune g through the frozen decoder on the true hand and object displacements, keeping it near the encoder's latents."""
    opt = torch.optim.AdamW(g.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(1)
    for step in range(steps):
        g.train()
        opt.zero_grad()
        total = 0.0
        for i in rng.choice(train, batch):
            frames, seg = wins.get(i)
            u, c, p, zn = tensors(lab, [i], g, device)
            zn_hat = g(u, c, p)
            pred, true = decode_window(model, frames, (zn_hat * g.z_std + g.z_mean)[0])
            eh, eo = hand_obj_error(pred, true, seg)
            loss = (eh + 0.5 * eo) * 1000.0 / batch + 0.1 * ((zn_hat - zn) ** 2).mean() / batch
            loss.backward()
            total += float(loss)
        opt.step()
        if step % 25 == 0:
            log(f"decoder-fit step {step}: loss {total:.3f}")


@torch.no_grad()
def evaluate(g, model, wins, lab, rows, device, prior_mean):
    """Held-out decoded errors (mm, hand and object; and hand at step 1) for g's latents, the encoder's (ceiling), the prior mean, and zero motion (floor)."""
    acc = {k: [] for k in ("g", "encoder", "prior_mean", "zero")}
    for i in rows:
        frames, seg = wins.get(i)
        u, c, p, _ = tensors(lab, [i], g, device)
        cands = dict(g=g.latents(u, c, p)[0], encoder=torch.as_tensor(lab["z"][i], device=device), prior_mean=torch.as_tensor(prior_mean, device=device))
        for k, z in cands.items():
            pred, true = decode_window(model, frames, z.float())
            acc[k].append(_errs(pred, true, seg))
        acc["zero"].append(_errs(torch.zeros_like(true), true, seg))
    return {k: dict(zip(("hand_mm", "object_mm", "hand_step1_mm", "hand_step5_mm"), np.mean(v, 0).round(3).tolist())) for k, v in acc.items()}


def _errs(pred, true, segment):
    err = torch.linalg.norm(pred - true, dim=-1) * 1000.0
    seg = torch.as_tensor(segment, device=pred.device)
    h = err[:, seg == 0]
    o = err[:, seg == 1] if (seg == 1).any() else torch.zeros_like(h)
    return [float(h.mean()), float(o.mean()), float(h[0].mean()), float(h[-1].mean())]


def main():
    """python -m lawm.sim.dexwm.planspace.train_g --labels labels.npz --play DIR --out DIR [--z-steps 3000] [--dec-steps 300]."""
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--play", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--z-steps", type=int, default=3000)
    p.add_argument("--dec-steps", type=int, default=300)
    p.add_argument("--dec-batch", type=int, default=8)
    p.add_argument("--eval-windows", type=int, default=300)
    p.add_argument("--fake-backbone", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_g.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(0)
    device = torch.device("cpu" if a.fake_backbone else "cuda")
    if not a.fake_backbone:
        wait_for_gpu(2500, log)
    model = load(a.fake_backbone)
    for prm in model.parameters():
        prm.requires_grad_(False)
    lab = dict(np.load(a.labels))
    train, val = np.flatnonzero(lab["split"] == "train"), np.flatnonzero(lab["split"] == "val")
    assert len(train) and len(val), "need training and held-out windows"
    if a.fake_backbone:
        z_mean, z_std = lab["z"][train].mean(0), lab["z"][train].std(0)
    else:
        z_mean, z_std = model_adapter.load_prior(PRIOR, CHECKPOINT)
    st = stats_of(lab, train, z_mean, z_std)
    g = ActionToLatent(lab["pooled"].shape[1], lab["z"].shape[1:], stats=st).to(device)
    log(f"{len(train)} train / {len(val)} val windows")
    fit_z(g, lab, train, val, device, a.z_steps, log)
    wins = Windows(a.play, lab)
    rows = np.random.default_rng(2).choice(val, min(a.eval_windows, len(val)), replace=False)
    g.eval()
    rep_z = evaluate(g, model, wins, lab, rows, device, z_mean)
    log(f"after z-fit: {rep_z}")
    g.save(out / "g_zfit.pt", extra=dict(report=rep_z))
    chosen, rep = "zfit", rep_z
    if a.dec_steps > 0:
        fit_decoder(g, model, wins, lab, train, device, a.dec_steps, a.dec_batch, log)
        g.eval()
        rep_d = evaluate(g, model, wins, lab, rows, device, z_mean)
        log(f"after decoder-fit: {rep_d}")
        g.save(out / "g_decfit.pt", extra=dict(report=rep_d))
        if rep_d["g"]["hand_mm"] < rep_z["g"]["hand_mm"]:
            chosen, rep = "decfit", rep_d
    best = g.load(out / f"g_{chosen}.pt", device)
    best.save(out / "g.pt", extra=dict(report=rep, chosen=chosen))
    ratio = rep["g"]["hand_mm"] / max(rep["encoder"]["hand_mm"], 1e-9)
    passed = bool(ratio <= GATE_RATIO)
    write_json(out / "g_report.json", dict(chosen=chosen, heldout_windows=len(rows), after_zfit=rep_z, final=rep, g_over_zero_hand=round(rep["g"]["hand_mm"] / max(rep["zero"]["hand_mm"], 1e-9), 3),
                                           g_over_encoder_hand=round(ratio, 3), gate_ratio=GATE_RATIO, passed=passed))
    log(f"gate: A2L/encoder hand error = {ratio:.3f} (pass <= {GATE_RATIO}) -> {'PASS' if passed else 'FAIL'}")
    l2a = fit_l2a(lab, train, val, st, device, a.z_steps, log)
    l2a_rep = l2a_errors(l2a, lab, val, device)
    l2a.save(out / "l2a.pt", extra=dict(report=l2a_rep))
    write_json(out / "l2a_report.json", dict(heldout_windows=len(val), heldout_step_errors_rms=l2a_rep))
    log(f"L2A held-out: {l2a_rep}")


if __name__ == "__main__":
    main()
