"""Model adapters with the same interface as lawm.sim.model_adapter.ModelAdapter (prior, decoder(coords, segment) -> decode(candidates)): the action-space one maps normalised Allegro action chunks through g to latents; both count world-model decodes and fall back to smaller decode chunks when the shared GPU runs out of memory."""
from __future__ import annotations

import numpy as np
import torch

from lawm.sim import model_adapter as ma
from lawm.sim.dexwm.planspace.common import ACT_DIM, N_STEPS, R_MODEL
from lawm.sim.dexwm.planspace.g_model import pooled_feature, state_features


class _Decoding:
    """Chunked decoding of one encoded cloud that halves the chunk size on CUDA out-of-memory."""

    def __init__(self, model, chain, n_points, copies):
        self.model, self.chain, self.n, self.copies = model, chain, n_points, copies
        self.queries = ma.repeated_queries(model, chain, copies)

    def __call__(self, z):
        while True:
            try:
                return ma.decode_chunked(self.model, self.queries, z, self.n, self.copies)
            except torch.OutOfMemoryError:
                if self.copies == 1:
                    raise
                self.copies //= 2
                self.queries = None
                torch.cuda.empty_cache()
                self.queries = ma.repeated_queries(self.model, self.chain, self.copies)
                print(f"[adapter] CUDA OOM: decode chunk -> {self.copies}", flush=True)


class LatentAdapter:
    """The existing latent planner's adapter, with a decode counter and the OOM fallback."""

    def __init__(self, model, prior, copies=ma.CHUNK):
        self.model, self.prior, self.copies = model, prior, copies
        self.wm_calls = 0
        self._seen = {}

    def set_obs(self, obs, limits):
        self._seen = {}

    def decoder(self, coords, segment):
        dec = _Decoding(self.model, ma.encode_cloud(self.model, np.asarray(coords, np.float32) @ R_MODEL.T), len(coords), self.copies)

        def decode(z):
            self.wm_calls += len(z)
            out = (dec(z) @ R_MODEL).astype(np.float32)  # back to the world frame
            self.copies = dec.copies
            for i in range(len(out)):
                self._seen[_key(out[i])] = np.asarray(z[i], np.float32).copy()
            return out
        return decode

    def chosen_latent(self, delta):
        """The latent chunk (n_steps, nq, zd), in the prior's raw units, whose decoded displacements are `delta` (the plan the CEM kept)."""
        return self._seen[_key(delta)]


class ActionAdapter:
    """CEM over normalised action chunks u (5, 22) with prior N(0, 1), finger targets clipped to the joint limits, z = g(a), decoded by LAWM.
    inc_prior = (mean, std) (5, 22) of the per-step increments (action_prior.py): u parameterises each step's increment, inc = u * std + mean,
    and the chunk is their running sum (realistic ~2 cm steps). Without it (old behaviour, superseded): u parameterises the cumulative chunk directly,
    a = u * act_std + act_mean, whose independent per-step draws give unrealistic 10-15 cm steps."""

    def __init__(self, model, g, copies=ma.CHUNK, inc_prior=None):
        self.model, self.g, self.copies = model, g, copies
        self.inc_prior = None if inc_prior is None else tuple(np.asarray(x, np.float32) for x in inc_prior)
        self.prior = (np.zeros((N_STEPS, ACT_DIM), np.float32), np.ones((N_STEPS, ACT_DIM), np.float32))
        self.wm_calls = 0
        self.obs = self.limits = None
        self._seen = {}

    def set_obs(self, obs, limits):
        """The observation of this replan (hand state) and the finger joint limits."""
        self.obs, self.limits = obs, np.asarray(limits, np.float32)
        self._seen = {}

    def actions(self, u):
        """Raw actions (n, 5, 22) of normalised candidates, with q_now + dq inside the limits."""
        g = self.g
        if self.inc_prior is not None:
            m, s = self.inc_prior
            a = np.cumsum(u * s + m, axis=-2)
        else:
            a = u * g.act_std.cpu().numpy() + g.act_mean.cpu().numpy()
        q = self.obs.q.astype(np.float32)
        a[..., 6:] = np.clip(q + a[..., 6:], self.limits[:, 0], self.limits[:, 1]) - q
        return a.astype(np.float32)

    def decoder(self, coords, segment):
        chain = ma.encode_cloud(self.model, np.asarray(coords, np.float32) @ R_MODEL.T)
        dev = next(self.model.parameters()).device
        pooled = pooled_feature(chain).unsqueeze(0)
        ctx = torch.as_tensor(state_features(self.obs.root_pose, self.obs.q, coords, segment), device=dev).unsqueeze(0)
        dec = _Decoding(self.model, chain, len(coords), self.copies)
        g = self.g

        def decode(u):
            a = self.actions(np.asarray(u, np.float32))
            with torch.no_grad():
                un = (torch.as_tensor(a, device=dev) - g.act_mean) / g.act_std
                z = g.latents(un, ctx.expand(len(a), -1), pooled.expand(len(a), -1)).cpu().numpy().astype(np.float32)
            self.wm_calls += len(z)
            out = (dec(z) @ R_MODEL).astype(np.float32)  # back to the world frame
            self.copies = dec.copies
            for i in range(len(out)):
                self._seen[_key(out[i])] = a[i]
            return out
        return decode

    def chosen_action(self, delta):
        """The raw action chunk (5, 22) whose decoded displacements are `delta` (the plan the CEM kept)."""
        return self._seen[_key(delta)]


def _key(delta):
    """An exact fingerprint of one candidate's decoded displacements."""
    flat = np.ascontiguousarray(delta).ravel()
    return flat[:: max(1, len(flat) // 64)].tobytes()
