"""Per-step maps between hand actions and LAWM latents, as in LAC-WM (Huang et al., ICML 2026): action-to-latent (A2L, 2-layer MLP, action only) and latent-to-action (L2A, 2-layer MLP, latent only)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from lawm.sim.dexwm.planspace.common import ACT_DIM, N_STEPS

CTX_DIM = 28  # q (16), wrist rotation 6D (6), wrist position from the cloud mean (3), object centroid from the wrist (3)


def state_features(root_pose, q, coords, segment):
    """(28,) hand-state features of one observation, in the frame of the centred cloud the model sees."""
    center = coords.mean(0)
    rot6 = root_pose[:3, :3][:, :2].T.ravel()
    obj = coords[segment == 1].mean(0) if (segment == 1).any() else center
    return np.concatenate([q, rot6, root_pose[:3, 3] - center, obj - root_pose[:3, 3]]).astype(np.float32)


def pooled_feature(chain):
    """(C,) mean of the coarsest backbone level of an encoded frame-0 cloud."""
    return chain[0].feat.float().mean(0)


def window_actions(root_pose_s, q_s, cmd_pos, cmd_rot, cmd_q):
    """(5, 22) actions of a window: per model step the commanded wrist translation (world frame), wrist rotation vector (world frame) and finger change, relative to the window's start state."""
    R_s, p_s = root_pose_s[:3, :3], root_pose_s[:3, 3]
    dpos = cmd_pos - p_s
    drot = Rotation.from_matrix(cmd_rot @ R_s.T).as_rotvec()
    return np.concatenate([dpos, drot, cmd_q - q_s], axis=1).astype(np.float32)


def step_inputs(a):
    """Per-step split of window actions a (..., 5, 22), which are relative to the window's start state: (state reached before step h relative to the start, i.e. actions 1..h-1 applied; the step's own increment)."""
    prev = torch.cat([torch.zeros_like(a[..., :1, :]), a[..., :-1, :]], dim=-2)
    return prev, a - prev


def mlp(din, dout, hidden, layers):
    """Linear -> LayerNorm -> SiLU blocks and a linear head."""
    dims, mods = [din] + [hidden] * layers, []
    for i in range(layers):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.SiLU()]
    return nn.Sequential(*mods, nn.Linear(hidden, dout))


STAT_NAMES = ("act_mean", "act_std", "ctx_mean", "ctx_std", "pooled_mean", "pooled_std", "z_mean", "z_std", "prev_mean", "prev_std", "inc_mean", "inc_std")


def _stat_shapes(pooled_dim, z_shape):
    return ((N_STEPS, ACT_DIM), (N_STEPS, ACT_DIM), (CTX_DIM,), (CTX_DIM,), (pooled_dim,), (pooled_dim,), tuple(z_shape), tuple(z_shape),
            (ACT_DIM,), (ACT_DIM,), (ACT_DIM,), (ACT_DIM,))


class _PerStep(nn.Module):
    """Shared buffers (normalisation statistics) and step features of the two per-step maps."""

    def _init_stats(self, pooled_dim, z_shape, stats):
        for n, s in zip(STAT_NAMES, _stat_shapes(pooled_dim, z_shape)):
            v = torch.as_tensor(stats[n], dtype=torch.float32) if stats is not None else (torch.zeros(s) if "mean" in n else torch.ones(s))
            self.register_buffer(n, v.reshape(s).clone())

    def _context(self, ctx, pooled, prev, n_steps):
        """Per step (B, H, .): the start-state features, the pooled scene feature and the state reached before the step (relative to the start), normalised."""
        B = ctx.shape[0]
        c = ((ctx - self.ctx_mean) / self.ctx_std).unsqueeze(1).expand(B, n_steps, -1)
        p = ((pooled - self.pooled_mean) / self.pooled_std).unsqueeze(1).expand(B, n_steps, -1)
        return torch.cat([c, p, (prev - self.prev_mean) / self.prev_std], dim=-1)

    def save(self, path, extra=None):
        """Store weights and config."""
        torch.save(dict(state=self.state_dict(), config=self.config, extra=extra or {}), path)

    @classmethod
    def load(cls, path, device):
        """Load a stored map."""
        blob = torch.load(path, map_location=device, weights_only=False)
        m = cls(**blob["config"])
        m.load_state_dict(blob["state"])
        return m.to(device).eval()


class ActionToLatent(_PerStep):
    """Action-to-latent map (A2L), per step, as LAC-WM's action projector: a 2-layer MLP from one step's action (the 22-D increment) to that step's latent; no state, no scene.
    One network for every step, so any horizon works; the planner's chunk interface (B, H, 22) normalised chunk actions -> normalised latents (B, H, nq, zd) is kept (ctx, pooled unused)."""

    def __init__(self, pooled_dim, z_shape, hidden=512, layers=1, stats=None):
        """stats: dict of numpy arrays named as STAT_NAMES."""
        super().__init__()
        self.z_shape = tuple(z_shape)
        self.net = mlp(ACT_DIM, int(np.prod(self.z_shape[1:])), hidden, layers)
        self.config = dict(pooled_dim=pooled_dim, z_shape=list(z_shape), hidden=hidden, layers=layers)
        self._init_stats(pooled_dim, z_shape, stats)

    def forward(self, u, ctx=None, pooled=None):
        """u (B, H, 22) normalised chunk actions (relative to the start state); returns normalised z (B, H, nq, zd)."""
        B, H = u.shape[:2]
        _, inc = step_inputs(u * self.act_std[:H] + self.act_mean[:H])
        return self.net((inc - self.inc_mean) / self.inc_std).reshape(B, H, *self.z_shape[1:])

    def latents(self, u, ctx=None, pooled=None):
        """Un-normalised latents (B, 5, nq, zd) for the decoder."""
        H = u.shape[1]
        return self(u, ctx, pooled) * self.z_std[:H] + self.z_mean[:H]


class LatentToAction(_PerStep):
    """Latent-to-action map (L2A), per step, as LAC-WM's motion decoder minus the image conditioning (our latents are 3D, not viewpoint dependent): a 2-layer MLP from one step's latent to that step's action (22-D increment)."""

    def __init__(self, pooled_dim, z_shape, hidden=512, layers=1, stats=None):
        super().__init__()
        self.z_shape = tuple(z_shape)
        self.net = mlp(int(np.prod(self.z_shape[1:])), ACT_DIM, hidden, layers)
        self.config = dict(pooled_dim=pooled_dim, z_shape=list(z_shape), hidden=hidden, layers=layers)
        self._init_stats(pooled_dim, z_shape, stats)

    def forward(self, zn):
        """zn (B, H, nq, zd) normalised latents; returns normalised increments (B, H, 22)."""
        B, H = zn.shape[:2]
        return self.net(zn.reshape(B, H, -1))

    def increments(self, zn):
        """Raw step increments (B, H, 22)."""
        return self(zn) * self.inc_std + self.inc_mean
