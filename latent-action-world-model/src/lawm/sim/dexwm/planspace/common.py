"""Shared paths and helpers of the planspace experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DEXWM_CONFIG = REPO / "configs/stage3/dexwm.json"
EVAL_MANIFEST = REPO / "configs/stage3/dexwm_episodes.json"
TUNING_DEMOS = ("demo_1368", "demo_1369", "demo_1370", "demo_1371", "demo_1372")
PLAY_SOURCES = ("finetune_scenes", "tuning")  # open_close demos use the RobotHoldPos env, which this robosuite does not register
CHECKPOINT = Path(os.environ.get("PLANSPACE_CHECKPOINT", REPO / "checkpoints/adaln_s0_k4/best.pt"))  # the original; a fine-tune is chosen with PLANSPACE_CHECKPOINT, never written over it
RUN_MANIFEST = Path("/vol/dissolve/vinay/lawm-deployment/checkpoint/run_manifest.json")
PRIOR = Path(os.environ.get("PLANSPACE_PRIOR", "/vol/dissolve/vinay/work/priors/adaln_s0_k4_h2o_subject1.npz"))  # the latent prior of CHECKPOINT (it records the checkpoint hash)

CONTROL_PER_MODEL_STEP = 2  # 10 Hz control, 0.2 s model step
N_STEPS = 5
ACT_DIM = 22  # per model step: wrist dpos (3, m, world frame), wrist drot (3, rotvec), finger dq (16)
TRAINING_UP = (-0.022, -0.546, -0.837)  # "up" (table towards hands) in the training data's frame, as in lawm-deployment configs/sim.yaml
SPIN_DEG = 315.0  # world horizontal direction mapped to the training x axis, as in lawm-deployment


def model_frame(training_up=TRAINING_UP, spin_deg=SPIN_DEG):
    """Rotation (3, 3) from the world frame (z up) to the training data's frame (lawm_deploy.model.model_frame): z goes to training_up, the horizontal direction at spin_deg to the training x axis."""
    up = np.asarray(training_up, float) / np.linalg.norm(training_up)
    x = np.array([1.0, 0.0, 0.0]) - up * up[0]
    x /= np.linalg.norm(x)
    t = np.radians(spin_deg)
    h = np.array([np.cos(t), np.sin(t), 0.0])
    return np.stack([x, np.cross(up, x), up], 1) @ np.stack([h, np.cross([0.0, 0.0, 1.0], h), [0.0, 0.0, 1.0]], 1).T


R_MODEL = model_frame().astype(np.float32)  # clouds go to the model as coords @ R_MODEL.T; predicted displacements come back as delta @ R_MODEL


def data_root():
    """LAWM_DATA_ROOT."""
    return Path(os.environ["LAWM_DATA_ROOT"])


def eval_demos():
    """The 50 DexWM evaluation demos, in manifest order."""
    return [e["demo"] for e in json.loads(EVAL_MANIFEST.read_text())["episodes"]]


def eval_dir():
    """Folder of the extracted evaluation demos."""
    return data_root() / json.loads(EVAL_MANIFEST.read_text())["episodes_dir"]


def play_scenes():
    """(folder, demo) of every non-evaluation DexWM demo usable as a play kitchen, de-duplicated; asserts none is an evaluation demo."""
    seen, out = set(), []
    for sub in PLAY_SOURCES:
        folder = data_root() / "stage3/dexwm" / sub
        for d in sorted(p.name for p in folder.iterdir() if p.name.startswith("demo_")):
            if d not in seen:
                seen.add(d)
                out.append((folder, d))
    clash = seen & set(eval_demos())
    assert not clash, f"play scenes overlap the DexWM evaluation set: {sorted(clash)}"
    return out


def write_json(path, obj):
    """Write JSON atomically."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
    tmp.replace(path)


def hold_action(obs):
    """The controller action that keeps the hand where it is."""
    from lawm.sim.dexwm.control.executor import ACTION_DIM
    from lawm.sim.dexwm.control.hand_fit import hand_state_to_action
    pos, rv, f = hand_state_to_action(obs.root_pose[:3, :3], obs.root_pose[:3, 3], obs.q, obs.root_pose[:3, :3], obs.root_pose[:3, 3], obs.ref_pos, obs.ref_rot)
    a = np.zeros(ACTION_DIM, np.float32)
    a[0:3], a[3:6], a[15:31] = pos, rv, f
    return a


def wait_for_gpu(need_mb, log=print):
    """Block until the GPU has need_mb free (polls every 60 s); the GPU is shared with other users' jobs."""
    import time
    import torch
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free / 2**20 >= need_mb:
            return
        log(f"waiting for GPU memory: {free / 2**20:.0f} MB free, need {need_mb}")
        time.sleep(60)
