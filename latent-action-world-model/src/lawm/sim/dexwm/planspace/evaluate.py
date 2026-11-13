"""Closed-loop reach / grasp runs on the selected episodes: A1 = action search, executed through hand_fit of the predicted cloud (as the latent planner); A2 = action search, the chosen action executed directly; L = the current latent planner;
L2A = latent search, the chosen first-step latent mapped to an action by the latent-to-action map and executed directly.
Never overwrites: an episode is done once <demo>_<task>.json exists (written last, atomically) and is then skipped; a fresh <demo>_<task>.lock (touched every replan) keeps a second worker off a running episode; arm.json must match the run's settings. Same TaskPlanner, cost, gates, loop stops and executor as lawm.sim.dexwm.runner."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lawm.sim import model_adapter as ma
from lawm.sim.control import StallDetector
from lawm.sim.costs import SEG_HAND, SEG_OBJECT
from lawm.sim.dexwm.control.executor import Executor
from lawm.sim.dexwm.env import benchmark, kitchen
from lawm.sim.dexwm.planning.plan_costs import CostWeights
from lawm.sim.dexwm.planning.task_planner import TaskPlanner
from lawm.sim.dexwm.planspace.adapter import ActionAdapter, LatentAdapter
from lawm.sim.dexwm.planspace.common import CHECKPOINT, DEXWM_CONFIG, PRIOR, RUN_MANIFEST, eval_dir, wait_for_gpu, write_json
from lawm.sim.dexwm.planspace.g_model import ActionToLatent, LatentToAction
from lawm.sim.dexwm.runner import episode_seed, load_config

ARMS = dict(A1=("action", "fit"), A2=("action", "direct"), L=("latent", "fit"), L2A=("latent", "l2a"))
TASKS = ("reach", "grasp", "place")
# Place under partial observability: one stage, no secure gate, no grasp-quality / contact terms. The predicted object points to the observed goal-object points (one-sided
# Chamfer; object from object_track.py, no segmentation model), a penalty when the object moves away from the hand, and a low-weight hand goal (the goal-frame hand rows).
PLACE_SIMPLE = dict(hand_scale=0.25, fingertip=1.0, object_still=0.0, object_goal=1.0, grasp_quality=0.0, rigid_attachment=0.0, grasp_tight=0.0, attachment=1.0)
LOCK_STALE_S = 20 * 60  # a lock untouched this long belongs to a dead worker
SETTINGS_KEYS = ("arm", "task", "search_space", "execution", "cem", "loop", "exec", "checkpoint", "prior", "g", "l2a", "action_prior", "scene", "weights", "segmentation")


def direct_state(adapter, plan, obs):
    """(root rotation, root position, q) target of the chosen action's first step (hold: the current state)."""
    R_now, p_now = obs.root_pose[:3, :3], obs.root_pose[:3, 3]
    if plan.executed == "hold":
        return R_now, p_now, obs.q.copy()
    a = adapter.chosen_action(plan.delta)[0]
    return Rotation.from_rotvec(a[3:6]).as_matrix() @ R_now, p_now + a[:3], obs.q + a[6:]


def l2a_state(adapter, l2a, plan, obs, limits):
    """(root rotation, root position, q) target of the chosen latent's first step mapped through L2A (hold: the current state); fingers clipped to the joint limits."""
    R_now, p_now = obs.root_pose[:3, :3], obs.root_pose[:3, 3]
    if plan.executed == "hold":
        return R_now, p_now, obs.q.copy()
    z = torch.as_tensor(adapter.chosen_latent(plan.delta)[None, :1], device=l2a.z_mean.device)
    with torch.no_grad():
        a = l2a.increments((z - l2a.z_mean[:1]) / l2a.z_std[:1])[0, 0].cpu().numpy().astype(np.float64)
    lim = np.asarray(limits, np.float64)
    q = np.clip(obs.q + a[6:], lim[:, 0], lim[:, 1])
    return Rotation.from_rotvec(a[3:6]).as_matrix() @ R_now, p_now + a[:3], q


def run_episode(sim, planner, adapter, execution, exec_settings, settings, on_step=None, l2a=None, heartbeat=None):
    """lawm.sim.dexwm.closed_loop.run_episode with the execution switch and per-replan timing, decode counts and fit residuals; on_step() after every control step (video)."""
    obs = sim.observe()
    executor = Executor(exec_settings, Rotation.from_matrix(obs.ref_rot).as_rotvec())
    stall = None if settings.stall_patience is None else StallDetector(settings.stall_patience, settings.stall_tol_mm)
    streak, ok, records = 0, False, []
    for replan in range(settings.max_replans):
        if heartbeat is not None:
            heartbeat()
        adapter.set_obs(obs, sim.kinematics.limits)
        calls0, t0 = adapter.wm_calls, time.time()
        hand, plan = planner.plan(obs.coords, replan, getattr(sim, "obs_segment", None))
        t_plan = time.time() - t0
        fit_res = None
        if execution == "fit":
            root_rot, root_pos, q, info = sim.kinematics.fit(hand.astype(np.float64), obs.q)
            fit_res = info["mean_residual_mm"]
        else:
            root_rot, root_pos, q = direct_state(adapter, plan, obs) if execution == "direct" else l2a_state(adapter, l2a, plan, obs, sim.kinematics.limits)
        action, commanded_mm = executor.command((root_rot, root_pos, q), obs.root_pose, obs.ref_pos, obs.ref_rot)
        for _ in range(exec_settings.raw_steps):
            ok = bool(sim.step(action))
            if on_step is not None:
                on_step()
            streak = streak + 1 if ok else 0
            if streak >= settings.sustain_steps:
                break
        error_mm = float(sim.distance() * 1000.0)
        records.append(dict(replan=replan, error_mm=error_mm, commanded_mm=commanded_mm, executed=plan.executed, plan_cost=plan.cost,
                            predicted_error_mm=plan.final_hand_error_mm, streak=streak, plan_seconds=round(t_plan, 3), wm_calls=adapter.wm_calls - calls0,
                            fit_residual_mm=fit_res, **getattr(sim, "obs_info", {})))
        reason = "success" if streak >= settings.sustain_steps else "converged" if error_mm < settings.converge_mm else "stalled" if stall and stall.update(error_mm) else None
        if reason:
            return dict(success=streak >= settings.sustain_steps, holds_at_end=ok, stop_reason=reason, n_replans=replan + 1, records=records)
        obs = sim.observe()
        records[-1]["phase_changed"] = planner.update(obs.coords, replan)
    return dict(success=False, holds_at_end=ok, stop_reason="max_replans", n_replans=settings.max_replans, records=records)


def production_scene(sim, demo, log, place=False, task=None):
    """Switch the episode to the production scene: sim.observe() then returns the FK hand + camera cloud; the goal is the target hand pose (FK hand rows of the task's target) with the start camera scene (unused by the reach cost).
    place: the held object without any segmentation model (object_track.py): identified once from the start and target frames of the same cameras (+ FK hand/robot), then
    transported (held / released) to every observation; the scene rows near the object points are SEG_OBJECT, the rest SEG_REST (sim.obs_segment, the rows change between
    observations); returns (goal, segment, prod, goal_object_points) with the observed goal-object points for the one-sided Chamfer goal term."""
    from lawm.sim.dexwm.planspace.production import ProductionScene
    prod = ProductionScene(sim, seed=int(demo.split("_")[1]))
    base = sim.observe
    obs = base()
    prod.set_start(obs.root_pose)
    coords, segment, views, seen = prod(sim.data, obs.root_pose, obs.q)
    hand = segment == SEG_HAND
    assert hand.sum() == (sim.segment == SEG_HAND).sum()
    log(f"production scene: {hand.sum()} FK hand + {(~hand).sum()} camera points ({seen} seen); FK vs mesh hand rows max {np.abs(coords[hand] - obs.coords[sim.segment == SEG_HAND]).max() * 1000:.2f} mm")
    goal = np.concatenate([sim.goal[sim.segment == SEG_HAND], coords[~hand]])
    if not place:
        def observe():
            o = base()
            o.coords = prod(sim.data, o.root_pose, o.q)[0]
            return o
        sim.observe = observe
        return goal, segment, prod
    from lawm.sim.dexwm.planspace import object_track as ot
    from lawm.sim.dexwm.planspace.production import back_project, voxel_down
    k = sim.kinematics
    root = k.bodies[k.root]
    log(f"camera alignment to the FK robot: median {prod.alignment_mm} mm per view (centre pose: {prod.centre_pose})")
    start_hand, start_robot = coords[hand].astype(np.float64), prod.robot(sim.data)
    start, target = benchmark.get_indices(sim.episode["waypoints"], task)
    sim.switch(target)
    R_t, p_t, q_t = sim.data.xmat[root].reshape(3, 3).copy(), sim.data.xpos[root].copy(), sim.data.qpos[k.qpos_adr].copy()
    goal_views, goal_hand, goal_robot = prod.views(sim.data), k.points(R_t, p_t, q_t), prod.robot(sim.data)
    true_target = sim.data.xpos[sim.object_id].copy()
    sim.switch(start)
    start_obj, goal_obj, info = ot.identify(prod.centred(views), prod.centred(goal_views), start_hand, start_robot, goal_hand, goal_robot, prod.lo, prod.hi)
    sim.place_ident = info
    sim.goal_object_found, sim.start_object_found = bool(len(goal_obj)), bool(len(start_obj))
    # the object as pixel masks -> back in the planner's (legacy-pose) frame: scene-row labels and the goal-object points come from the same back-projection as the cloud
    pix = lambda vs, obj: [ot.pixel_mask(v, obj, prod.lo, prod.hi) for v in prod.centred(vs)]
    goal_pts = np.zeros((0, 3))
    if len(goal_obj):
        both = [back_project(d, K, pose, mask=m) for (_, d, K, pose), m in zip(goal_views, pix(goal_views, goal_obj))]
        goal_pts = voxel_down(np.concatenate([pts[lab] for pts, lab in both]))
        sim.goal_object_err_mm = float(np.linalg.norm(goal_obj.mean(0) - true_target) * 1000)
    log(f"object (no segmentation model): start {len(start_obj)} pts ({info['start'].get('flag') or info['start']['source']}), goal {len(goal_obj)} pts "
        f"({info['goal'].get('flag') or info['goal']['source']}; centroid {getattr(sim, 'goal_object_err_mm', float('nan')):.0f} mm from the true target body, logged only)")
    tracker = ot.ObjectTracker(start_obj, start_hand, prod.lo, prod.hi, prod.centred(views), start_robot)
    coords, segment = prod(sim.data, obs.root_pose, obs.q, masks=lambda vs: pix(vs, start_obj))[:2]
    sim.obs_segment = segment
    goal = np.concatenate([sim.goal[sim.segment == SEG_HAND], coords[segment != SEG_HAND]])
    log(f"start: {(segment == SEG_OBJECT).sum()} object rows of {(segment != SEG_HAND).sum()} scene rows; goal object {len(goal_pts)} points")

    def observe():
        o = base()
        hand_pts = k.points(o.root_pose[:3, :3], o.root_pose[:3, 3], o.q)
        robot = prod.robot(sim.data)
        track = {}

        def masks(vs):
            obj, track["info"] = tracker.update(prod.centred(vs), hand_pts, robot)
            return pix(vs, obj)
        o.coords, sim.obs_segment = prod(sim.data, o.root_pose, o.q, masks=masks)[:2]
        ti = track["info"]
        n_obj = int((sim.obs_segment == SEG_OBJECT).sum())
        info = dict(n_object_points=n_obj, object_source=ti["source"], object_hypothesis=ti.get("hypothesis"), object_flag=ti.get("flag"))
        if n_obj:  # diagnostics only, never seen by the planner
            info["object_centroid_err_mm"] = float(np.linalg.norm(o.coords[sim.obs_segment == SEG_OBJECT].mean(0) - sim.data.xpos[sim.object_id]) * 1000)
        sim.obs_info = info
        return o
    sim.observe = observe
    return goal, segment, prod, goal_pts.astype(np.float64)


def main():
    """python -m lawm.sim.dexwm.planspace.evaluate --arm A1|A2|L|L2A --task reach|grasp --episodes episodes.json [--g a2l.pt] [--l2a l2a.pt] --out DIR [--demos ...] [--pop P --iters I] [--max-replans N]."""
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--episodes", required=True)
    p.add_argument("--task", default="reach", choices=TASKS)
    p.add_argument("--g", default=None, help="A2L map (arms A1, A2)")
    p.add_argument("--l2a", default=None, help="L2A map (arm L2A)")
    p.add_argument("--action-prior", default=None, help="per-step increment prior (action_prior.py) for the action-search arms")
    p.add_argument("--out", required=True)
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--max-replans", type=int, default=None)
    p.add_argument("--copies", type=int, default=ma.CHUNK)
    p.add_argument("--need-mb", type=int, default=2500)
    p.add_argument("--scene", choices=["production", "privileged"], default="production", help="production: FK hand + 2 RGB-D camera scene (no privileged scene state)")
    p.add_argument("--video", action="store_true", help="save <demo>_reach.mp4: both cameras at every control step")
    p.add_argument("--fake-backbone", action="store_true", help="CPU smoke test with the tests' stand-in backbone")
    a = p.parse_args()
    TASK = a.task
    space, execution = ARMS[a.arm]
    assert (a.g is not None) == (space == "action"), "--g is needed by (and only by) the action-search arms"
    assert (a.l2a is not None) == (execution == "l2a"), "--l2a is needed by (and only by) arm L2A"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [{a.arm}] {msg}", flush=True)

    cfg = load_config(DEXWM_CONFIG)
    cem = cfg.cem[TASK] if TASK != "place" else cfg.cem["grasp"]  # place: the same search budget as reach and grasp in these runs
    weights = CostWeights(**PLACE_SIMPLE) if TASK == "place" else cfg.weights[TASK]
    assert not isinstance(weights, (list, tuple)), "multi-stage (place) weights are not supported here"
    assert TASK != "place" or a.scene == "production", "place here is the production-scene (object_track.py) variant"
    if a.pop or a.iters:
        cem = type(cem)(a.pop or cem.pop, a.iters or cem.iters, cem.elite_frac, cem.min_std, cem.hold_candidate)
    loop = cfg.loop[TASK]
    if a.max_replans:
        loop = type(loop)(a.max_replans, loop.converge_mm, loop.stall_patience, loop.stall_tol_mm, loop.sustain_steps)
    episodes = json.loads(Path(a.episodes).read_text())
    assert episodes.get("task", "reach") == TASK, f"{a.episodes} holds {episodes.get('task', 'reach')} episodes, not {TASK}"
    assert episodes.get("done", True), f"{a.episodes}: selection not finished"
    demos = a.demos or episodes["selected"]
    settings = json.loads(json.dumps(dict(arm=a.arm, task=TASK, search_space=space, execution=execution, cem=cem.__dict__, loop=loop.__dict__, exec=cfg.exec.__dict__,
                                          checkpoint=str(CHECKPOINT), prior=str(PRIOR), g=a.g, l2a=a.l2a, action_prior=a.action_prior, scene=a.scene,
                                          weights=weights.__dict__, segmentation="geom_change_transport" if TASK == "place" else None)))
    arm_file = out / "arm.json"
    if arm_file.exists():
        old = json.loads(arm_file.read_text())
        old.setdefault("task", "reach")
        old.setdefault("l2a", None)
        old.setdefault("action_prior", None)
        old.setdefault("weights", cfg.weights[TASK].__dict__ if not isinstance(cfg.weights[TASK], (list, tuple)) else None)
        old.setdefault("segmentation", None)
        diff = [k for k in SETTINGS_KEYS if old.get(k) != settings[k]]
        assert not diff, f"{arm_file} was written with other settings ({diff}); refusing to mix results. Use another --out."
    else:
        write_json(arm_file, dict(settings, demos=demos))
    with open(out / "launches.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(), pid=os.getpid(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                                demos=demos)) + "\n")
    done = lambda d: (out / f"{d}_{TASK}.json").exists()
    todo = [d for d in demos if not done(d)]
    if not todo:
        log("all episodes done")
        return
    if a.fake_backbone:
        from lawm.sim.dexwm.planspace.labels import load
        device, model = torch.device("cpu"), load(True)
    else:
        wait_for_gpu(a.need_mb, log)
        device = torch.device("cuda")
        model = ma.load_model(CHECKPOINT, RUN_MANIFEST, device)
    if space == "latent":
        adapter = LatentAdapter(model, ma.load_prior(PRIOR, CHECKPOINT), a.copies)
    else:
        inc_prior = None
        if a.action_prior:
            pr = np.load(a.action_prior)
            inc_prior = (pr["inc_mean"], pr["inc_std"])
        adapter = ActionAdapter(model, ActionToLatent.load(a.g, device), a.copies, inc_prior)
    l2a = LatentToAction.load(a.l2a, device) if a.l2a else None
    place = TASK == "place"
    for demo in todo:
        lock = out / f"{demo}_{TASK}.lock"
        if done(demo) or not take_lock(lock, log):
            continue
        if done(demo):  # another worker finished it between the check and the lock
            lock.unlink(missing_ok=True)
            continue
        try:
            run_demo(a, demo, TASK, cfg, cem, loop, execution, adapter, l2a, out, log, lambda: lock.touch(), weights, place)
        finally:
            lock.unlink(missing_ok=True)


def take_lock(lock, log):
    """True if this worker now owns the episode: no lock, or a stale one (untouched for LOCK_STALE_S, its worker died)."""
    me = json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), slurm_job=os.environ.get("SLURM_JOB_ID"), time=time.strftime("%H:%M:%S")))
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < LOCK_STALE_S:
                log(f"{lock.name}: held by {lock.read_text()} ({age:.0f} s ago) -> skip")
                return False
            log(f"{lock.name}: stale ({age:.0f} s) -> take over")
            lock.unlink(missing_ok=True)
    return False


def run_demo(a, demo, TASK, cfg, cem, loop, execution, adapter, l2a, out, log, heartbeat, weights, place=False):
    """One episode; its mp4 and then its json are written atomically (tmp + rename). Place episodes whose object is not found still run (hand goal + whatever object
    term is available) and are flagged in their json: no episode is dropped."""
    t0 = time.time()
    sim = kitchen.open_demo(benchmark.load_episode(eval_dir(), demo), cfg.env)
    try:
        sim.reset(TASK, cfg.scene_seed)
        goal, segment = sim.goal, sim.segment
        frames, on_step = [], None
        goal_object_points = None
        if a.scene == "production" and place:
            goal, segment, prod, goal_object_points = production_scene(sim, demo, log, place=True, task=TASK)
            missing = [w for w, ok in (("start", sim.start_object_found), ("goal", sim.goal_object_found)) if not ok]
            if missing:
                log(f"!!! FLAG {demo}: no object found at the {' and '.join(missing)} frame -> runs without that part of the object cost (flagged in the json)")
        elif a.scene == "production":
            goal, segment, prod = production_scene(sim, demo, log)
        if a.scene == "production":
            if a.video:
                on_step = lambda: frames.append(np.concatenate([v[0] for v in prod.views(sim.data)], axis=1))
                on_step()
        seed = episode_seed(cfg.seed, demo, TASK)
        planner = TaskPlanner(TASK, goal, segment, sim.finger_mask, adapter, seed, cem, weights, cfg.gates, goal_object_points=goal_object_points)
        r = run_episode(sim, planner, adapter, execution, cfg.exec, loop, on_step, l2a, heartbeat)
    finally:
        sim.env.close()
    if frames:
        import imageio
        tmp = out / f".{demo}_{TASK}.tmp.mp4"
        imageio.mimsave(tmp, frames, fps=int(cfg.env.control_freq))
        os.replace(tmp, out / f"{demo}_{TASK}.mp4")
    extra = {}
    if place:
        extra = dict(start_object_found=getattr(sim, "start_object_found", None), goal_object_found=getattr(sim, "goal_object_found", None),
                     goal_object_err_mm=getattr(sim, "goal_object_err_mm", None), place_ident=getattr(sim, "place_ident", None))
    summary = dict(demo=demo, task=TASK, arm=a.arm, seed=seed, success=r["success"], **extra, holds_at_end=r["holds_at_end"], stop_reason=r["stop_reason"],
                   n_replans=r["n_replans"], final_error_mm=r["records"][-1]["error_mm"], wall_seconds=round(time.time() - t0, 1), records=r["records"])
    tmp = out / f".{demo}_{TASK}.tmp.json"
    write_json(tmp, summary)
    os.replace(tmp, out / f"{demo}_{TASK}.json")
    log(f"{demo}: success {r['success']} ({r['stop_reason']}), {r['n_replans']} replans, final {summary['final_error_mm']:.1f} mm, {summary['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
